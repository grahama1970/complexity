# src/pdf_extractor/arangodb/search_api/search_basic.py
"""
Basic search methods for Lessons Learned in ArangoDB:
 - Tags: INTERSECTION(AQL) + rapidfuzz in Python
 - Text: LIKE (AQL) + rapidfuzz partial_ratio in Python (OR vs AND)
 - Results sorted by match-count or summed similarity
"""

import os
import sys
import uuid
import json
from typing import List, Dict, Any, Tuple

from loguru import logger
from arango.database import StandardDatabase
from rapidfuzz import fuzz

# --- Load config -----------------------------------------------------
try:
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from pdf_extractor.arangodb.config import COLLECTION_NAME, SEARCH_FIELDS
    from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

    logger.debug(
        f"Loaded config: COLLECTION_NAME={COLLECTION_NAME}, SEARCH_FIELDS={SEARCH_FIELDS}"
    )
except ImportError:
    COLLECTION_NAME = os.environ.get("ARANGO_VERTEX_COLLECTION", "lessons_learned")
    SEARCH_FIELDS = ["problem", "solution", "context", "example"]
    logger.warning(
        f"Using fallback config: COLLECTION_NAME={COLLECTION_NAME}, SEARCH_FIELDS={SEARCH_FIELDS}"
    )


def find_lessons_by_tags_advanced(
    db: StandardDatabase,
    tag_keywords: List[str],
    limit: int = 10,
    match_all: bool = False,
    similarity_threshold: float = 97.0,
) -> List[Dict[str, Any]]:
    """
    Two-stage tag search:
      1) AQL: fetch docs where INTERSECTION(doc.tags, @keywords) > 0 (OR)
         or == len(@keywords) (AND)
      2) Python: fuzzy-match each keyword against doc.tags,
         count how many keywords matched (ratio ≥ threshold),
         drop zero-match docs, sort by match-count desc.
    """
    if not tag_keywords:
        return []

    # Stage 1: coarse AQL fetch
    op = "==" if match_all else ">"
    required = len(tag_keywords) if match_all else 0
    aql = f"""
    FOR doc IN {COLLECTION_NAME}
      FILTER LENGTH(INTERSECTION(doc.tags, @keywords)) {op} {required}
      LIMIT {limit * 5}
      RETURN doc
    """
    bind_vars = {"keywords": tag_keywords}
    try:
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        candidates = list(cursor)
    except Exception as e:
        logger.error(f"AQL error in tag stage: {e}")
        return []

    # Stage 2: Python fuzzy filtering & sorting
    def tag_match_count(doc: Dict[str, Any]) -> int:
        tags = doc.get("tags", [])
        cnt = 0
        for kw in tag_keywords:
            if any(
                fuzz.ratio(kw.lower(), str(t).lower()) >= similarity_threshold
                for t in tags
            ):
                cnt += 1
        return cnt

    scored = []
    for doc in candidates:
        cnt = tag_match_count(doc)
        if cnt > 0:
            scored.append((cnt, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:limit]]


def find_lessons_by_text_like(
    db: StandardDatabase,
    text_keywords: List[str],
    limit: int = 10,
    match_all: bool = False,
    similarity_threshold: float = 97.0,
) -> List[Dict[str, Any]]:
    """
    Two-stage text search:
      1) AQL: for OR, any field LIKE any keyword;
              for AND, every keyword must LIKE (across any field)
      2) Python:
           - AND: require fuzz.partial_ratio ≥ threshold for every keyword
           - OR:  require fuzz.partial_ratio ≥ threshold for at least one keyword
         Sum up only the passing scores for ranking.
    """
    # --- Stage 1: coarse AQL ---
    bind_vars: Dict[str, Any] = {}
    if text_keywords:
        clauses = []
        for i, kw in enumerate(text_keywords):
            var = f"kw{i}"
            bind_vars[var] = kw
            likes = [
                f"LOWER(doc.`{field}`) LIKE CONCAT('%', LOWER(@{var}), '%')"
                for field in SEARCH_FIELDS
            ]
            clause = "(" + " OR ".join(likes) + ")"
            clauses.append(clause)
        joiner = " AND " if match_all else " OR "
        filter_clause = "FILTER " + joiner.join(clauses)
    else:
        filter_clause = ""
    aql = f"""
    FOR doc IN {COLLECTION_NAME}
      {filter_clause}
      LIMIT {limit * 5}
      RETURN doc
    """
    try:
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        candidates = list(cursor)
    except Exception as e:
        logger.error(f"AQL error in text stage: {e}")
        return []

    # --- Stage 2: Python fuzzy filtering & scoring ---
    def keyword_score(doc: Dict[str, Any], kw: str) -> float:
        best = 0.0
        for field in SEARCH_FIELDS:
            val = doc.get(field)
            if isinstance(val, str):
                score = fuzz.partial_ratio(kw.lower(), val.lower())
                if score > best:
                    best = score
        return best

    scored = []
    for doc in candidates:
        scores = [keyword_score(doc, kw) for kw in text_keywords]
        if text_keywords:
            if match_all:
                # require every keyword ≥ threshold
                if any(s < similarity_threshold for s in scores):
                    continue
            else:
                # require at least one keyword ≥ threshold
                if all(s < similarity_threshold for s in scores):
                    continue
        total = sum(s for s in scores if s >= similarity_threshold)
        scored.append((total, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:limit]]

def validate_search_basic(
    tag_results: List[Dict[str, Any]], 
    text_results: List[Dict[str, Any]], 
    fixture_path: str
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate search_basic results against known good fixture data.
    
    Args:
        tag_results: Results from find_lessons_by_tags_advanced
        text_results: Results from find_lessons_by_text_like
        fixture_path: Path to the fixture file containing expected results
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    # Load fixture data
    try:
        with open(fixture_path, "r") as f:
            expected_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load fixture data: {e}")
        return False, {"fixture_loading_error": {"expected": "Valid JSON file", "actual": str(e)}}
    
    # Track all validation failures
    validation_failures = {}
    
    # Validate tag search results
    if "expected_tag_results" in expected_data:
        expected_tag_keys = expected_data["expected_tag_results"]
        actual_tag_keys = [doc.get("_key", "") for doc in tag_results]
        
        # Check count
        if len(expected_tag_keys) != len(actual_tag_keys):
            validation_failures["tag_result_count"] = {
                "expected": len(expected_tag_keys),
                "actual": len(actual_tag_keys)
            }
        
        # Check for missing expected keys
        expected_tag_set = set(expected_tag_keys)
        actual_tag_set = set(actual_tag_keys)
        
        if not expected_tag_set.issubset(actual_tag_set):
            missing_keys = expected_tag_set - actual_tag_set
            validation_failures["missing_tag_keys"] = {
                "expected": list(expected_tag_set),
                "actual": list(actual_tag_set),
                "missing": list(missing_keys)
            }
    
    # Validate text search results
    if "expected_text_results" in expected_data:
        expected_text_keys = expected_data["expected_text_results"]
        actual_text_keys = [doc.get("_key", "") for doc in text_results]
        
        # Check count
        if len(expected_text_keys) != len(actual_text_keys):
            validation_failures["text_result_count"] = {
                "expected": len(expected_text_keys),
                "actual": len(actual_text_keys)
            }
        
        # Check for missing expected keys
        expected_text_set = set(expected_text_keys)
        actual_text_set = set(actual_text_keys)
        
        if not expected_text_set.issubset(actual_text_set):
            missing_keys = expected_text_set - actual_text_set
            validation_failures["missing_text_keys"] = {
                "expected": list(expected_text_set),
                "actual": list(actual_text_set),
                "missing": list(missing_keys)
            }
    
    return len(validation_failures) == 0, validation_failures

# --- Standalone Verification Harness -------------------------------
if __name__ == "__main__":
    import sys
    from loguru import logger as _logger
    
    try:
        from pdf_extractor.arangodb.lessons import add_lesson, delete_lesson
    except ImportError:
        # Mock implementations for testing
        def add_lesson(db, lesson_data):
            return db.collection(COLLECTION_NAME).insert(lesson_data)
        
        def delete_lesson(db, key):
            return db.collection(COLLECTION_NAME).delete(key)
    
    # Configure logging
    _logger.remove()
    _logger.add(
        sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}"
    )
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/lessons_learned_expected.json"
    
    try:
        # 1) Set up database connection
        client = connect_arango()
        db = ensure_database(client)
        
        # 2) Fresh test collection with unique ID to avoid conflicts
        run_id = str(uuid.uuid4())[:8]
        test_coll = f"{COLLECTION_NAME}_tests_{run_id}"
        
        # Drop if exists and recreate
        try:
            db.delete_collection(test_coll)
        except Exception:
            pass
        
        db.create_collection(test_coll)
        
        # Store original collection name and update for testing
        orig_collection = COLLECTION_NAME
        COLLECTION_NAME = test_coll
        _logger.info(f"Using test collection: {COLLECTION_NAME}")
        
        # 3) Insert test documents
        TEST_DOCS = [
            {
                "_key": f"search_test_{run_id}_1",
                "problem": f"Unique problem alpha {run_id}",
                "solution": "Common solution.",
                "tags": ["search", "alpha", run_id],
                "role": "Searcher",
            },
            {
                "_key": f"search_test_{run_id}_2",
                "problem": f"Common problem {run_id}",
                "solution": "Unique solution beta.",
                "tags": ["search", "beta", run_id],
                "role": "Finder",
            },
            {
                "_key": f"search_test_{run_id}_3",
                "problem": "Another common problem.",
                "solution": "Common solution.",
                "tags": ["search", "alpha", "beta", run_id],
                "role": "Tester",
            },
        ]
        
        keys = []
        for doc in TEST_DOCS:
            meta = db.collection(COLLECTION_NAME).insert(doc)
            if meta and meta.get("_key"):
                keys.append(meta["_key"])
                _logger.success(f"Inserted: {meta['_key']}")
            else:
                _logger.error(f"Insert failed: {doc['_key']}")
                sys.exit(1)
        
        # 4) Run test searches
        tag_results = find_lessons_by_tags_advanced(db, ["alpha", "beta", run_id])
        text_results = find_lessons_by_text_like(db, [run_id, "beta"])
        
        # 5) Create test fixture if it doesn't exist
        try:
            with open(fixture_path, "r") as f:
                fixture_exists = True
        except FileNotFoundError:
            # Create a minimal fixture file with the test data
            expected_data = {
                "expected_tag_results": [
                    f"search_test_{run_id}_1",
                    f"search_test_{run_id}_2",
                    f"search_test_{run_id}_3"
                ],
                "expected_text_results": [
                    f"search_test_{run_id}_2"
                ]
            }
            
            with open(fixture_path, "w") as f:
                json.dump(expected_data, f, indent=2)
        
        # 6) Validate the results
        validation_passed, validation_failures = validate_search_basic(
            tag_results, text_results, fixture_path
        )
        
        # 7) Clean up
        for k in keys:
            delete_lesson(db, k)
        
        db.delete_collection(COLLECTION_NAME)
        _logger.info(f"Dropped test collection: {COLLECTION_NAME}")
        
        # Restore original collection name
        COLLECTION_NAME = orig_collection
        
        # 8) Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All search_basic results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - search_basic results don't match expected values") 
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
