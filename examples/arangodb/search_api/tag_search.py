# src/pdf_extractor/arangodb/search_api/tag_search.py
import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError

from pdf_extractor.arangodb.config import (
    COLLECTION_NAME,
    ALL_DATA_FIELDS_PREVIEW,
    TAG_ANALYZER
)
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

def tag_search(
    db: StandardDatabase,
    tags: List[str],
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    require_all_tags: bool = False,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Search for documents by tags.
    
    Args:
        db: ArangoDB database
        tags: List of tags to search for
        collections: Optional list of collections to search
        filter_expr: Optional AQL filter expression
        require_all_tags: Whether all tags must be present
        limit: Maximum number of results
        offset: Result offset for pagination
        
    Returns:
        Dictionary with search results
    """
    start_time = time.time()
    logger.info(f"Searching for documents with tags: {tags}")
    
    try:
        # Use default collection if not specified
        if not collections:
            collections = [COLLECTION_NAME]
        
        # Build filter clause based on tags
        tag_operator = " AND " if require_all_tags else " OR "
        tag_conditions = []
        
        for tag in tags:
            tag_conditions.append(f'@tag_{len(tag_conditions)} IN doc.tags')
        
        # Create tag filter
        tag_filter = f"FILTER {tag_operator.join(tag_conditions)}"
        
        # Add additional filter if provided
        if filter_expr:
            tag_filter += f" AND ({filter_expr})"
        
        # Build the AQL query
        aql = f"""
        FOR doc IN {collections[0]}
        {tag_filter}
        SORT doc._key
        LIMIT {offset}, {limit}
        RETURN {{
            "doc": doc,
            "collection": "{collections[0]}"
        }}
        """
        
        # Create bind variables for tags
        bind_vars = {}
        
        # Add tag bind variables
        for i, tag in enumerate(tags):
            bind_vars[f"tag_{i}"] = tag
        
        # Execute the query
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        # Get total count
        count_aql = f"""
        RETURN LENGTH(
            FOR doc IN {collections[0]}
            {tag_filter}
            RETURN 1
        )
        """
        
        count_cursor = db.aql.execute(count_aql, bind_vars=bind_vars)
        total_count = next(count_cursor)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "results": results,
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "tags": tags,
            "require_all_tags": require_all_tags,
            "time": elapsed
        }
    
    except Exception as e:
        logger.error(f"Tag search error: {e}")
        return {
            "results": [],
            "total": 0,
            "offset": offset,
            "limit": limit,
            "tags": tags,
            "error": str(e)
        }

def validate_tag_search(search_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate tag search results against known good fixture data.
    
    Args:
        search_results: The results returned from tag_search
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
    
    # Structural validation
    if "results" not in search_results:
        validation_failures["missing_results"] = {
            "expected": "Results field present",
            "actual": "Results field missing"
        }
        return False, validation_failures
    
    # Validate attributes
    required_attrs = ["total", "offset", "limit", "tags"]
    for attr in required_attrs:
        if attr not in search_results:
            validation_failures[f"missing_{attr}"] = {
                "expected": f"{attr} field present",
                "actual": f"{attr} field missing"
            }
    
    # Validate result count matches total
    if "total" in search_results and "results" in search_results:
        if search_results["total"] != expected_data.get("total"):
            validation_failures["total_count"] = {
                "expected": expected_data.get("total"),
                "actual": search_results["total"]
            }
        
        if len(search_results["results"]) > search_results["limit"]:
            validation_failures["results_exceed_limit"] = {
                "expected": f"<= {search_results['limit']}",
                "actual": len(search_results["results"])
            }
    
    # Validate tags parameter
    if "tags" in search_results and "tags" in expected_data:
        if set(search_results["tags"]) != set(expected_data["tags"]):
            validation_failures["tags"] = {
                "expected": expected_data["tags"],
                "actual": search_results["tags"]
            }
    
    # Validate result content
    if "results" in search_results and "expected_result_keys" in expected_data:
        found_keys = set()
        for result in search_results["results"]:
            if "doc" in result and "_key" in result["doc"]:
                found_keys.add(result["doc"]["_key"])
        
        expected_keys = set(expected_data["expected_result_keys"])
        if not expected_keys.issubset(found_keys):
            missing_keys = expected_keys - found_keys
            validation_failures["missing_expected_keys"] = {
                "expected": list(expected_keys),
                "actual": list(found_keys),
                "missing": list(missing_keys)
            }
    
    return len(validation_failures) == 0, validation_failures

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/tag_search_expected.json"
    
    try:
        # Set up database connection
        client = connect_arango()
        db = ensure_database(client)

        # Load expected results (will fail initially as file doesn't exist)
        logger.info(f"Loading expected results from: {fixture_path}")
        try:
            with open(fixture_path, 'r') as f:
                expected_data = json.load(f)
            logger.info("Expected results loaded.")
        except FileNotFoundError:
             logger.error(f"Fixture file not found at {fixture_path}. Cannot validate.")
             print(f"❌ FIXTURE ERROR: Fixture file not found at {fixture_path}")
             # Run search anyway to capture output for creating the fixture
             test_tags = ["python", "json"] # Default test tags
             search_results = tag_search(db=db, tags=test_tags, limit=10)
             print("\n--- Search Results (for fixture creation) ---")
             print(json.dumps(search_results, indent=2))
             print("--- End Search Results ---")
             sys.exit(1) # Exit indicating fixture error
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from fixture file {fixture_path}")
            print(f"❌ FIXTURE ERROR: Could not decode JSON from {fixture_path}")
            sys.exit(1)

        # Run a test tag search using tags from fixture
        test_tags = expected_data.get("tags", ["python", "json"]) # Use fixture tags
        search_results = tag_search(
            db=db,
            tags=test_tags,
            limit=10
        )
        
        # Validate the results
        validation_passed, validation_failures = validate_tag_search(search_results, fixture_path)
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All tag search results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Tag search results don't match expected values") 
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
