# src/pdf_extractor/arangodb/search_api/bm25.py
import time
import sys
from typing import Dict, Any, List, Optional

from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError

# Import config variables
from complexity.arangodb.config import (
    COLLECTION_NAME,
    SEARCH_FIELDS,
    ALL_DATA_FIELDS_PREVIEW,
    TEXT_ANALYZER,
    VIEW_NAME,
)
from complexity.arangodb.arango_setup_unknown import connect_arango, ensure_database

def _fetch_bm25_candidates(
    db: StandardDatabase,
    query_text: str,
    top_n: int = 20,
    min_score: float = 0.0,
    tag_filter_clause: str = ""
) -> Dict[str, Any]:
    """
    Fetch BM25 candidates for a query.
    
    Args:
        db: ArangoDB database connection.
        query_text: The search query text.
        top_n: Maximum number of results to return.
        min_score: Minimum BM25 score threshold.
        tag_filter_clause: Optional AQL filter clause for tag filtering.
    
    Returns:
        Dictionary with results and timing information.
    """
    start_time = time.time()
    
    try:
        # Construct AQL query with BM25 scoring
        # fields_str = ", ".join(f'"{field}"' for field in SEARCH_FIELDS) # Not used directly in SEARCH
        preview_fields_str = ", ".join(f'"{field}"' for field in ALL_DATA_FIELDS_PREVIEW)

        # Build the SEARCH clause dynamically from SEARCH_FIELDS
        search_field_conditions = " OR ".join([
            f'ANALYZER(doc.{field} IN search_tokens, "{TEXT_ANALYZER}")'
            for field in SEARCH_FIELDS
        ])

        # Build the AQL query with optional tag filtering
        aql = f"""
        LET search_tokens = TOKENS(@query, "{TEXT_ANALYZER}")
        FOR doc IN {VIEW_NAME}
        SEARCH {search_field_conditions}
        {tag_filter_clause}
        LET score = BM25(doc)
        FILTER score >= @min_score
        SORT score DESC
        LIMIT {top_n}
        RETURN {{
            "doc": KEEP(doc, [{preview_fields_str}]),
            "score": score
        }}
        """
        
        # Execute the query
        bind_vars = {
            "query": query_text,
            # "top_n": top_n, # Removed: Must be literal in LIMIT
            "min_score": min_score
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "results": results,
            "count": len(results),
            "query": query_text,
            "time": elapsed
        }
    
    except (AQLQueryExecuteError, ArangoServerError) as e:
        logger.error(f"ArangoDB query error: {e}")
        return {
            "results": [],
            "count": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error in BM25 search: {e}")
        return {
            "results": [],
            "count": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e)
        }

def bm25_search(
    db: StandardDatabase,
    query_text: str,
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    min_score: float = 0.0,
    top_n: int = 10,
    offset: int = 0,
    tag_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search for documents using BM25 algorithm.
    
    Args:
        db: ArangoDB database
        query_text: Search query text
        collections: Optional list of collections to search
        filter_expr: Optional AQL filter expression
        min_score: Minimum BM25 score threshold
        top_n: Maximum number of results to return
        offset: Offset for pagination
        tag_list: Optional list of tags to filter by
        
    Returns:
        Dict with search results
    """
    try:
        start_time = time.time()
        
        # Use default collection if not specified
        if not collections:
            collections = [COLLECTION_NAME]
        
        # Build filter clause
        filter_clauses = []
        if filter_expr:
            filter_clauses.append(f"({filter_expr})")
        
        # Add tag filter if provided
        if tag_list:
            # Correctly handle tag filtering: require ALL tags if multiple are given?
            # Assuming AND logic for tags based on typical filtering needs.
            # If OR is needed, change " AND " below back to " OR ".
            # Also ensure tags are properly escaped if they contain special chars (though unlikely here).
            tag_conditions = [f'"{tag}" IN doc.tags' for tag in tag_list]
            # Use AND logic for multiple tags
            tag_filter = " AND ".join(tag_conditions)
            filter_clauses.append(f"({tag_filter})")
        
        # Combine filter clauses with AND
        filter_clause = ""
        if filter_clauses:
            filter_clause = "FILTER " + " AND ".join(filter_clauses)
        
        # Build the SEARCH clause dynamically from SEARCH_FIELDS
        search_field_conditions = " OR ".join([
            f'ANALYZER(doc.{field} IN search_tokens, "{TEXT_ANALYZER}")'
            for field in SEARCH_FIELDS
        ])

        # Build the AQL query
        aql = f"""
        LET search_tokens = TOKENS(@query, "{TEXT_ANALYZER}")
        FOR doc IN {VIEW_NAME}
        SEARCH {search_field_conditions}
        {filter_clause}
        LET score = BM25(doc)
        FILTER score >= @min_score
        SORT score DESC
        LIMIT {offset}, {top_n} // Use f-string interpolation as required
        RETURN {{
            "doc": doc,
            "score": score
        }}
        """

        # Execute the query
        bind_vars = {
            "query": query_text,
            "min_score": min_score, # Restore bind variable
            # offset and top_n are interpolated directly into AQL
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        # Get the total count
        # Build the SEARCH clause dynamically for the count query
        search_field_conditions_count = " OR ".join([
            f'ANALYZER(doc.{field} IN search_tokens, "{TEXT_ANALYZER}")'
            for field in SEARCH_FIELDS
        ])

        count_aql = f"""
        RETURN LENGTH(
            LET search_tokens = TOKENS(@query, "{TEXT_ANALYZER}")
            FOR doc IN {VIEW_NAME}
            SEARCH {search_field_conditions_count}
            {filter_clause}
            LET score = BM25(doc)
            FILTER score >= @min_score
            RETURN 1
        )
        """
        # Create separate bind_vars for the count query (only needs query and min_score)
        count_bind_vars = {
            "query": query_text,
            "min_score": min_score
        }
        count_cursor = db.aql.execute(count_aql, bind_vars=count_bind_vars)
        total_count = next(count_cursor)

        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "results": results,
            "total": total_count,
            "offset": offset,
            "query": query_text,
            "time": elapsed
        }
    
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return {
            "results": [],
            "total": 0,
            "offset": offset,
            "query": query_text,
            "error": str(e)
        }

if __name__ == "__main__":
    import json
    import math

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Path to test fixture and expected results definition
    fixture_path = "src/test_fixtures/bm25_search_expected_20250422_181050.json"
    EXPECTED_QUERY = "python error" # Query used to generate the fixture
    EXPECTED_TOP_N = 3 # top_n used for the search that generated the fixture
    EXPECTED_MIN_SCORE = 0.0

    validation_passed = True
    error_messages = []

    try:
        # Load expected results from fixture
        logger.info(f"Loading expected results from: {fixture_path}")
        with open(fixture_path, 'r') as f:
            expected_data = json.load(f)
        logger.info(f"Expected results loaded successfully.")
        expected_results = expected_data.get("results", [])
        expected_total = expected_data.get("total", 0)
        # Extract expected keys and scores for easier comparison
        expected_keys_scores = {res["doc"]["_key"]: res["score"] for res in expected_results} # Use 'score' key from fixture


        # Set up database connection
        logger.info("Connecting to ArangoDB...")
        client = connect_arango()
        db = ensure_database(client)
        logger.info("Database connection established.")

        # Run the test search query
        logger.info(f"Running BM25 search with query: '{EXPECTED_QUERY}'")
        search_results = bm25_search(
            db=db,
            query_text=EXPECTED_QUERY,
            top_n=EXPECTED_TOP_N,
            min_score=EXPECTED_MIN_SCORE
        )
        logger.info(f"Search completed. Found {search_results.get('total', 0)} total results, returned {len(search_results.get('results', []))} results.")

        # --- Validation ---
        actual_results = search_results.get("results", [])
        actual_total = search_results.get("total", 0)

        # 1. Validate total count
        if actual_total != expected_total:
            validation_passed = False
            error_messages.append(f"Total count mismatch: Expected {expected_total}, Got {actual_total}")

        # 2. Validate number of returned results (can be <= top_n and <= total)
        # The fixture has 2 results, even though top_n might have been 3.
        if len(actual_results) != len(expected_results):
             validation_passed = False
             error_messages.append(f"Returned results count mismatch: Expected {len(expected_results)}, Got {len(actual_results)}")
        else:
            # 3. Validate individual results (key and score)
            actual_keys_scores = {res["doc"]["_key"]: res["score"] for res in actual_results}

            if set(actual_keys_scores.keys()) != set(expected_keys_scores.keys()):
                validation_passed = False
                error_messages.append(f"Result keys mismatch: Expected {sorted(expected_keys_scores.keys())}, Got {sorted(actual_keys_scores.keys())}")
            else:
                # Check scores if keys match
                for key, expected_score in expected_keys_scores.items():
                    actual_score = actual_keys_scores[key]
                    # Use math.isclose for float comparison
                    if not math.isclose(actual_score, expected_score, rel_tol=1e-6):
                        validation_passed = False
                        error_messages.append(f"Score mismatch for key '{key}': Expected {expected_score:.6f}, Got {actual_score:.6f}")

        # --- Reporting ---
        # --- Reporting ---
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All results match expected values from fixture.")
            # Optionally print results for confirmation
            print(f"  Query: '{EXPECTED_QUERY}'")
            print(f"  Total results found: {actual_total}")
            print(f"  Results returned (top_n={EXPECTED_TOP_N}): {len(actual_results)}")
            for i, result in enumerate(actual_results):
                 doc = result["doc"]
                 score = result["score"]
                 print(f"    Result {i+1}: Key={doc['_key']}, Score={score:.4f}")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Results do not match expected values from fixture.")
            for msg in error_messages:
                print(f"  - {msg}")
            # Print actual results for debugging
            print("\nActual Results:")
            print(json.dumps(search_results, indent=2))
            sys.exit(1)

    except FileNotFoundError:
        print(f"❌ ERROR: Fixture file not found at {fixture_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ ERROR: Could not decode JSON from fixture file {fixture_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR during execution or validation: {str(e)}")
        logger.exception("Detailed traceback:") # Log full traceback
        sys.exit(1)
