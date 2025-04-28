# src/pdf_extractor/arangodb/search_api/semantic.py
import time
import json
import sys
from typing import Dict, Any, List, Optional, Tuple, Union

from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError

# Import config variables and embedding utils
try:
    from pdf_extractor.arangodb.config import (
        COLLECTION_NAME,
        VIEW_NAME,  # Keep VIEW_NAME import for potential future use or other functions
        ALL_DATA_FIELDS_PREVIEW,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSIONS
    )
    from pdf_extractor.arangodb.arango_setup import EMBEDDING_FIELD, connect_arango, ensure_database
    from pdf_extractor.arangodb.embedding_utils import get_embedding
except ImportError as e:
    logger.critical(f"CRITICAL: Failed module-level import in semantic.py: {e}. Functionality will be broken.")
    # Define fallbacks to allow module to load (but not function)
    COLLECTION_NAME = "documents"
    VIEW_NAME = "document_view"
    ALL_DATA_FIELDS_PREVIEW = ["_key"]
    EMBEDDING_MODEL = "text-embedding-ada-002"
    EMBEDDING_DIMENSIONS = 1536
    EMBEDDING_FIELD = "embedding"

def _fetch_semantic_candidates(
    db: StandardDatabase,
    query_text: str,
    top_n: int = 20,
    min_score: float = 0.0,
    tag_filter_clause: str = ""
) -> Dict[str, Any]:
    """
    Fetch semantic candidates for a query using vector similarity.
    Uses APPROX_NEAR_COSINE and targets the COLLECTION directly, assuming a vector index exists.

    Args:
        db: ArangoDB database connection.
        query_text: The search query text.
        top_n: Maximum number of results to return.
        min_score: Minimum similarity score threshold.
        tag_filter_clause: Optional AQL filter clause for tag filtering.

    Returns:
        Dictionary with results and timing information.
    """
    start_time = time.time()

    try:
        # Get query embedding
        query_embedding = get_embedding(query_text, EMBEDDING_MODEL)
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            return {
                "results": [],
                "count": 0,
                "query": query_text,
                "time": time.time() - start_time,
                "error": "Failed to generate embedding"
            }

        # Format preview fields string
        preview_fields_str = ", ".join(f'"{field}"' for field in ALL_DATA_FIELDS_PREVIEW)

        # Build the AQL query with vector search - Target COLLECTION_NAME and force index hint
        aql = f"""
        FOR doc IN {COLLECTION_NAME}
          OPTIONS {{ indexHint: "vector_index", forceIndexHint: true }}
        {tag_filter_clause}
        LET score = APPROX_NEAR_COSINE(doc.{EMBEDDING_FIELD}, @query_embedding)
        FILTER score >= @min_score
        SORT score DESC
        LIMIT @top_n
        RETURN {{
            "doc": KEEP(doc, [{preview_fields_str}]),
            "score": score
        }}
        """

        # Execute the query
        bind_vars = {
            "query_embedding": query_embedding,
            "top_n": top_n,
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
            "embedding_model": EMBEDDING_MODEL,
            "time": elapsed
        }

    except (AQLQueryExecuteError, ArangoServerError) as e:
        logger.error(f"ArangoDB query error in semantic search: {e}")
        return {
            "results": [],
            "count": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error in semantic search: {e}")
        return {
            "results": [],
            "count": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e)
        }

# --- Start of Restored semantic_search ---
def semantic_search(
    db: StandardDatabase,
    query: Union[str, List[float]],
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    min_score: float = 0.7,
    top_n: int = 10,
    tag_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search for documents using semantic vector similarity.

    Args:
        db: ArangoDB database
        query: Search query text or embedding vector
        collections: Optional list of collections to search
        filter_expr: Optional AQL filter expression
        min_score: Minimum similarity score threshold (0-1)
        top_n: Maximum number of results to return
        tag_list: Optional list of tags to filter by

    Returns:
        Dict with search results
    """
    try:
        start_time = time.time()

        # Use default collection if not specified
        if not collections:
            collections = [COLLECTION_NAME]

        # Get query embedding if string is provided
        query_embedding = query
        query_text_for_return = "vector query"
        if isinstance(query, str):
            query_text_for_return = query
            query_embedding = get_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return {
                    "results": [],
                    "total": 0,
                    "query": query,
                    "error": "Failed to generate embedding"
                }

        # Build the SEARCH filter expression (Only for filter_expr now, tags handled in Python)
        search_filter_parts = []
        bind_vars = { # Initialize bind_vars here
            "query_embedding": query_embedding,
            "min_score": min_score,
            # top_n handled in Python
            # tag_list handled in Python
        }
        if filter_expr:
            # Assuming filter_expr is a valid ArangoSearch expression part
            search_filter_parts.append(f"({filter_expr})")
        # Tag list is NOT added to bind_vars or search_filter_parts here

        search_filter_expression = " AND ".join(search_filter_parts) if search_filter_parts else ""
        search_clause = f"SEARCH {search_filter_expression}" if search_filter_expression else ""

        # Build AQL using COSINE_SIMILARITY on the View
        aql = f"""
        FOR doc IN {VIEW_NAME}
        {search_clause} // SEARCH clause only contains filter_expr now (if any)
        LET similarity = COSINE_SIMILARITY(doc.{EMBEDDING_FIELD}, @query_embedding) // Calculate similarity once
        FILTER similarity >= @min_score // Filter by score first
        SORT similarity DESC // Sort the filtered results
        // LIMIT is handled in Python
        RETURN {{
            "doc": doc,
            "similarity_score": similarity // Return pre-calculated similarity
        }}
        """
        # bind_vars only contains query_embedding and min_score

        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)

        # Apply tag filtering in Python if tag_list is provided
        if tag_list:
            filtered_results = []
            required_tags = set(tag_list)
            for result in results:
                doc_tags = set(result.get("doc", {}).get("tags", []))
                # Check if all required tags are present in the document's tags (AND logic)
                if required_tags.issubset(doc_tags):
                    filtered_results.append(result)
            results = filtered_results # Replace results with filtered list

        # Apply limit in Python after filtering
        results = results[:top_n]

        # --- Count Query Temporarily Disabled for Debugging ---
        # # Get the total count using the view and COSINE_SIMILARITY
        # count_search_clause = f"SEARCH {search_filter_expression}" if search_filter_expression else "" # Only add SEARCH if there's a filter
        # count_aql = f"""
        # RETURN COUNT( // Use COUNT for efficiency
        #     FOR doc IN {VIEW_NAME}
        #     {count_search_clause} // Use the generated SEARCH clause (or empty string)
        #     LET similarity = COSINE_SIMILARITY(doc.{EMBEDDING_FIELD}, @query_embedding)
        #     FILTER similarity >= @min_score
        #     RETURN 1
        # )
        # """
        # # Need to pass query_embedding and min_score (and potentially tag_list) to count query bind_vars
        # count_bind_vars = {
        #      "query_embedding": query_embedding,
        #      "min_score": min_score
        # }
        # if "tag_list" in bind_vars: # Pass tag_list if it was used
        #      count_bind_vars["tag_list"] = bind_vars["tag_list"]
        #
        # count_cursor = db.aql.execute(count_aql, bind_vars=count_bind_vars)
        # total_count = next(count_cursor) # COUNT returns the count directly
        # --- Count Query Temporarily Disabled for Debugging ---
        # Re-enable count query, but modify it to match the main query's filtering logic (no SEARCH)
        # Build filter clause string for count query (excluding score filter)
        count_filter_clauses = []
        if filter_expr:
             count_filter_clauses.append(f"({filter_expr})")
        # Add tag filter using standard AQL FILTER for count query
        if tag_list:
             count_filter_clauses.append("doc.tags ALL IN @tag_list")

        count_filter_clause_str = f"FILTER {' AND '.join(count_filter_clauses)}" if count_filter_clauses else ""

        count_aql = f"""
        RETURN COUNT(
            FOR doc IN {VIEW_NAME}
            {count_filter_clause_str} // Apply non-score filters
            LET similarity = COSINE_SIMILARITY(doc.{EMBEDDING_FIELD}, @query_embedding)
            FILTER similarity >= @min_score // Apply score filter
            RETURN 1
        )
        """
        # Bind vars for count query
        count_bind_vars = {
             "query_embedding": query_embedding,
             "min_score": min_score
        }
        if tag_list: # Pass tag_list if it was used
             count_bind_vars["tag_list"] = tag_list

        try:
            count_cursor = db.aql.execute(count_aql, bind_vars=count_bind_vars)
            total_count = next(count_cursor) # COUNT returns the count directly
        except Exception as count_e:
             logger.warning(f"Failed to execute count query: {count_e}. Total count will be inaccurate.")
             total_count = len(results) # Fallback to length of filtered+limited results

        # --- End Count Query ---

        end_time = time.time()
        elapsed = end_time - start_time

        return {
            "results": results,
            "total": total_count if total_count != -1 else len(results), # Use len(results) if count query disabled
            "query": query_text_for_return,
            "time": elapsed
        }

    except Exception as e:
        logger.error(f"Caught exception of type: {type(e)}") # Log type first
        # Attempt to extract specific ArangoDB error details safely
        error_code = getattr(e, 'error_code', None)
        error_message = getattr(e, 'error_message', None)
        http_exception = getattr(e, 'http_exception', None)
        http_code = getattr(http_exception, 'code', None) if http_exception else None

        # Construct a detailed log message
        log_details = []
        if error_code: log_details.append(f"ErrorCode: {error_code}")
        if http_code: log_details.append(f"HTTPCode: {http_code}")
        if error_message: log_details.append(f"Msg: {error_message}")

        log_message = f"Semantic search failed. Details: {', '.join(log_details)}"

        # Fallback to repr only if no specific details were found
        if not log_details:
            try:
                error_repr = repr(e)
                log_message += f", Raw Exception (repr): {error_repr}"
            except TypeError: # Catch the problematic TypeError during repr
                 log_message += ", Raw Exception: (repr failed with TypeError)"

        logger.error(log_message) # Log the constructed message

        # Construct error string for return value
        error_str_details = []
        if error_code: error_str_details.append(f"Code: {error_code}")
        if http_code: error_str_details.append(f"HTTP: {http_code}")
        if error_message: error_str_details.append(f"Msg: {error_message}")
        if not error_str_details: # Fallback for return value
             error_str_details.append(f"Raw: {repr(e) if isinstance(e, BaseException) else 'Unknown Error Type'}")


        return {
            "results": [],
            "total": 0,
            "query": query if isinstance(query, str) else "vector query",
            "error": ", ".join(error_str_details) # Return extracted details or fallback
        }
# --- End of Restored semantic_search ---


def validate_semantic_search(search_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate semantic search results against known good fixture data.

    Args:
        search_results: The results returned from semantic_search
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

    # Validate result count
    if len(search_results["results"]) != len(expected_data["results"]):
        validation_failures["result_count"] = {
            "expected": len(expected_data["results"]),
            "actual": len(search_results["results"])
        }

    # Validate total count
    if search_results.get("total") != expected_data.get("total"):
        validation_failures["total_count"] = {
            "expected": expected_data.get("total"),
            "actual": search_results.get("total")
        }

    # Content validation - compare actual results with expected
    for i, (expected_result, actual_result) in enumerate(
        zip(expected_data["results"], search_results["results"])
    ):
        # Check structure of each result
        if "doc" not in actual_result:
            validation_failures[f"result_{i}_missing_doc"] = {
                "expected": "doc field present",
                "actual": "doc field missing"
            }
            continue

        # Check specific, meaningful document fields, ignoring volatile ones
        fields_to_validate = ["problem", "solution", "context", "tags", "_key"] # Keep _key for now, remove if unstable
        for key in fields_to_validate:
            if key not in expected_result.get("doc", {}):
                 # If key is not in expected, don't validate it (allows fixture flexibility)
                 continue
            if key not in actual_result.get("doc", {}):
                validation_failures[f"result_{i}_doc_missing_{key}"] = {
                    "expected": f"'{key}' field present",
                    "actual": f"'{key}' field missing"
                }
            elif actual_result["doc"][key] != expected_result["doc"][key]:
                 # Special handling for tags list comparison (order might not matter)
                 if key == "tags" and isinstance(actual_result["doc"][key], list) and isinstance(expected_result["doc"][key], list):
                     if sorted(actual_result["doc"][key]) != sorted(expected_result["doc"][key]):
                         validation_failures[f"result_{i}_doc_{key}"] = {
                             "expected": sorted(expected_result["doc"][key]),
                             "actual": sorted(actual_result["doc"][key])
                         }
                 else:
                    validation_failures[f"result_{i}_doc_{key}"] = {
                        "expected": expected_result["doc"][key],
                        "actual": actual_result["doc"][key]
                    }

        # Check similarity score field (allow for score or similarity_score key)
        expected_score = expected_result.get("similarity_score", expected_result.get("score"))
        actual_score = actual_result.get("similarity_score", actual_result.get("score"))

        # For floating point scores, use approximate comparison
        if expected_score is not None and actual_score is not None:
            # Allow small differences in scores due to floating point precision
            if abs(expected_score - actual_score) > 0.01:
                validation_failures[f"result_{i}_score"] = {
                    "expected": expected_score,
                    "actual": actual_score
                }
        elif expected_score != actual_score:
            validation_failures[f"result_{i}_score"] = {
                "expected": expected_score,
                "actual": actual_score
            }

    return len(validation_failures) == 0, validation_failures

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Path to test fixture
    fixture_path = "src/test_fixtures/semantic_search_expected_20250422_181101.json"

    try:
        # Set up database connection
        client = connect_arango()
        db = ensure_database(client)

        # Run a test search query
        test_query = "python error handling"  # Known query that should match fixture results
        search_results = semantic_search(
            db=db,
            query=test_query,
            top_n=10,
            min_score=0.6 # This min_score IS used now
        )

        # Validate the results (only if not in debug mode)
        validation_passed, validation_failures = validate_semantic_search(search_results, fixture_path)

        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All semantic search results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Semantic search results don't match expected values")
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error in __main__: {e}") # Log full traceback
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        sys.exit(1)
