"""
Keyword Search Module for PDF Extractor ArangoDB Integration.

This module provides functionality for performing keyword searches with fuzzy matching
using ArangoDB and RapidFuzz.

Third-Party Package Documentation:
- python-arango: https://python-driver.arangodb.com/
- rapidfuzz: https://rapidfuzz.github.io/RapidFuzz/

Sample Input:
Search term and database connection details

Expected Output:
List of documents matching the keyword search with fuzzy matching
"""
import sys
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import re
from loguru import logger

from arango.database import StandardDatabase
from arango.cursor import Cursor  # Import Cursor for type checking
from rapidfuzz import fuzz, process

# Import config variables and connection setup
from complexity.arangodb.arango_setup_unknown import connect_arango, ensure_database
from complexity.arangodb.config import (
    VIEW_NAME,
    COLLECTION_NAME,
    TEXT_ANALYZER
)

def search_keyword(
    db: StandardDatabase,
    search_term: str,
    similarity_threshold: float = 97.0,
    top_n: int = 10,
    view_name: str = VIEW_NAME, 
    tags: Optional[List[str]] = None,
    collection_name: str = COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Perform a keyword search with fuzzy matching.
    
    Args:
        db: ArangoDB database connection
        search_term: The keyword to search for
        similarity_threshold: Minimum similarity score (0-100) for fuzzy matching
        top_n: Maximum number of results to return
        view_name: Name of the ArangoDB search view
        collection_name: Name of the collection
        
    Returns:
        Dictionary containing results and metadata
        
    Raises:
        ValueError: If search_term is empty
        Exception: For any other errors
    """
    if not search_term or search_term.strip() == "":
        raise ValueError("Search term cannot be empty")
    
    # Clean search term
    search_term = search_term.strip()
    
    # AQL query with bind parameters
    aql_query = f"""
    FOR doc IN {view_name}
      SEARCH ANALYZER(doc.problem LIKE @search_pattern OR 
                    doc.solution LIKE @search_pattern OR 
                    doc.context LIKE @search_pattern, 
                    "{TEXT_ANALYZER}")
      SORT BM25(doc) DESC
      LIMIT @top_n
      RETURN {{ 
        doc: KEEP(doc, "_key", "_id", "problem", "solution", "context", "tags")
      }}
    """
    
    # Bind parameters: Use a simple pattern without word boundaries
    # The word matching will be done with rapidfuzz instead
    bind_vars = {
        "search_pattern": f"%{search_term}%",
        "top_n": top_n
    }
    
    try:
        # Execute AQL query
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        # Iterate over the cursor correctly, adding checks for safety
        # Iterate over the cursor correctly, adding explicit type check
        initial_results = []
        if isinstance(cursor, Cursor):
            try:
                initial_results = [doc for doc in cursor]
            except Exception as e:  # Catch potential errors during iteration
                logger.error(f"Error iterating over cursor results: {e}", exc_info=True)
                raise  # Re-raise to signal failure
        elif cursor is None:
             logger.warning("db.aql.execute returned None, expected a cursor.")
        else:
             # Log if it's an unexpected type (like AsyncJob/BatchJob in sync context)
             logger.error(f"db.aql.execute returned unexpected type: {type(cursor)}. Expected Cursor.")
             # Decide how to handle - raise error?
             raise TypeError(f"Unexpected cursor type: {type(cursor)}")

        # Filter results using rapidfuzz for whole word matching
        filtered_results = []
        for item in initial_results:
            doc = item.get("doc", {})
            
            # Combine searchable text
            text = " ".join([
                str(doc.get("problem", "")),
                str(doc.get("solution", "")),
                str(doc.get("context", ""))
            ]).lower()
            
            # Extract whole words from the text
            words = re.findall(r'\b\w+\b', text)
            
            # Use rapidfuzz to find words with similarity to search_term
            matches = process.extract(
                search_term.lower(),
                words,
                scorer=fuzz.ratio,
                score_cutoff=similarity_threshold
            )
            
            if matches:
                # Add the match and its similarity score
                best_match = matches[0]  # tuple of (match, score)
                item["keyword_score"] = best_match[1] / 100.0  # convert to 0-1 scale
                filtered_results.append(item)
        
        # Sort results by keyword_score (highest first)
        filtered_results.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
        
        # Limit to top_n
        filtered_results = filtered_results[:top_n]
        
        # Create result object
        result = {
            "results": filtered_results,
            "total": len(filtered_results),
            "search_term": search_term,
            "similarity_threshold": similarity_threshold
        }
        
        logger.info(f"Keyword search for '{search_term}' found {len(filtered_results)} results")
        return result
    
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return {
            "results": [],
            "total": 0,
            "search_term": search_term,
            "error": str(e)
        }

def validate_keyword_search(search_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate keyword search results against known good fixture data.
    
    Args:
        search_results: The results returned from search_keyword
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
    
    # Validate search term
    if "search_term" in expected_data and "search_term" in search_results:
        if search_results["search_term"] != expected_data["search_term"]:
            validation_failures["search_term"] = {
                "expected": expected_data["search_term"],
                "actual": search_results["search_term"]
            }
    
    # Validate similarity threshold
    if "similarity_threshold" in expected_data and "similarity_threshold" in search_results:
        if search_results["similarity_threshold"] != expected_data["similarity_threshold"]:
            validation_failures["similarity_threshold"] = {
                "expected": expected_data["similarity_threshold"],
                "actual": search_results["similarity_threshold"]
            }
    
    # Validate total count
    if "total" in expected_data and "total" in search_results:
        if search_results["total"] < expected_data["total"]:
            validation_failures["total_count"] = {
                "expected": f">= {expected_data['total']}",
                "actual": search_results["total"]
            }
    
    # Validate result content
    if "results" in search_results and "expected_result_keys" in expected_data:
        found_keys = set()
        for item in search_results["results"]:
            if "doc" in item and "_key" in item["doc"]:
                found_keys.add(item["doc"]["_key"])
        
        expected_keys = set(expected_data["expected_result_keys"])
        if not expected_keys.issubset(found_keys):
            missing_keys = expected_keys - found_keys
            validation_failures["missing_expected_keys"] = {
                "expected": list(expected_keys),
                "actual": list(found_keys),
                "missing": list(missing_keys)
            }
    
    # Validate scores
    if "results" in search_results and len(search_results["results"]) > 0:
        # Check if all results have keyword_score
        for i, item in enumerate(search_results["results"]):
            if "keyword_score" not in item:
                validation_failures[f"missing_score_result_{i}"] = {
                    "expected": "keyword_score present",
                    "actual": "keyword_score missing"
                }
    
    return len(validation_failures) == 0, validation_failures

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    )
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/keyword_search_expected.json"
    
    try:
        # Connect to ArangoDB
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
             search_term = "python" # Default test term
             search_results = search_keyword(db=db, search_term=search_term, similarity_threshold=97.0, top_n=10)
             print("\n--- Search Results (for fixture creation) ---")
             print(json.dumps(search_results, indent=2))
             print("--- End Search Results ---")
             sys.exit(1) # Exit indicating fixture error
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from fixture file {fixture_path}")
            print(f"❌ FIXTURE ERROR: Could not decode JSON from {fixture_path}")
            sys.exit(1)

        # Run a test keyword search using term from fixture
        search_term = expected_data.get("search_term", "python") # Use fixture term
        search_results = search_keyword(
            db=db,
            search_term=search_term,
            similarity_threshold=97.0,
            top_n=10
        )
        
        # Validate the results
        validation_passed, validation_failures = validate_keyword_search(search_results, fixture_path)
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All keyword search results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Keyword search results don't match expected values") 
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
