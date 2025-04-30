# src/pdf_extractor/arangodb/search_api/glossary.py
import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError

from complexity.arangodb.config import (
    COLLECTION_NAME,
    ALL_DATA_FIELDS_PREVIEW,
    TEXT_ANALYZER
)
from complexity.arangodb.arango_setup_unknown import connect_arango, ensure_database

def glossary_search(
    db: StandardDatabase,
    terms: List[str],
    collections: Optional[List[str]] = None,
    match_exact: bool = False,
    fuzzy_threshold: float = 0.7,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search for glossary terms in documents.
    
    Args:
        db: ArangoDB database
        terms: List of glossary terms to search for
        collections: Optional list of collections to search
        match_exact: Whether to require exact term matches
        fuzzy_threshold: Threshold for fuzzy matching (0-1)
        limit: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    start_time = time.time()
    logger.info(f"Searching for glossary terms: {terms}")
    
    try:
        # Use default collection if not specified
        if not collections:
            collections = [COLLECTION_NAME]
        
        # Define searchable fields
        searchable_fields = ["problem", "solution", "context", "tags"] # Assuming these are the relevant fields

        # Build search clause based on terms and fields
        term_match_clauses = []
        bind_vars_terms = {}

        for i, term in enumerate(terms):
            field_clauses = []
            term_bind_var = f"term_{i}"
            bind_vars_terms[term_bind_var] = term

            for field in searchable_fields:
                if match_exact:
                    # Exact phrase matching within fields using LIKE (case-insensitive with analyzer)
                    # Note: LIKE might not be the most efficient for exact phrase with analyzer.
                    # Consider PHRASE() if exact phrase semantics are needed.
                    # Using LIKE '%term%' for broader substring matching for now.
                    field_clauses.append(f"ANALYZER(doc.{field} LIKE CONCAT('%', @{term_bind_var}, '%'), '{TEXT_ANALYZER}')")
                else:
                    # Fuzzy term matching with NGRAM_MATCH per field
                    # Ensure the field exists or handle potential errors if needed
                    field_clauses.append(f"NGRAM_MATCH(doc.{field}, @{term_bind_var}, {fuzzy_threshold}, '{TEXT_ANALYZER}')")

            # Combine field clauses with OR for a single term
            term_match_clauses.append(f"({' OR '.join(field_clauses)})")

        # Combine term clauses with OR (document matches if ANY term matches in ANY field)
        search_filter = f"FILTER {' OR '.join(term_match_clauses)}"

        # Build the AQL query
        aql = f"""
        FOR doc IN {collections[0]}
        {search_filter}
        SORT doc._key
        LIMIT @limit
        RETURN {{
            "doc": doc,
            "collection": "{collections[0]}"
        }}
        """
        
        # Create bind variables for terms (only needed for fuzzy matching)
        bind_vars = {
            "limit": limit
        }
        
        # Add term bind variables collected earlier
        bind_vars.update(bind_vars_terms)

        # Execute the query
        logger.debug(f"Executing AQL: {aql}")
        logger.debug(f"With Bind Vars: {bind_vars}")
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "results": results,
            "count": len(results),
            "terms": terms,
            "match_exact": match_exact,
            "fuzzy_threshold": fuzzy_threshold,
            "time": elapsed
        }
    
    except Exception as e:
        logger.error(f"Glossary search error: {e}")
        return {
            "results": [],
            "count": 0,
            "terms": terms,
            "error": str(e)
        }

def validate_glossary_search(search_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate glossary search results against known good fixture data.
    
    Args:
        search_results: The results returned from glossary_search
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
    required_attrs = ["count", "terms", "match_exact", "fuzzy_threshold"]
    for attr in required_attrs:
        if attr not in search_results:
            validation_failures[f"missing_{attr}"] = {
                "expected": f"{attr} field present",
                "actual": f"{attr} field missing"
            }
    
    # Validate result count
    if "count" in search_results and "expected_count" in expected_data:
        if search_results["count"] < expected_data["expected_count"]:
            validation_failures["count"] = {
                "expected": f">= {expected_data['expected_count']}",
                "actual": search_results["count"]
            }
    
    # Validate search terms
    if "terms" in search_results and "expected_terms" in expected_data:
        if set(search_results["terms"]) != set(expected_data["expected_terms"]):
            validation_failures["terms"] = {
                "expected": expected_data["expected_terms"],
                "actual": search_results["terms"]
            }
    
    # Validate result content
    if "results" in search_results and "expected_result_keys" in expected_data:
        found_keys = set()
        for result in search_results["results"]:
            if "doc" in result and "_key" in result["doc"]:
                found_keys.add(result["doc"]["_key"])
        
        expected_keys = set(expected_data["expected_result_keys"])
        if not expected_keys.issubset(found_keys):
            validation_failures["missing_expected_keys"] = {
                "expected": list(expected_keys),
                "actual": list(found_keys)
            }
    
    return len(validation_failures) == 0, validation_failures

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/glossary_expected.json"
    
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
             test_terms = ["error", "python"] # Default test terms
             search_results = glossary_search(db=db, terms=test_terms, limit=10)
             print("\n--- Search Results (for fixture creation) ---")
             print(json.dumps(search_results, indent=2))
             print("--- End Search Results ---")
             sys.exit(1) # Exit indicating fixture error
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from fixture file {fixture_path}")
            print(f"❌ FIXTURE ERROR: Could not decode JSON from {fixture_path}")
            sys.exit(1)


        # Run a test glossary search using terms from fixture if available
        test_terms = expected_data.get("expected_terms", ["error", "python"]) # Use fixture terms
        search_results = glossary_search(
            db=db,
            terms=test_terms,
            limit=10
        )
        
        # Validate the results
        validation_passed, validation_failures = validate_glossary_search(search_results, fixture_path)
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All glossary search results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Glossary search results don't match expected values") 
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
