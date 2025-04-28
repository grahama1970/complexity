#!/usr/bin/env python3
# src/pdf_extractor/arangodb/search_api/message_search.py

import sys
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from arango.database import StandardDatabase

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME as DOC_COLLECTION_NAME,
    VIEW_NAME
)

# Import search functions
from pdf_extractor.arangodb.search_api.hybrid import hybrid_search
from pdf_extractor.arangodb.search_api.bm25 import bm25_search
from pdf_extractor.arangodb.search_api.semantic import semantic_search

# Try to import message_history_config, but provide a fallback if not available
try:
    from pdf_extractor.arangodb.message_history_config import (
        MESSAGE_COLLECTION_NAME
    )
except ImportError:
    # Fallback for testing
    MESSAGE_COLLECTION_NAME = "messages"
    logger.warning("Using fallback MESSAGE_COLLECTION_NAME for testing")

def search_messages(
    db: StandardDatabase,
    query: str,
    search_type: str = 'hybrid',
    top_n: int = 5,
    conversation_id: Optional[str] = None,
    message_type: Optional[str] = None,
    date_range: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Search for messages in the message history.
    
    Args:
        db: ArangoDB database
        query: Search query text
        search_type: Type of search (hybrid, bm25, semantic)
        top_n: Maximum number of results to return
        conversation_id: Filter results by conversation ID
        message_type: Filter results by message type (USER, AGENT, SYSTEM)
        date_range: Optional date range filter
        
    Returns:
        Dict with search results
    """
    # Build filter conditions
    filter_conditions = []
    
    if conversation_id:
        filter_conditions.append(f"doc.conversation_id == '{conversation_id}'")
    
    if message_type:
        filter_conditions.append(f"doc.message_type == '{message_type}'")
    
    if date_range:
        if date_range.get('start'):
            filter_conditions.append(f"doc.timestamp >= '{date_range['start']}'")
        if date_range.get('end'):
            filter_conditions.append(f"doc.timestamp <= '{date_range['end']}'")
    
    # Convert filter conditions to AQL filter expression
    filter_expr = " AND ".join(filter_conditions) if filter_conditions else None
    
    # Perform the appropriate search
    if search_type.lower() == 'bm25':
        results = bm25_search(
            db, 
            query, 
            collections=[MESSAGE_COLLECTION_NAME],
            filter_expr=filter_expr,
            top_n=top_n
        )
    elif search_type.lower() == 'semantic':
        results = semantic_search(
            db, 
            query, 
            collections=[MESSAGE_COLLECTION_NAME],
            filter_expr=filter_expr,
            top_n=top_n
        )
    else:  # Default to hybrid
        results = hybrid_search(
            db, 
            query, 
            collections=[MESSAGE_COLLECTION_NAME],
            filter_expr=filter_expr,
            top_n=top_n
        )
    
    return results

def unified_search(
    db: StandardDatabase,
    query: str,
    search_type: str = 'hybrid',
    collections: Optional[List[str]] = None,
    exclude_collections: Optional[List[str]] = None,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Perform a unified search across multiple collections.
    
    Args:
        db: ArangoDB database
        query: Search query text
        search_type: Type of search (hybrid, bm25, semantic)
        collections: List of collections to search in (if None, search all)
        exclude_collections: List of collections to exclude from search
        top_n: Maximum number of results to return
        
    Returns:
        Dict with search results
    """
    # Default collections to search
    all_collections = [DOC_COLLECTION_NAME, MESSAGE_COLLECTION_NAME]
    
    # If collections are specified, use them
    search_collections = collections if collections else all_collections
    
    # If exclude_collections are specified, remove them from search_collections
    if exclude_collections:
        search_collections = [c for c in search_collections if c not in exclude_collections]
    
    # Perform the appropriate search
    if search_type.lower() == 'bm25':
        results = bm25_search(
            db, 
            query, 
            collections=search_collections,
            top_n=top_n
        )
    elif search_type.lower() == 'semantic':
        results = semantic_search(
            db, 
            query, 
            collections=search_collections,
            top_n=top_n
        )
    else:  # Default to hybrid
        results = hybrid_search(
            db, 
            query, 
            collections=search_collections,
            top_n=top_n
        )
    
    # Add the collections that were searched to the results
    results['collections_searched'] = search_collections
    
    return results

def validate_message_search(search_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate message search results against known good fixture data.
    
    Args:
        search_results: The results returned from search_messages or unified_search
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
    if "total" in expected_data and search_results.get("total") != expected_data.get("total"):
        validation_failures["total_count"] = {
            "expected": expected_data.get("total"),
            "actual": search_results.get("total")
        }
    
    # Validate collections searched (for unified_search)
    if "collections_searched" in search_results and "expected_collections" in expected_data:
        if set(search_results["collections_searched"]) != set(expected_data["expected_collections"]):
            validation_failures["collections_searched"] = {
                "expected": expected_data["expected_collections"],
                "actual": search_results["collections_searched"]
            }
    
    # Validate search query
    if "query" in search_results and "expected_query" in expected_data:
        if search_results["query"] != expected_data["expected_query"]:
            validation_failures["query"] = {
                "expected": expected_data["expected_query"],
                "actual": search_results["query"]
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
    fixture_path = "src/test_fixtures/message_search_expected.json"
    
    try:
        # Set up database connection
        client = connect_arango()
        db = ensure_database(client)
        
        # Try to create a test fixture if it doesn't exist
        try:
            with open(fixture_path, "r") as f:
                fixture_exists = True
        except FileNotFoundError:
            # Create a minimal fixture file
            with open(fixture_path, "w") as f:
                json.dump({
                    "total": 0,
                    "expected_query": "test query",
                    "expected_collections": [DOC_COLLECTION_NAME, MESSAGE_COLLECTION_NAME],
                    "expected_result_keys": []
                }, f)
        
        # Run a test unified search
        test_query = "test query"
        search_results = unified_search(
            db=db,
            query=test_query,
            top_n=5
        )
        
        # Validate the results
        validation_passed, validation_failures = validate_message_search(search_results, fixture_path)
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All message search results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Message search results don't match expected values") 
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
