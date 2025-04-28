#!/usr/bin/env python3
# src/pdf_extractor/arangodb/search_api/setup_views.py

import sys
import json
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from arango.database import StandardDatabase

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME as DOC_COLLECTION_NAME,
    VIEW_NAME,
    TEXT_ANALYZER,
    TAG_ANALYZER,
    EMBEDDING_DIMENSIONS
)

# Try to import message_history_config, but provide a fallback if not available
try:
    from pdf_extractor.arangodb.message_history_config import (
        MESSAGE_COLLECTION_NAME
    )
except ImportError:
    # Fallback for testing
    MESSAGE_COLLECTION_NAME = "messages"
    logger.warning("Using fallback MESSAGE_COLLECTION_NAME for testing")

def setup_search_views(db: StandardDatabase) -> bool:
    """
    Set up search views for both documents and messages.
    
    Args:
        db: ArangoDB database connection
    
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Define view properties
        view_properties = {
            "links": {
                # Document collection
                DOC_COLLECTION_NAME: {
                    "fields": {
                        "content": {
                            "analyzers": [TEXT_ANALYZER]
                        },
                        "tags": {
                            "analyzers": [TAG_ANALYZER]
                        }
                    },
                    "includeAllFields": False,
                    "storeValues": "none",
                    "trackListPositions": False
                },
                # Message history collection
                MESSAGE_COLLECTION_NAME: {
                    "fields": {
                        "content": {
                            "analyzers": [TEXT_ANALYZER]
                        },
                        "metadata.tags": {
                            "analyzers": [TAG_ANALYZER]
                        }
                    },
                    "includeAllFields": False,
                    "storeValues": "none",
                    "trackListPositions": False
                }
            }
        }
        
        # Create or update the view
        if db.has_view(VIEW_NAME):
            db.update_view(VIEW_NAME, view_properties)
            logger.info(f"Updated search view: {VIEW_NAME}")
        else:
            db.create_view(VIEW_NAME, "arangosearch", view_properties)
            logger.info(f"Created search view: {VIEW_NAME}")
        
        # Set up vector indices for both collections
        ensure_vector_index(db, DOC_COLLECTION_NAME)
        ensure_vector_index(db, MESSAGE_COLLECTION_NAME)
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up search views: {e}")
        return False


def ensure_vector_index(
    db: StandardDatabase,
    collection_name: str,
    field_name: str = "embedding",
    dimensions: int = EMBEDDING_DIMENSIONS
) -> bool:
    """
    Ensure a vector index exists for a collection.
    
    Args:
        db: ArangoDB database connection
        collection_name: Name of the collection
        field_name: Name of the field to index
        dimensions: Dimensions of the vector
    
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Get the collection
        collection = db.collection(collection_name)
        
        # Check if vector index already exists
        vector_index = None
        for index in collection.indexes():
            if index.get("type") == "vector" and field_name in index.get("fields", []):
                vector_index = index
                break
        
        if vector_index:
            logger.info(f"Vector index already exists on {collection_name}.{field_name}")
            return True
        
        # Create the vector index
        index_props = {
            "type": "vector",
            "fields": [field_name],
            "name": f"idx_{collection_name}_{field_name}",
            "params": {
                "dimension": dimensions,
                "metric": "cosine",
                "nLists": 2
            }
        }
        
        collection.add_index(index_props)
        logger.info(f"Created vector index on {collection_name}.{field_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to ensure vector index: {e}")
        return False

def validate_view_setup(db: StandardDatabase, fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate that search views and vector indices have been set up correctly.
    
    Args:
        db: ArangoDB database connection
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
    
    # Check if view exists
    if not db.has_view(VIEW_NAME):
        validation_failures["view_existence"] = {
            "expected": f"View {VIEW_NAME} exists",
            "actual": f"View {VIEW_NAME} does not exist"
        }
        return False, validation_failures
    
    # Check view properties
    try:
        view_properties = db.view(VIEW_NAME)
        
        # Check if the required collections are linked
        links = view_properties.get("links", {})
        
        if DOC_COLLECTION_NAME not in links:
            validation_failures[f"link_{DOC_COLLECTION_NAME}"] = {
                "expected": f"Collection {DOC_COLLECTION_NAME} linked in view",
                "actual": f"Collection {DOC_COLLECTION_NAME} not linked in view"
            }
        
        if MESSAGE_COLLECTION_NAME not in links:
            validation_failures[f"link_{MESSAGE_COLLECTION_NAME}"] = {
                "expected": f"Collection {MESSAGE_COLLECTION_NAME} linked in view",
                "actual": f"Collection {MESSAGE_COLLECTION_NAME} not linked in view"
            }
    except Exception as e:
        validation_failures["view_properties"] = {
            "expected": "View properties accessible",
            "actual": f"Error: {str(e)}"
        }
    
    # Check vector indices
    for collection_name in [DOC_COLLECTION_NAME, MESSAGE_COLLECTION_NAME]:
        try:
            # Check if collection exists
            if not db.has_collection(collection_name):
                validation_failures[f"collection_{collection_name}"] = {
                    "expected": f"Collection {collection_name} exists",
                    "actual": f"Collection {collection_name} does not exist"
                }
                continue
            
            # Check vector index
            collection = db.collection(collection_name)
            vector_index_exists = False
            
            for index in collection.indexes():
                if index.get("type") == "vector" and "embedding" in index.get("fields", []):
                    vector_index_exists = True
                    break
            
            if not vector_index_exists and expected_data.get("vector_indices_required", True):
                validation_failures[f"vector_index_{collection_name}"] = {
                    "expected": f"Vector index on {collection_name}.embedding exists",
                    "actual": f"Vector index on {collection_name}.embedding does not exist"
                }
        except Exception as e:
            validation_failures[f"collection_check_{collection_name}"] = {
                "expected": f"Collection {collection_name} check successful",
                "actual": f"Error: {str(e)}"
            }
    
    return len(validation_failures) == 0, validation_failures

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/setup_expected.json"
    
    try:
        # Connect to ArangoDB
        client = connect_arango()
        db = ensure_database(client)
        
        # Create a test fixture if it doesn't exist
        try:
            with open(fixture_path, "r") as f:
                fixture_exists = True
        except FileNotFoundError:
            # Create a minimal fixture file
            with open(fixture_path, "w") as f:
                json.dump({
                    "collections": [DOC_COLLECTION_NAME, MESSAGE_COLLECTION_NAME],
                    "view_name": VIEW_NAME,
                    "vector_indices_required": True,
                    "graph_name": "knowledge_graph"
                }, f)
        
        # Set up search views
        setup_result = setup_search_views(db)
        
        # Validate the setup
        validation_passed, validation_failures = validate_view_setup(db, fixture_path)
        
        # Report validation status
        if setup_result and validation_passed:
            print("✅ VALIDATION COMPLETE - Search views setup and validation successful")
            sys.exit(0)
        else:
            if not setup_result:
                print("❌ SETUP FAILED - Error setting up search views")
            
            if not validation_passed:
                print("❌ VALIDATION FAILED - Search views don't match expected configuration") 
                print(f"FAILURE DETAILS:")
                for field, details in validation_failures.items():
                    print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
                print(f"Total errors: {len(validation_failures)} fields mismatched")
            
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
