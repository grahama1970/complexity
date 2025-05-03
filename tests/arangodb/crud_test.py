#!/usr/bin/env python3
"""
Test for CRUD operations with embedding integration.
This script tests whether the generic CRUD operations from src/complexity/arangodb/crud/generic.py
work properly with our embedding solution.
"""

import sys
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} | {level:<7} | {message}"
)

# Import ArangoDB modules
try:
    from arango import ArangoClient
    from arango.database import StandardDatabase
    from arango.exceptions import (
        DocumentInsertError,
        DocumentGetError,
        DocumentUpdateError,
        DocumentDeleteError
    )
except ImportError:
    logger.error("Could not import python-arango. Please install it using: pip install python-arango")
    sys.exit(1)

# Import CRUD operations from our module (test both versions)
try:
    # Import from generic.py (original)
    import sys
    sys.path.append("/home/graham/workspace/experiments/complexity")
    from src.complexity.arangodb.crud.generic import (
        create_document,
        get_document,
        update_document,
        delete_document,
        query_documents
    )
    logger.info("Successfully imported CRUD operations from generic.py")
    
    # Import from db_operations.py (consolidated version)
    from src.complexity.arangodb.db_operations import (
        create_document as create_document_consolidated,
        get_document as get_document_consolidated,
        update_document as update_document_consolidated,
        delete_document as delete_document_consolidated,
        query_documents as query_documents_consolidated
    )
    logger.info("Successfully imported consolidated CRUD operations from db_operations.py")
    
    # Import embedding utilities
    from src.complexity.arangodb.embedding_utils import get_embedding, get_EmbedderModel
    logger.info("Successfully imported embedding utilities")
    
except ImportError as e:
    logger.error(f"Could not import required modules: {e}")
    sys.exit(1)

# Database connection details
DB_HOST = "localhost"
DB_PORT = 8529
DB_NAME = "complexity_test"
DB_USER = "root"
DB_PASS = ""  # Fill in if needed

# Test collection names
TEST_COLLECTION = "crud_test"
TEST_VIEW = "crud_test_view"

def connect_to_db() -> StandardDatabase:
    """Connect to ArangoDB and return database handle."""
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts=f"http://{DB_HOST}:{DB_PORT}")
        sys_db = client.db("_system", username=DB_USER, password=DB_PASS)
        
        # Create test database if it doesn't exist
        if not sys_db.has_database(DB_NAME):
            sys_db.create_database(DB_NAME)
            logger.info(f"Created database: {DB_NAME}")
        
        # Connect to the test database
        db = client.db(DB_NAME, username=DB_USER, password=DB_PASS)
        
        # Create test collection if it doesn't exist
        if not db.has_collection(TEST_COLLECTION):
            db.create_collection(TEST_COLLECTION)
            logger.info(f"Created collection: {TEST_COLLECTION}")
        
        # Create test view if it doesn't exist
        if not db.has_view(TEST_VIEW):
            # Create an Analyzer
            if not db.has_analyzer("text_en"):
                db.create_analyzer({
                    "name": "text_en",
                    "type": "text",
                    "properties": {
                        "locale": "en.utf-8",
                        "stemming": True,
                        "case": "lower",
                        "stop_words": []
                    }
                })
                logger.info("Created text analyzer: text_en")
            
            # Create ArangoSearch view
            db.create_arangosearch_view(
                name=TEST_VIEW,
                properties={
                    "links": {
                        TEST_COLLECTION: {
                            "includeAllFields": True,
                            "fields": {
                                "content": {
                                    "analyzers": ["text_en"]
                                }
                            }
                        }
                    }
                }
            )
            logger.info(f"Created ArangoSearch view: {TEST_VIEW}")
        
        return db
    
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        sys.exit(1)

def clean_up(db: StandardDatabase):
    """Clean up test data after tests are done."""
    try:
        # Truncate test collection instead of deleting it
        db.collection(TEST_COLLECTION).truncate()
        logger.info(f"Truncated collection: {TEST_COLLECTION}")
    except Exception as e:
        logger.error(f"Clean up error: {e}")


def test_crud_with_embeddings():
    """
    Test CRUD operations with embeddings.
    
    This test verifies that:
    1. We can create documents with embeddings
    2. We can retrieve documents with embeddings
    3. We can update documents and their embeddings
    4. We can query documents using embedding vector search
    """
    logger.info("Starting CRUD tests with embeddings")
    
    # Connect to the database
    db = connect_to_db()
    
    # ==========================================================================
    # TEST 1: Create documents with embeddings using original CRUD operations
    # ==========================================================================
    logger.info("Test 1: Create documents with embeddings (original CRUD)")
    
    # Get embedding model info
    embedding_model = get_EmbedderModel()
    logger.info(f"Using embedding model: {embedding_model['model']} ({embedding_model['dimensions']} dimensions)")
    
    # Create test documents
    test_docs = []
    for i in range(3):
        content = f"This is test document {i} about embeddings and vector search."
        # Generate embedding for the content
        embedding = get_embedding(content)
        
        # Create document
        doc = {
            "title": f"Test Document {i}",
            "content": content,
            "embedding": embedding,
            "tags": ["test", "embedding", f"doc{i}"]
        }
        test_docs.append(doc)
    
    # Insert all test documents
    created_docs = []
    for doc in test_docs:
        result = create_document(db, TEST_COLLECTION, doc)
        if not result:
            logger.error("Failed to create document")
            clean_up(db)
            sys.exit(1)
        created_docs.append(result)
    
    logger.info(f"Created {len(created_docs)} documents with embeddings")
    
    # Verify document structure
    first_doc = created_docs[0]
    assert "embedding" in first_doc, "Embedding field not found in created document"
    assert isinstance(first_doc["embedding"], list), "Embedding is not a list"
    assert len(first_doc["embedding"]) == embedding_model["dimensions"], f"Embedding has wrong dimensions: {len(first_doc['embedding'])}"
    
    # ==========================================================================
    # TEST 2: Retrieve documents with embeddings
    # ==========================================================================
    logger.info("Test 2: Retrieve documents with embeddings")
    
    # Retrieve a document
    retrieved_doc = get_document(db, TEST_COLLECTION, first_doc["_key"])
    assert retrieved_doc, "Failed to retrieve document"
    assert "embedding" in retrieved_doc, "Embedding field not found in retrieved document"
    
    # ==========================================================================
    # TEST 3: Update documents with new embeddings
    # ==========================================================================
    logger.info("Test 3: Update documents with new embeddings")
    
    # Update content and embedding
    updated_content = "This is an updated document about embeddings and vector search with new information."
    updated_embedding = get_embedding(updated_content)
    
    # Update document
    update_result = update_document(
        db, 
        TEST_COLLECTION, 
        first_doc["_key"], 
        {
            "content": updated_content,
            "embedding": updated_embedding,
            "tags": ["test", "embedding", "updated"]
        }
    )
    
    assert update_result, "Failed to update document"
    assert update_result["content"] == updated_content, "Content not updated correctly"
    assert "embedding" in update_result, "Embedding field lost during update"
    
    # ==========================================================================
    # TEST 4: Perform a semantic search using AQL
    # ==========================================================================
    logger.info("Test 4: Perform a semantic search using AQL")
    
    # Create a query vector
    query_text = "vector search embeddings"
    query_embedding = get_embedding(query_text)
    
    # Perform AQL search with vector search
    aql = """
    FOR doc IN @@collection
    LET score = COSINE_SIMILARITY(doc.embedding, @query_vector)
    FILTER score > 0.7
    SORT score DESC
    LIMIT 10
    RETURN {
        doc: doc,
        score: score
    }
    """
    
    bind_vars = {
        "@collection": TEST_COLLECTION,
        "query_vector": query_embedding
    }
    
    try:
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        search_results = list(cursor)
        
        logger.info(f"Semantic search found {len(search_results)} results")
        
        # Print search results
        for i, result in enumerate(search_results):
            doc = result["doc"]
            score = result["score"]
            logger.info(f"Result {i+1}: {doc['title']} (Score: {score:.4f})")
            
        assert len(search_results) > 0, "No search results found"
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        clean_up(db)
        sys.exit(1)
    
    # ==========================================================================
    # TEST 5: Test consolidated CRUD operations from db_operations.py
    # ==========================================================================
    logger.info("Test 5: Test consolidated CRUD operations")
    
    # Create a document using consolidated operations
    new_doc = {
        "title": "Consolidated CRUD Test",
        "content": "This document tests the consolidated CRUD operations with embeddings",
        "embedding": get_embedding("Consolidated CRUD operations with embeddings"),
        "tags": ["test", "consolidated", "embedding"]
    }
    
    consolidated_doc = create_document_consolidated(db, TEST_COLLECTION, new_doc)
    assert consolidated_doc, "Failed to create document with consolidated CRUD"
    assert "embedding" in consolidated_doc, "Embedding field not found in document created with consolidated CRUD"
    
    # Retrieve with consolidated operations
    retrieved_consolidated = get_document_consolidated(db, TEST_COLLECTION, consolidated_doc["_key"])
    assert retrieved_consolidated, "Failed to retrieve document with consolidated CRUD"
    assert "embedding" in retrieved_consolidated, "Embedding field not found in retrieved consolidated document"
    
    # Update with consolidated operations
    updated_consolidated = update_document_consolidated(
        db,
        TEST_COLLECTION,
        consolidated_doc["_key"],
        {
            "content": "Updated consolidated CRUD test content",
            "embedding": get_embedding("Updated consolidated CRUD operations with embeddings")
        }
    )
    assert updated_consolidated, "Failed to update document with consolidated CRUD"
    assert "embedding" in updated_consolidated, "Embedding field lost during consolidated update"
    
    # ==========================================================================
    # TEST 6: Clean up and summary
    # ==========================================================================
    logger.info("Test 6: Clean up and summary")
    
    # Delete documents
    for doc in created_docs:
        success = delete_document(db, TEST_COLLECTION, doc["_key"])
        assert success, f"Failed to delete document {doc['_key']}"
    
    # Delete the consolidated test document
    delete_result = delete_document_consolidated(db, TEST_COLLECTION, consolidated_doc["_key"])
    assert delete_result, "Failed to delete document with consolidated CRUD"
    
    # Final clean up
    clean_up(db)
    
    logger.info("All tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_crud_with_embeddings()
    except Exception as e:
        logger.exception(f"Test failed with exception: {e}")
        sys.exit(1)