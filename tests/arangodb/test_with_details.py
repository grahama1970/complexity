#!/usr/bin/env python3
"""
Create a test document with embedding and verify it's stored in the database.
"""
import sys
import uuid
import json
import time
from typing import Dict, Any
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level:^8} | {message}")

# Import database operations
from complexity.arangodb.arango_setup import connect_arango, ensure_database
from complexity.arangodb.embedding_utils import get_embedding
from complexity.arangodb.db_operations import create_document

# Generate a unique test document
test_key = f"test_embed_{uuid.uuid4().hex[:8]}"
test_content = f"This is a test document with ID {test_key} to verify that embeddings are properly generated and stored in ArangoDB."

test_doc = {
    "_key": test_key,
    "title": "Embedding Test Document",
    "content": test_content,
    "created_at": time.time()
}

logger.info(f"Created test document with key: {test_key}")
logger.info(f"Content: {test_content[:50]}...")

# Connect to database
logger.info("Connecting to ArangoDB...")
client = connect_arango()
db = ensure_database(client)
logger.info(f"Connected to database: {db.name}")

# Create document with embedding
logger.info("Creating document with embedding...")
# Generate embedding manually
embedding = get_embedding(test_content)
if embedding:
    logger.info(f"Generated embedding with {len(embedding)} dimensions")
    test_doc["embedding"] = embedding
else:
    logger.warning("Could not generate embedding - skipping embedding test")

# Create document with the embedding field
doc_with_embedding = create_document(db, "test_docs", test_doc)

if not doc_with_embedding:
    logger.error("Failed to create document")
    sys.exit(1)

# Verify embedding was generated
if "embedding" in doc_with_embedding:
    embedding_length = len(doc_with_embedding["embedding"])
    logger.info(f"SUCCESS: Document created with embedding of {embedding_length} dimensions")
    
    # Fetch the document directly from the database to verify it's really there
    logger.info("Verifying document in database...")
    collection = db.collection("test_docs")
    stored_doc = collection.get(test_key)
    
    if stored_doc and "embedding" in stored_doc:
        stored_embedding_length = len(stored_doc["embedding"])
        logger.info(f"SUCCESS: Document verified in database with {stored_embedding_length} dimensions")
        
        # Show a small sample of the embedding vector to verify it's a real embedding
        embedding_sample = stored_doc["embedding"][:5]
        logger.info(f"Embedding sample (first 5 values): {embedding_sample}")
        
        # Show some database stats to verify we're working with a real database
        doc_count = collection.count()
        logger.info(f"Total documents in test_docs collection: {doc_count}")
        
        # Clean up
        logger.info(f"Cleaning up test document: {test_key}")
        collection.delete(test_key)
    else:
        logger.error("Document not found in database or missing embedding")
else:
    logger.error("No embedding was generated for the document")