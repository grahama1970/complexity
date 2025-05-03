"""
Enhanced database operations for ArangoDB with embedding generation.

This module extends db_operations.py to automatically generate and store 
embeddings for documents during creation and updates.
"""

import sys
from typing import Dict, Any, List, Optional, Union
from loguru import logger

from arango.database import StandardDatabase

# Import base db operations
from complexity.arangodb.db_operations import (
    create_document as base_create_document,
    update_document as base_update_document,
    get_document,
    delete_document,
    query_documents
)

# Import embedding utilities
from complexity.arangodb.embedding_utils import get_embedding
from complexity.arangodb.config import EMBEDDING_FIELD, COLLECTION_NAME, MESSAGES_COLLECTION_NAME

# Collections that should have embeddings
EMBEDDING_COLLECTIONS = [COLLECTION_NAME, MESSAGES_COLLECTION_NAME]

def create_document_with_embedding(
    db: StandardDatabase,
    collection_name: str,
    document: Dict[str, Any],
    document_key: Optional[str] = None,
    return_new: bool = True,
    embedding_field: str = EMBEDDING_FIELD,
    text_field: str = "content"
) -> Optional[Dict[str, Any]]:
    """
    Insert a document into a collection with automatic embedding generation.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document: Document data to insert
        document_key: Optional key for the document (auto-generated if not provided)
        return_new: Whether to return the new document
        embedding_field: Field name to store embedding
        text_field: Field to use for embedding generation

    Returns:
        Optional[Dict[str, Any]]: The inserted document or metadata if successful, None otherwise
    """
    # Create a copy of the document to avoid modifying the original
    doc_copy = document.copy()
    
    # Check if this collection should have embeddings and if text content exists
    if collection_name in EMBEDDING_COLLECTIONS and text_field in doc_copy:
        text_content = doc_copy.get(text_field)
        if text_content and isinstance(text_content, str):
            # Generate embedding
            logger.info(f"Generating embedding for document in {collection_name}")
            embedding = get_embedding(text_content)
            
            if embedding:
                # Store embedding in document
                doc_copy[embedding_field] = embedding
                logger.info(f"Added embedding ({len(embedding)} dimensions) to document")
            else:
                logger.warning(f"Failed to generate embedding for document in {collection_name}")
    
    # Create document with base operation
    return base_create_document(db, collection_name, doc_copy, document_key, return_new)


def update_document_with_embedding(
    db: StandardDatabase,
    collection_name: str,
    document_key: str,
    updates: Dict[str, Any],
    return_new: bool = True,
    check_rev: bool = False,
    rev: Optional[str] = None,
    embedding_field: str = EMBEDDING_FIELD,
    text_field: str = "content"
) -> Optional[Dict[str, Any]]:
    """
    Update a document with new values and regenerate embedding if text content changes.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to update
        updates: Dictionary of fields to update
        return_new: Whether to return the updated document
        check_rev: Whether to check document revision
        rev: Document revision (required if check_rev is True)
        embedding_field: Field name to store embedding
        text_field: Field to use for embedding generation

    Returns:
        Optional[Dict[str, Any]]: The updated document if successful, None otherwise
    """
    # Create a copy of the updates to avoid modifying the original
    updates_copy = updates.copy()
    
    # Check if this collection should have embeddings and if text content is being updated
    if collection_name in EMBEDDING_COLLECTIONS and text_field in updates_copy:
        text_content = updates_copy.get(text_field)
        if text_content and isinstance(text_content, str):
            # Generate new embedding for updated text
            logger.info(f"Regenerating embedding for updated document in {collection_name}")
            embedding = get_embedding(text_content)
            
            if embedding:
                # Store updated embedding in document
                updates_copy[embedding_field] = embedding
                logger.info(f"Updated embedding ({len(embedding)} dimensions) for document")
            else:
                logger.warning(f"Failed to generate embedding for updated document in {collection_name}")
    
    # Update document with base operation
    return base_update_document(db, collection_name, document_key, updates_copy, return_new, check_rev, rev)


# Provide access to other base operations
__all__ = [
    'create_document_with_embedding',
    'update_document_with_embedding',
    'get_document',
    'delete_document',
    'query_documents'
]