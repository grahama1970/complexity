# src/pdf_extractor/arangodb/crud/generic.py
"""Generic CRUD operations for ArangoDB collections."""

import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from loguru import logger

from arango.database import StandardDatabase
from arango.exceptions import (
    DocumentInsertError,
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError
)

# -------------------- Generic CRUD Operations --------------------

def create_document(
    db: StandardDatabase,
    collection_name: str,
    document: Dict[str, Any],
    document_key: Optional[str] = None,
    return_new: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Insert a document into a collection.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document: Document data to insert
        document_key: Optional key for the document (auto-generated if not provided)
        return_new: Whether to return the new document

    Returns:
        Optional[Dict[str, Any]]: The inserted document or metadata if successful, None otherwise
    """
    try:
        # Generate a key if not provided
        if document_key:
            document["_key"] = document_key
        elif "_key" not in document:
            document["_key"] = str(uuid.uuid4())

        # Add timestamp if not present
        if "timestamp" not in document:
            document["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Get the collection and insert document
        collection = db.collection(collection_name)
        result = collection.insert(document, return_new=return_new)

        logger.info(f"Created document in {collection_name}: {result.get('_key', result)}")
        return result["new"] if return_new and "new" in result else result

    except DocumentInsertError as e:
        logger.error(f"Failed to create document in {collection_name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error creating document in {collection_name}: {e}")
        return None

def get_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document by key.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to retrieve

    Returns:
        Optional[Dict[str, Any]]: The document if found, None otherwise
    """
    try:
        collection = db.collection(collection_name)
        document = collection.get(document_key)

        if document:
            logger.debug(f"Retrieved document from {collection_name}: {document_key}")
        else:
            logger.warning(f"Document not found in {collection_name}: {document_key}")

        return document

    except DocumentGetError as e:
        logger.error(f"Failed to get document from {collection_name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error getting document from {collection_name}: {e}")
        return None


def update_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str,
    updates: Dict[str, Any],
    return_new: bool = True,
    check_rev: bool = False,
    rev: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Update a document with new values.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to update
        updates: Dictionary of fields to update
        return_new: Whether to return the updated document
        check_rev: Whether to check document revision
        rev: Document revision (required if check_rev is True)

    Returns:
        Optional[Dict[str, Any]]: The updated document if successful, None otherwise
    """
    try:
        collection = db.collection(collection_name)

        # 1. Get the existing document
        existing_doc = collection.get(document_key)
        if not existing_doc:
            logger.error(f"Document {document_key} not found in {collection_name} for update.")
            return None

        # 2. Merge updates into the existing document
        merged_doc = existing_doc.copy()
        merged_doc.update(updates)

        # Add/update timestamp
        merged_doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        # Ensure required fields like _key are present for replace
        merged_doc["_key"] = document_key # Ensure _key is set

        # 3. Replace the document
        # Add revision check if needed
        params = {}
        if check_rev:
            # If check_rev is True, we MUST use the _rev from the fetched doc
            if "_rev" not in existing_doc:
                 logger.warning(f"Revision check requested but _rev not found in fetched document {document_key}")
                 # Decide how to handle: error out or proceed without check? Proceeding without for now.
                 check_rev = False # Disable check if _rev is missing
            else:
                 # Note: python-arango's replace doesn't directly use check_rev param like update/delete
                 # Instead, we include _rev in the document body for replace.
                 merged_doc["_rev"] = existing_doc["_rev"]


        # Use replace instead of update
        result = collection.replace(
            merged_doc, # Pass the entire merged document
            return_new=return_new,
            # **params # 'rev' is passed within merged_doc if check_rev was possible
        )

        logger.info(f"Replaced document in {collection_name}: {document_key}")
        return result["new"] if return_new and "new" in result else result

    except DocumentUpdateError as e: # Replace might still raise DocumentUpdateError on rev mismatch
        logger.error(f"Failed to update document in {collection_name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error updating document in {collection_name}: {e}")
        return None


def delete_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str,
    ignore_missing: bool = True,
    return_old: bool = False,
    check_rev: bool = False,
    rev: Optional[str] = None
) -> bool:
    """
    Delete a document from a collection.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to delete
        ignore_missing: Whether to ignore if document doesn't exist
        return_old: Whether to return the old document
        check_rev: Whether to check document revision
        rev: Document revision (required if check_rev is True)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the collection and delete document
        collection = db.collection(collection_name)

        # Add revision if needed
        params = {}
        if check_rev and rev:
            params["rev"] = rev

        result = collection.delete(
            document=document_key,
            ignore_missing=ignore_missing,
            return_old=return_old,
            check_rev=check_rev,
            **params
        )

        if result is False and ignore_missing:
            logger.info(f"Document not found for deletion in {collection_name}: {document_key}")
            return True

        logger.info(f"Deleted document from {collection_name}: {document_key}")
        return True

    except DocumentDeleteError as e:
        logger.error(f"Failed to delete document from {collection_name}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error deleting document from {collection_name}: {e}")
        return False

def query_documents(
    db: StandardDatabase,
    collection_name: str,
    filter_clause: str = "",
    sort_clause: str = "",
    limit: int = 100,
    offset: int = 0,
    bind_vars: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Query documents from a collection.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        filter_clause: AQL filter clause (e.g., "FILTER doc.field == @value")
        sort_clause: AQL sort clause (e.g., "SORT doc.field DESC")
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        bind_vars: Bind variables for the query

    Returns:
        List[Dict[str, Any]]: List of documents matching the query
    """
    try:
        # Build AQL query
        aql = f"""
        FOR doc IN {collection_name}
        {filter_clause}
        {sort_clause}
        LIMIT {offset}, {limit}
        RETURN doc
        """

        # Set default bind variables
        if bind_vars is None:
            bind_vars = {}

        # Execute query
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)

        logger.info(f"Query returned {len(results)} documents from {collection_name}")
        return results

    except Exception as e:
        logger.exception(f"Error querying documents from {collection_name}: {e}")
        return []