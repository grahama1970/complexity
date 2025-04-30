# src/pdf_extractor/arangodb/crud/relationships.py
"""CRUD operations specific to relationships between collections."""

import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from loguru import logger

from arango.database import StandardDatabase

# Import necessary functions using absolute paths
from complexity.arangodb.crud.generic import create_document, query_documents, delete_document
from complexity.arangodb._archive.message_history_config import (
    MESSAGE_COLLECTION_NAME,
    RELATIONSHIP_TYPE_REFERS_TO
)
from complexity.arangodb.config import (
    COLLECTION_NAME as DOC_COLLECTION_NAME, # Import COLLECTION_NAME and alias
    EDGE_COLLECTION_NAME as DOC_EDGE_COLLECTION_NAME, # Use the main edge collection
    GRAPH_NAME as DOC_GRAPH_NAME # Import the main graph name
)

# -------------------- Relationship Management Operations --------------------

def link_message_to_document(
    db: StandardDatabase,
    message_key: str,
    document_key: str,
    relationship_type: str = RELATIONSHIP_TYPE_REFERS_TO,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Create an edge linking a message to a document in the main document graph.

    Args:
        db: ArangoDB database handle
        message_key: Key of the message document
        document_key: Key of the document (e.g., lesson learned)
        relationship_type: Type of relationship (e.g., REFERS_TO)
        metadata: Optional metadata for the edge

    Returns:
        Optional[Dict[str, Any]]: The created edge document if successful, None otherwise
    """
    edge = {
        "_from": f"{MESSAGE_COLLECTION_NAME}/{message_key}",
        "_to": f"{DOC_COLLECTION_NAME}/{document_key}",
        "type": relationship_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(metadata or {})
    }
    # Use the main document edge collection here
    return create_document(db, DOC_EDGE_COLLECTION_NAME, edge)


def get_documents_for_message(
    db: StandardDatabase,
    message_key: str,
    relationship_type: Optional[str] = None,
    max_depth: int = 1
) -> List[Dict[str, Any]]:
    """
    Get documents related to a specific message using the main document graph.

    Args:
        db: ArangoDB database handle
        message_key: Key of the starting message
        relationship_type: Optional type of relationship to filter by
        max_depth: Maximum traversal depth

    Returns:
        List[Dict[str, Any]]: List of related documents
    """
    try:
        start_vertex = f"{MESSAGE_COLLECTION_NAME}/{message_key}"
        aql = f"""
        FOR v, e, p IN 1..{max_depth} ANY @start_vertex GRAPH @graph_name
        FILTER @rel_type == null OR e.type == @rel_type
        FILTER IS_SAME_COLLECTION(@doc_collection, v)
        RETURN DISTINCT v
        """
        bind_vars = {
            "start_vertex": start_vertex,
            "graph_name": DOC_GRAPH_NAME,
            "rel_type": relationship_type,
            "doc_collection": DOC_COLLECTION_NAME
        }
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        return list(cursor)
    except Exception as e:
        logger.exception(f"Error getting documents for message {message_key}: {e}")
        return []


def get_messages_for_document(
    db: StandardDatabase,
    document_key: str,
    relationship_type: Optional[str] = None,
    max_depth: int = 1
) -> List[Dict[str, Any]]:
    """
    Get messages related to a specific document using the main document graph.

    Args:
        db: ArangoDB database handle
        document_key: Key of the starting document
        relationship_type: Optional type of relationship to filter by
        max_depth: Maximum traversal depth

    Returns:
        List[Dict[str, Any]]: List of related messages
    """
    try:
        start_vertex = f"{DOC_COLLECTION_NAME}/{document_key}"
        aql = f"""
        FOR v, e, p IN 1..{max_depth} ANY @start_vertex GRAPH @graph_name
        FILTER @rel_type == null OR e.type == @rel_type
        FILTER IS_SAME_COLLECTION(@msg_collection, v)
        RETURN DISTINCT v
        """
        bind_vars = {
            "start_vertex": start_vertex,
            "graph_name": DOC_GRAPH_NAME,
            "rel_type": relationship_type,
            "msg_collection": MESSAGE_COLLECTION_NAME
        }
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        return list(cursor)
    except Exception as e:
        logger.exception(f"Error getting messages for document {document_key}: {e}")
        return []

# --- Generic Relationship Functions for Main Document Graph ---

def create_relationship(
    db: StandardDatabase,
    from_doc_key: str,
    to_doc_key: str,
    relationship_type: str,
    rationale: str,
    attributes: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Create a generic edge between two documents in the main document graph.

    Args:
        db: ArangoDB database handle
        from_doc_key: Key of the source document (in DOC_COLLECTION_NAME)
        to_doc_key: Key of the target document (in DOC_COLLECTION_NAME)
        relationship_type: Type/category of the relationship
        rationale: Explanation for the relationship
        attributes: Optional additional metadata for the edge

    Returns:
        Optional[Dict[str, Any]]: The created edge document if successful, None otherwise
    """
    edge = {
        "_from": f"{DOC_COLLECTION_NAME}/{from_doc_key}",
        "_to": f"{DOC_COLLECTION_NAME}/{to_doc_key}",
        "type": relationship_type,
        "rationale": rationale,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(attributes or {})
    }
    return create_document(db, DOC_EDGE_COLLECTION_NAME, edge)


def delete_relationship_by_key(
    db: StandardDatabase,
    edge_key: str
) -> bool:
    """
    Delete a relationship edge by its key from the main document edge collection.

    Args:
        db: ArangoDB database handle
        edge_key: The _key of the edge document to delete

    Returns:
        bool: True if successful or edge already gone, False on error
    """
    return delete_document(db, DOC_EDGE_COLLECTION_NAME, edge_key, ignore_missing=True)