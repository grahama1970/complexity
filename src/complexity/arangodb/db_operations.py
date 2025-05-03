"""
Consolidated database operations for ArangoDB integration.

This module provides a unified interface for database operations including:
- Generic CRUD operations for any collection
- Message history operations
- Relationship management between documents
"""

import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from loguru import logger

from arango.database import StandardDatabase
from arango.exceptions import (
    DocumentInsertError,
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError,
    AQLQueryExecuteError
)

# Import configuration
from complexity.arangodb.config import (
    COLLECTION_NAME,
    EDGE_COLLECTION_NAME,
    MESSAGES_COLLECTION_NAME,
    GRAPH_NAME
)

# Message history configuration
# Defined here rather than importing from archived configs
MESSAGE_COLLECTION_NAME = MESSAGES_COLLECTION_NAME  # Use from main config
MESSAGE_EDGE_COLLECTION_NAME = "message_links"
MESSAGE_GRAPH_NAME = "message_graph"

# Message Types
MESSAGE_TYPE_USER = "USER"
MESSAGE_TYPE_AGENT = "AGENT"
MESSAGE_TYPE_SYSTEM = "SYSTEM"

# Relationship Types
RELATIONSHIP_TYPE_NEXT = "NEXT"
RELATIONSHIP_TYPE_REFERS_TO = "REFERS_TO"

# =============================================================================
# GENERIC CRUD OPERATIONS
# =============================================================================

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
        merged_doc["_key"] = document_key  # Ensure _key is set

        # 3. Replace the document
        # Add revision check if needed
        params = {}
        if check_rev:
            # If check_rev is True, we MUST use the _rev from the fetched doc
            if "_rev" not in existing_doc:
                logger.warning(f"Revision check requested but _rev not found in fetched document {document_key}")
                # Decide how to handle: error out or proceed without check? Proceeding without for now.
                check_rev = False  # Disable check if _rev is missing
            else:
                # Note: python-arango's replace doesn't directly use check_rev param like update/delete
                # Instead, we include _rev in the document body for replace.
                merged_doc["_rev"] = existing_doc["_rev"]

        # Use replace instead of update
        result = collection.replace(
            merged_doc,  # Pass the entire merged document
            return_new=return_new,
            # **params # 'rev' is passed within merged_doc if check_rev was possible
        )

        logger.info(f"Replaced document in {collection_name}: {document_key}")
        return result["new"] if return_new and "new" in result else result

    except DocumentUpdateError as e:  # Replace might still raise DocumentUpdateError on rev mismatch
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


# =============================================================================
# MESSAGE HISTORY OPERATIONS
# =============================================================================

def create_message(
    db: StandardDatabase,
    conversation_id: str,
    message_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    previous_message_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Create a message in the message history collection.

    Args:
        db: ArangoDB database handle
        conversation_id: ID of the conversation
        message_type: Type of message (USER, AGENT, SYSTEM)
        content: Message content
        metadata: Optional metadata
        timestamp: Optional timestamp (ISO format)
        previous_message_key: Optional key of the previous message to link to

    Returns:
        Optional[Dict[str, Any]]: The created message if successful, None otherwise
    """
    # Prepare message
    message_key = str(uuid.uuid4())
    message = {
        "_key": message_key,
        "conversation_id": conversation_id,
        "message_type": message_type,
        "content": content,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {}
    }

    # Create the message
    result = create_document(db, MESSAGE_COLLECTION_NAME, message)

    # Create relationship if previous message is provided
    if result and previous_message_key:
        # Create edge between messages
        edge = {
            "_from": f"{MESSAGE_COLLECTION_NAME}/{previous_message_key}",
            "_to": f"{MESSAGE_COLLECTION_NAME}/{message_key}",
            "type": RELATIONSHIP_TYPE_NEXT,
            "timestamp": message["timestamp"]
        }
        create_document(db, MESSAGE_EDGE_COLLECTION_NAME, edge)

    return result


def get_message(
    db: StandardDatabase,
    message_key: str
) -> Optional[Dict[str, Any]]:
    """
    Get a message by key.

    Args:
        db: ArangoDB database handle
        message_key: Key of the message

    Returns:
        Optional[Dict[str, Any]]: The message if found, None otherwise
    """
    return get_document(db, MESSAGE_COLLECTION_NAME, message_key)


def update_message(
    db: StandardDatabase,
    message_key: str,
    updates: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Update a message.

    Args:
        db: ArangoDB database handle
        message_key: Key of the message
        updates: Fields to update

    Returns:
        Optional[Dict[str, Any]]: The updated message if successful, None otherwise
    """
    return update_document(db, MESSAGE_COLLECTION_NAME, message_key, updates)


def delete_message(
    db: StandardDatabase,
    message_key: str,
    delete_relationships: bool = True
) -> bool:
    """
    Delete a message.

    Args:
        db: ArangoDB database handle
        message_key: Key of the message
        delete_relationships: Whether to delete related edges

    Returns:
        bool: True if successful, False otherwise
    """
    # Delete relationships if requested
    if delete_relationships:
        try:
            # Delete outgoing edges
            aql_out = f"""
            FOR edge IN {MESSAGE_EDGE_COLLECTION_NAME}
            FILTER edge._from == @from
            RETURN edge._key
            """
            cursor_out = db.aql.execute(
                aql_out,
                bind_vars={"from": f"{MESSAGE_COLLECTION_NAME}/{message_key}"}
            )
            for edge_key in cursor_out:
                delete_document(db, MESSAGE_EDGE_COLLECTION_NAME, edge_key)

            # Delete incoming edges
            aql_in = f"""
            FOR edge IN {MESSAGE_EDGE_COLLECTION_NAME}
            FILTER edge._to == @to
            RETURN edge._key
            """
            cursor_in = db.aql.execute(
                aql_in,
                bind_vars={"to": f"{MESSAGE_COLLECTION_NAME}/{message_key}"}
            )
            for edge_key in cursor_in:
                delete_document(db, MESSAGE_EDGE_COLLECTION_NAME, edge_key)

        except Exception as e:
            logger.error(f"Error deleting message relationships: {e}")
            return False

    # Delete the message
    return delete_document(db, MESSAGE_COLLECTION_NAME, message_key)


def get_conversation_messages(
    db: StandardDatabase,
    conversation_id: str,
    limit: int = 100,
    offset: int = 0,
    sort_order: str = "asc"
) -> List[Dict[str, Any]]:
    """
    Get all messages for a conversation.

    Args:
        db: ArangoDB database handle
        conversation_id: ID of the conversation
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        sort_order: Sort order ("asc" or "desc")

    Returns:
        List[Dict[str, Any]]: List of messages
    """
    # Validate sort order
    sort_direction = "ASC" if sort_order.lower() == "asc" else "DESC"

    # Build filter and sort clauses
    filter_clause = "FILTER doc.conversation_id == @conversation_id"
    sort_clause = f"SORT doc.timestamp {sort_direction}"

    # Query messages
    return query_documents(
        db,
        MESSAGE_COLLECTION_NAME,
        filter_clause,
        sort_clause,
        limit,
        offset,
        {"conversation_id": conversation_id}
    )


def delete_conversation(
    db: StandardDatabase,
    conversation_id: str
) -> bool:
    """
    Delete all messages for a conversation.

    Args:
        db: ArangoDB database handle
        conversation_id: ID of the conversation

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get all message keys for the conversation
        aql_keys = f"""
        FOR doc IN {MESSAGE_COLLECTION_NAME}
        FILTER doc.conversation_id == @conversation_id
        RETURN doc._key
        """
        cursor_keys = db.aql.execute(aql_keys, bind_vars={"conversation_id": conversation_id})
        message_keys = list(cursor_keys)

        if not message_keys:
            logger.info(f"No messages found for conversation: {conversation_id}")
            return True

        # Delete each message (and its relationships)
        all_deleted = True
        for key in message_keys:
            if not delete_message(db, key, delete_relationships=True):
                all_deleted = False
                logger.error(f"Failed to delete message {key} during conversation deletion.")

        if all_deleted:
            logger.info(f"Successfully deleted conversation: {conversation_id}")
        else:
            logger.warning(f"Partial deletion for conversation: {conversation_id}")

        return all_deleted

    except Exception as e:
        logger.exception(f"Error deleting conversation {conversation_id}: {e}")
        return False


# =============================================================================
# RELATIONSHIP MANAGEMENT OPERATIONS
# =============================================================================

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
        "_to": f"{COLLECTION_NAME}/{document_key}",
        "type": relationship_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(metadata or {})
    }
    # Use the main document edge collection here
    return create_document(db, EDGE_COLLECTION_NAME, edge)


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
            "graph_name": GRAPH_NAME,
            "rel_type": relationship_type,
            "doc_collection": COLLECTION_NAME
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
        start_vertex = f"{COLLECTION_NAME}/{document_key}"
        aql = f"""
        FOR v, e, p IN 1..{max_depth} ANY @start_vertex GRAPH @graph_name
        FILTER @rel_type == null OR e.type == @rel_type
        FILTER IS_SAME_COLLECTION(@msg_collection, v)
        RETURN DISTINCT v
        """
        bind_vars = {
            "start_vertex": start_vertex,
            "graph_name": GRAPH_NAME,
            "rel_type": relationship_type,
            "msg_collection": MESSAGE_COLLECTION_NAME
        }
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        return list(cursor)
    except Exception as e:
        logger.exception(f"Error getting messages for document {document_key}: {e}")
        return []


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
        from_doc_key: Key of the source document (in COLLECTION_NAME)
        to_doc_key: Key of the target document (in COLLECTION_NAME)
        relationship_type: Type/category of the relationship
        rationale: Explanation for the relationship
        attributes: Optional additional metadata for the edge

    Returns:
        Optional[Dict[str, Any]]: The created edge document if successful, None otherwise
    """
    edge = {
        "_from": f"{COLLECTION_NAME}/{from_doc_key}",
        "_to": f"{COLLECTION_NAME}/{to_doc_key}",
        "type": relationship_type,
        "rationale": rationale,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(attributes or {})
    }
    return create_document(db, EDGE_COLLECTION_NAME, edge)


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
    return delete_document(db, EDGE_COLLECTION_NAME, edge_key, ignore_missing=True)