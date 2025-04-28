# src/pdf_extractor/arangodb/crud/message_history.py
"""CRUD operations specific to message history."""

import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from loguru import logger

from arango.database import StandardDatabase

# Import necessary functions from generic CRUD and config
from .generic import create_document, get_document, update_document, delete_document, query_documents
from ..message_history_config import (
    MESSAGE_COLLECTION_NAME,
    MESSAGE_EDGE_COLLECTION_NAME,
    RELATIONSHIP_TYPE_NEXT,
    MESSAGE_TYPE_USER # Add missing import
)

# -------------------- Message History Specific Operations --------------------

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