# src/pdf_extractor/arangodb/crud/__init__.py
"""Initialize the CRUD package and export key functions."""

from .generic import (
    create_document,
    get_document,
    update_document,
    delete_document,
    query_documents
)
from .message_history import (
    create_message,
    get_message,
    update_message,
    delete_message,
    get_conversation_messages,
    delete_conversation
)
from .relationships import (
    link_message_to_document,
    get_documents_for_message,
    get_messages_for_document,
    create_relationship, # Add new generic relationship function
    delete_relationship_by_key # Add new generic relationship function
)

__all__ = [
    "create_document",
    "get_document",
    "update_document",
    "delete_document",
    "query_documents",
    "create_message",
    "get_message",
    "update_message",
    "delete_message",
    "get_conversation_messages",
    "delete_conversation",
    "link_message_to_document",
    "get_documents_for_message",
    "get_messages_for_document",
    "create_relationship", # Add to export list
    "delete_relationship_by_key", # Add to export list
]