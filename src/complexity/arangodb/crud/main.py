# src/pdf_extractor/arangodb/crud/main.py
"""Main execution block for testing CRUD operations."""

import sys
import uuid
from loguru import logger

# Import setup and specific CRUD functions needed for testing
# Use absolute imports based on the package structure
from complexity.arangodb.arango_setup_unknown import connect_arango, ensure_database
from complexity.arangodb.crud.generic import get_document, delete_document # Need generic delete_document for cleanup
from complexity.arangodb.crud.message_history import (
    create_message,
    get_message, # Add missing import
    update_message,
    delete_message, # Need specific message delete for relationship handling
    MESSAGE_COLLECTION_NAME,
    MESSAGE_TYPE_USER
)
from complexity.arangodb.crud.relationships import (
    link_message_to_document,
    get_documents_for_message,
    get_messages_for_document
)
# Import COLLECTION_NAME and alias it
from complexity.arangodb.config import COLLECTION_NAME as DOC_COLLECTION_NAME


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr,
               format="{time:HH:mm:ss} | {level:<5} | {message}",
               level="INFO",
               colorize=True)

    # --- Test Setup ---
    try:
        client = connect_arango()
        db = ensure_database(client)
        # Ensure necessary collections exist (setup script should handle this, but good practice)
        # ensure_collection(db, MESSAGE_COLLECTION_NAME)
        # ensure_collection(db, DOC_COLLECTION_NAME)
        # ensure_edge_collection(db, DOC_EDGE_COLLECTION_NAME) # Assuming this is the main edge collection
        # ensure_edge_collection(db, MESSAGE_EDGE_COLLECTION_NAME)
    except Exception as e:
        logger.exception(f"❌ Failed to connect or ensure database/collections: {e}")
        sys.exit(1)

    print("\nTesting CRUD operations...")
    test_key = None
    test_doc_key = "doc1_f5f1489c" # Use an existing doc key from setup
    conversation_id = f"test_conv_{uuid.uuid4()}"
    passed = True

    try:
        # 1. Create
        test_message = create_message(
            db,
            conversation_id=conversation_id,
            message_type=MESSAGE_TYPE_USER,
            content="Test message content"
        )
        if test_message and "_key" in test_message:
            test_key = test_message["_key"]
            print(f"✅ Created test message: {test_key}")
        else:
            print("❌ Failed to create test message")
            passed = False
            sys.exit(1) # Exit if create fails, subsequent steps depend on it

        # 2. Read
        retrieved = get_message(db, test_key)
        if retrieved and retrieved["_key"] == test_key:
            print(f"✅ Retrieved test message: {test_key}")
        else:
            print(f"❌ Failed to retrieve test message: {test_key}")
            passed = False

        # 3. Update
        update_result_meta = update_message(db, test_key, {"content": "Updated test message"})
        if update_result_meta: # Check if update call itself succeeded (returned metadata)
            # Verify update persisted by re-reading the document
            retrieved_after_update = get_message(db, test_key)
            if retrieved_after_update and retrieved_after_update.get("content") == "Updated test message":
                 print(f"✅ Updated test message: {test_key}")
            else:
                 print(f"❌ Update verification failed for message: {test_key}. Content: {retrieved_after_update.get('content') if retrieved_after_update else 'Not Found'}")
                 passed = False
        else:
            print(f"❌ Failed to execute update operation for message: {test_key}")
            passed = False

        # 4. Link message to document
        link_result = link_message_to_document(db, test_key, test_doc_key)
        if link_result:
            print(f"✅ Linked message {test_key} to document {test_doc_key}")
        else:
            print(f"❌ Failed to link message {test_key} to document {test_doc_key}")
            passed = False

        # 5. Get documents for message
        related_docs = get_documents_for_message(db, test_key)
        if any(doc["_key"] == test_doc_key for doc in related_docs):
             print(f"✅ Retrieved related document {test_doc_key} for message {test_key}")
        else:
             print(f"❌ Failed to retrieve related document for message {test_key}")
             passed = False

        # 6. Get messages for document
        related_msgs = get_messages_for_document(db, test_doc_key)
        if any(msg["_key"] == test_key for msg in related_msgs):
             print(f"✅ Retrieved related message {test_key} for document {test_doc_key}")
        else:
             print(f"❌ Failed to retrieve related message for document {test_doc_key}")
             passed = False

    except Exception as e:
        logger.exception(f"❌ ERROR during CRUD test execution: {e}")
        passed = False

    finally:
        # 7. Delete (Cleanup) - Use specific delete_message first
        if test_key:
            deleted = delete_message(db, test_key, delete_relationships=True) # Ensure relationships are deleted too
            if deleted:
                print(f"✅ Deleted test message: {test_key}")
                # Verify deletion
                if get_message(db, test_key) is not None:
                    print(f"❌ Verification failed: Message {test_key} still exists after deletion.")
                    passed = False
            else:
                print(f"❌ Failed to delete test message: {test_key}")
                passed = False

    # Final result
    if passed:
        print("\n✅ CRUD operations validation passed")
        sys.exit(0)
    else:
        print("\n❌ CRUD operations validation failed")
        sys.exit(1)