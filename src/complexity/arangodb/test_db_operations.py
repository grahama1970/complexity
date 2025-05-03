"""
Test script for db_operations.py to verify functionality.
"""

import sys
import uuid
from loguru import logger

# Import setup and specific database functions
from complexity.arangodb.arango_setup import connect_arango, ensure_database
from complexity.arangodb.db_operations import (
    # Generic operations
    create_document,
    get_document,
    update_document,
    delete_document,
    
    # Message operations
    create_message,
    get_message,
    update_message,
    delete_message,
    
    # Relationship operations
    link_message_to_document,
    get_documents_for_message,
    get_messages_for_document,
    
    # Constants
    MESSAGE_COLLECTION_NAME,
    MESSAGE_TYPE_USER,
    COLLECTION_NAME
)


def run_tests():
    """Run a series of tests on db_operations functionality."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr,
               format="{time:HH:mm:ss} | {level:<5} | {message}",
               level="INFO",
               colorize=True)
    
    print("\n===== Testing DB Operations =====")
    
    # --- Test Setup ---
    try:
        client = connect_arango()
        db = ensure_database(client)
        print("✅ Connected to ArangoDB")
    except Exception as e:
        logger.exception(f"❌ Failed to connect to database: {e}")
        sys.exit(1)
    
    test_results = {
        "passed": 0,
        "failed": 0,
        "total": 0
    }
    
    # Generate unique IDs for test
    test_key = None
    test_doc_key = None
    conversation_id = f"test_conv_{uuid.uuid4()}"
    test_doc_content = f"Test document {uuid.uuid4()}"
    
    # --- Test Generic CRUD ---
    
    # Create Document
    print("\n----- Testing Generic CRUD -----")
    try:
        test_doc = create_document(
            db,
            COLLECTION_NAME,
            {
                "question": test_doc_content,
                "label": 1,
                "validated": True,
                "tags": ["test", "crud"],
                "embedding": [0.1] * 1024  # 1024-dimension embedding vector
            }
        )
        
        if test_doc and "_key" in test_doc:
            test_doc_key = test_doc["_key"]
            print(f"✅ Created test document: {test_doc_key}")
            test_results["passed"] += 1
        else:
            print("❌ Failed to create test document")
            test_results["failed"] += 1
    except Exception as e:
        print(f"❌ Error creating document: {e}")
        test_results["failed"] += 1
    
    test_results["total"] += 1
    
    # Read Document
    if test_doc_key:
        try:
            doc = get_document(db, COLLECTION_NAME, test_doc_key)
            if doc and doc["question"] == test_doc_content:
                print(f"✅ Retrieved test document: {test_doc_key}")
                test_results["passed"] += 1
            else:
                print(f"❌ Failed to retrieve test document or content mismatch")
                test_results["failed"] += 1
        except Exception as e:
            print(f"❌ Error retrieving document: {e}")
            test_results["failed"] += 1
        
        test_results["total"] += 1
    
    # Update Document
    if test_doc_key:
        try:
            updated_content = f"Updated question {uuid.uuid4()}"
            update_result = update_document(
                db,
                COLLECTION_NAME,
                test_doc_key,
                {"question": updated_content}
            )
            
            # Verify update
            doc = get_document(db, COLLECTION_NAME, test_doc_key)
            if doc and doc["question"] == updated_content:
                print(f"✅ Updated test document: {test_doc_key}")
                test_results["passed"] += 1
            else:
                print(f"❌ Failed to update test document or content mismatch")
                test_results["failed"] += 1
        except Exception as e:
            print(f"❌ Error updating document: {e}")
            test_results["failed"] += 1
        
        test_results["total"] += 1
    
    # --- Test Message Operations ---
    
    print("\n----- Testing Message Operations -----")
    try:
        test_message = create_message(
            db,
            conversation_id=conversation_id,
            message_type=MESSAGE_TYPE_USER,
            content="Test message content"
        )
        
        if test_message and "_key" in test_message:
            test_key = test_message["_key"]
            print(f"✅ Created test message: {test_key}")
            test_results["passed"] += 1
        else:
            print("❌ Failed to create test message")
            test_results["failed"] += 1
    except Exception as e:
        print(f"❌ Error creating message: {e}")
        test_results["failed"] += 1
    
    test_results["total"] += 1
    
    # Get Message
    if test_key:
        try:
            msg = get_message(db, test_key)
            if msg and msg["content"] == "Test message content":
                print(f"✅ Retrieved test message: {test_key}")
                test_results["passed"] += 1
            else:
                print(f"❌ Failed to retrieve test message or content mismatch")
                test_results["failed"] += 1
        except Exception as e:
            print(f"❌ Error retrieving message: {e}")
            test_results["failed"] += 1
        
        test_results["total"] += 1
    
    # Update Message
    if test_key:
        try:
            updated_msg_content = "Updated message content"
            update_result = update_message(
                db,
                test_key,
                {"content": updated_msg_content}
            )
            
            # Verify update
            msg = get_message(db, test_key)
            if msg and msg["content"] == updated_msg_content:
                print(f"✅ Updated test message: {test_key}")
                test_results["passed"] += 1
            else:
                print(f"❌ Failed to update test message or content mismatch")
                test_results["failed"] += 1
        except Exception as e:
            print(f"❌ Error updating message: {e}")
            test_results["failed"] += 1
        
        test_results["total"] += 1
    
    # --- Test Relationship Operations ---
    
    print("\n----- Testing Relationship Operations -----")
    if test_key and test_doc_key:
        try:
            link_result = link_message_to_document(db, test_key, test_doc_key)
            if link_result:
                print(f"✅ Linked message {test_key} to document {test_doc_key}")
                test_results["passed"] += 1
            else:
                print(f"❌ Failed to link message to document")
                test_results["failed"] += 1
        except Exception as e:
            print(f"❌ Error creating relationship: {e}")
            test_results["failed"] += 1
        
        test_results["total"] += 1
    
    # Get Documents for Message
    if test_key and test_doc_key:
        try:
            related_docs = get_documents_for_message(db, test_key)
            if any(doc["_key"] == test_doc_key for doc in related_docs):
                print(f"✅ Retrieved related document for message")
                test_results["passed"] += 1
            else:
                print(f"❌ Failed to retrieve related document")
                test_results["failed"] += 1
        except Exception as e:
            print(f"❌ Error retrieving related documents: {e}")
            test_results["failed"] += 1
        
        test_results["total"] += 1
    
    # Get Messages for Document
    if test_key and test_doc_key:
        try:
            related_msgs = get_messages_for_document(db, test_doc_key)
            if any(msg["_key"] == test_key for msg in related_msgs):
                print(f"✅ Retrieved related message for document")
                test_results["passed"] += 1
            else:
                print(f"❌ Failed to retrieve related message")
                test_results["failed"] += 1
        except Exception as e:
            print(f"❌ Error retrieving related messages: {e}")
            test_results["failed"] += 1
        
        test_results["total"] += 1
    
    # --- Cleanup ---
    
    print("\n----- Cleanup -----")
    # Delete Message
    if test_key:
        try:
            deleted = delete_message(db, test_key, delete_relationships=True)
            if deleted:
                print(f"✅ Deleted test message and relationships")
                test_results["passed"] += 1
            else:
                print(f"❌ Failed to delete test message")
                test_results["failed"] += 1
        except Exception as e:
            print(f"❌ Error deleting message: {e}")
            test_results["failed"] += 1
        
        test_results["total"] += 1
    
    # Delete Document
    if test_doc_key:
        try:
            deleted = delete_document(db, COLLECTION_NAME, test_doc_key)
            if deleted:
                print(f"✅ Deleted test document")
                test_results["passed"] += 1
            else:
                print(f"❌ Failed to delete test document")
                test_results["failed"] += 1
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            test_results["failed"] += 1
        
        test_results["total"] += 1
    
    # --- Test Results Summary ---
    
    print("\n===== Test Results =====")
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    
    if test_results["failed"] == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {test_results['failed']} tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())