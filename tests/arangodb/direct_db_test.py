#!/usr/bin/env python3
"""
Direct database test script that interacts with ArangoDB directly 
to prove actual functionality.
"""
import json
import time
import uuid
from arango import ArangoClient
from complexity.arangodb.config import ARANGO_HOST, ARANGO_USER, ARANGO_PASSWORD, ARANGO_DB_NAME

def main():
    """Main test function"""
    print("\n=== DIRECT ARANGO DB TEST ===")
    print(f"Starting test at: {time.ctime()}")
    
    # Generate unique test IDs
    test_id = str(uuid.uuid4())[:8]
    doc1_key = f"test_doc_{test_id}_1"
    doc2_key = f"test_doc_{test_id}_2"
    
    # Connect to ArangoDB
    print("\n1. Connecting to ArangoDB...")
    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db(ARANGO_DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)
    print(f"Connected to database: {db.name} (ArangoDB v{db.version})")
    
    # Ensure collections exist
    print("\n2. Verifying collections...")
    for collection_name in ['test_docs', 'test_relationships']:
        if not db.has_collection(collection_name):
            is_edge = 'relationship' in collection_name
            db.create_collection(collection_name, edge=is_edge)
            print(f"Created {'edge ' if is_edge else ''}collection: {collection_name}")
        else:
            print(f"Found existing collection: {collection_name}")
    
    # Create test document 1
    print(f"\n3. Creating document with key: {doc1_key}")
    doc1 = {
        "_key": doc1_key,
        "content": "This is a test document for verification",
        "tags": ["test", "verification"],
        "test_timestamp": time.time()
    }
    
    result = db.collection("test_docs").insert(doc1)
    print(f"Insert result: {result}")
    
    # Verify document exists
    print(f"\n4. Verifying document exists...")
    retrieved_doc = db.collection("test_docs").get(doc1_key)
    print(f"Retrieved document: {json.dumps(retrieved_doc, indent=2)}")
    
    # Update the document
    print(f"\n5. Updating document...")
    
    # Create a document with just the fields to update
    update_data = {
        "content": "This document has been updated",
        "updated_at": time.time()
    }
    
    # Use AQL for updating the document
    query = """
    UPDATE @key WITH @data IN test_docs
    RETURN NEW
    """
    
    cursor = db.aql.execute(
        query,
        bind_vars={
            'key': doc1_key,
            'data': update_data
        }
    )
    
    update_result = list(cursor)[0]
    print(f"Update result: {json.dumps(update_result, indent=2)}")
    
    # Verify the update
    print(f"\n6. Verifying document update...")
    updated_doc = db.collection("test_docs").get(doc1_key)
    print(f"Updated document: {json.dumps(updated_doc, indent=2)}")
    
    # Create second document
    print(f"\n7. Creating second document with key: {doc2_key}")
    doc2 = {
        "_key": doc2_key,
        "content": "This is another test document for relationship testing",
        "tags": ["test", "relationship"],
        "test_timestamp": time.time()
    }
    
    result2 = db.collection("test_docs").insert(doc2)
    print(f"Insert result: {result2}")
    
    # Create relationship
    print(f"\n8. Creating relationship between documents...")
    edge = {
        "_from": f"test_docs/{doc1_key}",
        "_to": f"test_docs/{doc2_key}",
        "type": "TEST_REL",
        "rationale": "Testing relationship",
        "test_id": test_id
    }
    
    edge_result = db.collection("test_relationships").insert(edge)
    print(f"Edge creation result: {edge_result}")
    
    # Verify relationship with AQL
    print(f"\n9. Verifying relationship with AQL...")
    query = '''
    FOR edge IN test_relationships
        FILTER edge._from == @from_id AND edge._to == @to_id
        RETURN edge
    '''
    
    cursor = db.aql.execute(
        query,
        bind_vars={
            'from_id': f"test_docs/{doc1_key}",
            'to_id': f"test_docs/{doc2_key}"
        }
    )
    
    results = list(cursor)
    print(f"AQL query found {len(results)} relationships")
    if results:
        print(f"Relationship data: {json.dumps(results[0], indent=2)}")
    
    # Add documents for search
    print(f"\n10. Creating documents for search testing...")
    search_docs = []
    for i in range(3):
        search_key = f"search_doc_{test_id}_{i}"
        search_doc = {
            "_key": search_key,
            "content": f"Document about {'machine learning' if i==0 else 'artificial intelligence' if i==1 else 'natural language processing'}",
            "tags": ["search", "test", f"topic_{i}"],
            "test_timestamp": time.time()
        }
        db.collection("test_docs").insert(search_doc)
        search_docs.append(search_key)
    
    print(f"Created {len(search_docs)} search documents")
    
    # Perform search
    print(f"\n11. Performing search for 'machine learning'...")
    search_query = '''
    FOR doc IN test_docs
        FILTER doc.content LIKE "%machine learning%" 
        SORT doc.timestamp DESC
        LIMIT 5
        RETURN doc
    '''
    
    search_cursor = db.aql.execute(search_query)
    search_results = list(search_cursor)
    
    print(f"Search found {len(search_results)} documents")
    for i, result in enumerate(search_results):
        print(f"\nResult {i+1}:")
        print(f"Key: {result.get('_key', 'N/A')}")
        print(f"Content: {result.get('content', 'N/A')}")
        print(f"Tags: {', '.join(result.get('tags', []))}")
    
    # Cleanup - delete test documents
    print(f"\n12. Cleaning up test documents...")
    db.collection("test_docs").delete(doc1_key)
    db.collection("test_docs").delete(doc2_key)
    for key in search_docs:
        db.collection("test_docs").delete(key)
    
    print(f"\n=== TEST COMPLETED SUCCESSFULLY ===")
    print(f"Test ID: {test_id}")
    print(f"Test finished at: {time.ctime()}")

if __name__ == "__main__":
    main()