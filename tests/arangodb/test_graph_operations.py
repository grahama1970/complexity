#!/usr/bin/env python3
"""
Verify graph operations with enhanced_relationships.py - shows raw results with no success claims.
"""
import sys
import uuid
import time

# Import required modules
from complexity.arangodb.arango_setup import connect_arango, ensure_database
from complexity.arangodb.db_operations import create_document, delete_document
from complexity.arangodb.enhanced_relationships import create_edge_from_cli, delete_edge_from_cli

# Get database connection
print("Connecting to database...")
client = connect_arango()
db = ensure_database(client)
print(f"Connected to database: {db.name}")

# Set up test collections
TEST_DOC_COLLECTION = "test_docs"
TEST_EDGE_COLLECTION = "test_relationships"

# Create two test documents
source_id = f"source_{uuid.uuid4().hex[:8]}"
target_id = f"target_{uuid.uuid4().hex[:8]}"

source_doc = {
    "_key": source_id,
    "title": "Source Document for Graph Test",
    "content": "This document is the source node for testing graph operations.",
    "created": time.time()
}

target_doc = {
    "_key": target_id,
    "title": "Target Document for Graph Test",
    "content": "This document is the target node for testing graph operations.",
    "created": time.time()
}

print("\n=== DOCUMENT CREATION FOR GRAPH TEST ===")
source_result = create_document(db, TEST_DOC_COLLECTION, source_doc)
print(f"Source document created: {source_result.get('_key') if source_result else None}")

target_result = create_document(db, TEST_DOC_COLLECTION, target_doc)
print(f"Target document created: {target_result.get('_key') if target_result else None}")

print("\n=== GRAPH EDGE CREATION TEST ===")
try:
    edge_result = create_edge_from_cli(
        db,
        from_key=source_id,
        to_key=target_id,
        collection=TEST_DOC_COLLECTION,
        edge_collection=TEST_EDGE_COLLECTION,
        edge_type="TEST_RELATIONSHIP",
        rationale="Testing graph operations with enhanced relationships"
    )
    
    print(f"Edge creation returned: {type(edge_result).__name__}")
    if edge_result:
        print(f"Edge _key: {edge_result.get('_key')}")
        print(f"Edge _from: {edge_result.get('_from')}")
        print(f"Edge _to: {edge_result.get('_to')}")
        print(f"Edge type: {edge_result.get('type')}")
        print(f"Edge rationale: {edge_result.get('rationale')}")
        
        edge_key = edge_result.get('_key')
    else:
        print("Edge creation returned None")
        edge_key = None
except Exception as e:
    print(f"EDGE CREATION ERROR: {str(e)}")
    edge_key = None

# Verify the edge exists in the database with a direct query
print("\n=== EDGE VERIFICATION WITH DIRECT QUERY ===")
if edge_key:
    try:
        aql = f"FOR edge IN {TEST_EDGE_COLLECTION} FILTER edge._key == @key RETURN edge"
        cursor = db.aql.execute(aql, bind_vars={"key": edge_key})
        results = list(cursor)
        print(f"AQL query returned {len(results)} edges")
        if results:
            edge = results[0]
            print(f"Edge _key direct from DB: {edge.get('_key')}")
            print(f"Edge _from direct from DB: {edge.get('_from')}")
            print(f"Edge _to direct from DB: {edge.get('_to')}")
        else:
            print("AQL query returned no results - edge not found in database")
    except Exception as e:
        print(f"EDGE VERIFICATION ERROR: {str(e)}")

print("\n=== EDGE DELETION TEST ===")
if edge_key:
    try:
        delete_result = delete_edge_from_cli(db, edge_key, TEST_EDGE_COLLECTION)
        print(f"Edge deletion returned: {delete_result}")
        
        # Verify the edge is gone with another direct query
        aql = f"FOR edge IN {TEST_EDGE_COLLECTION} FILTER edge._key == @key RETURN edge"
        cursor = db.aql.execute(aql, bind_vars={"key": edge_key})
        results = list(cursor)
        print(f"After deletion: AQL query returned {len(results)} edges")
    except Exception as e:
        print(f"EDGE DELETION ERROR: {str(e)}")

print("\n=== CLEANUP ===")
# Clean up test documents
try:
    delete_source = delete_document(db, TEST_DOC_COLLECTION, source_id)
    print(f"Source document deletion: {delete_source}")
    
    delete_target = delete_document(db, TEST_DOC_COLLECTION, target_id)
    print(f"Target document deletion: {delete_target}")
except Exception as e:
    print(f"CLEANUP ERROR: {str(e)}")

print("\n=== RAW TEST RESULTS COMPLETE ===")
print("No success/failure interpretation added. Results above are the actual observations.")