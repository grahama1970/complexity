# src/pdf_extractor/arangodb/examples/simple_relationship_demo.py
import sys
import uuid
from loguru import logger
from arango.database import StandardDatabase
from complexity.arangodb.arango_setup_unknown import connect_arango, ensure_database, ensure_edge_collection, ensure_graph
from complexity.arangodb.config import COLLECTION_NAME, RELATIONSHIP_TYPE_PREREQUISITE
from complexity.arangodb.relationship_api import add_relationship, get_relationships
from complexity.arangodb.agent_decision import create_strategic_relationship

# Set up logging
logger.remove()
logger.add(sys.stderr, level="INFO")

def create_test_documents(db: StandardDatabase):
    """Create test documents with different types of content."""
    # Create a unique ID for this test run
    test_id = uuid.uuid4().hex[:8]
    
    # Create documents
    docs = [
        {
            "_key": f"python_basics_{test_id}",
            "content": "Python basics covering variables, data types, and control flow.",
            "tags": ["python", "programming", "basics"],
            "problem": "Understanding Python fundamentals"
        },
        {
            "_key": f"python_advanced_{test_id}",
            "content": "Advanced Python concepts including decorators, generators, and context managers.",
            "tags": ["python", "programming", "advanced"],
            "problem": "Mastering Python advanced features"
        }
    ]
    
    # Insert the documents
    vertex_collection = db.collection(COLLECTION_NAME)
    inserted_keys = []
    
    for doc in docs:
        vertex_collection.insert(doc)
        inserted_keys.append(doc["_key"])
        print(f"Created document: {doc['_key']}")
        print(f"  Content: {doc['content']}")
        print(f"  Tags: {doc['tags']}")
        print()
    
    return inserted_keys

def basic_document_search(db: StandardDatabase, test_id: str):
    """Find documents from our test set."""
    vertex_collection = db.collection(COLLECTION_NAME)
    
    # Simple AQL query to search for our test documents
    aql = f"""
    FOR doc IN {COLLECTION_NAME}
    FILTER CONTAINS(doc._key, @test_id)
    RETURN doc
    """
    
    cursor = db.aql.execute(aql, bind_vars={"test_id": test_id})
    results = list(cursor)
    
    print(f"Found {len(results)} documents for test ID: {test_id}")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc['_key']}")
        print(f"   Content: {doc['content']}")
        print(f"   Tags: {doc.get('tags', [])}")
    
    return results

def create_demo_relationship(db: StandardDatabase, docs):
    """Create a relationship between documents."""
    # Extract documents with "basics" and "advanced" in keys
    basic_doc = next((doc for doc in docs if "basics" in doc["_key"]), None)
    advanced_doc = next((doc for doc in docs if "advanced" in doc["_key"]), None)
    
    if not basic_doc or not advanced_doc:
        print("Could not find a basic and advanced document pair")
        return None
    
    print("\nRELATIONSHIP ASSESSMENT:")
    print("-------------------------")
    print(f"Found potential prerequisite relationship between:")
    print(f"- Basic document: {basic_doc['_key']}")
    print(f"- Advanced document: {advanced_doc['_key']}")
    
    # Create relationship with rationale
    rationale = ("Understanding Python basics is a prerequisite to learning advanced concepts. "
                "The basic document establishes the foundation needed to comprehend "
                "the advanced features covered in the second document.")
    
    print("\nCREATING RELATIONSHIP:")
    print("---------------------")
    print(f"From: {basic_doc['_key']} (Basic)")
    print(f"To: {advanced_doc['_key']} (Advanced)")
    print(f"Type: {RELATIONSHIP_TYPE_PREREQUISITE}")
    print(f"Rationale: {rationale}")
    print(f"Confidence Score: 1 (highest confidence)")
    
    # Use our existing relationship API
    relationship = add_relationship(
        db=db, 
        from_doc_key=basic_doc["_key"],
        to_doc_key=advanced_doc["_key"],
        rationale=rationale,
        relationship_type=RELATIONSHIP_TYPE_PREREQUISITE,
        confidence_score=1
    )
    
    if relationship:
        print("\nRelationship created successfully!")
        return relationship
    else:
        print("\nFailed to create relationship")
        return None

def verify_relationship(db: StandardDatabase, doc_key: str):
    """Verify the relationship exists by retrieving it."""
    print("\nVERIFYING RELATIONSHIP:")
    print("----------------------")
    print(f"Checking relationships for document: {doc_key}")
    
    relationships = get_relationships(db, doc_key, direction="OUTBOUND")
    
    if relationships:
        print(f"Found {len(relationships)} outbound relationships:")
        for i, rel in enumerate(relationships, 1):
            print(f"{i}. Type: {rel['type']}")
            print(f"   From: {rel['_from'].split('/')[1]}")
            print(f"   To: {rel['_to'].split('/')[1]}")
            print(f"   Confidence Score: {rel['confidence_score']}")
            print(f"   Rationale: {rel['rationale'][:100]}...")
        return True
    else:
        print("No relationships found")
        return False

def clean_up_resources(db: StandardDatabase, doc_keys: list):
    """Clean up created test documents."""
    print("\nCLEANING UP RESOURCES:")
    print("--------------------")
    
    vertex_collection = db.collection(COLLECTION_NAME)
    
    for key in doc_keys:
        # Find and delete all relationships
        rels = get_relationships(db, key)
        for rel in rels:
            try:
                db.collection("document_relationships").delete(rel["_key"])
                print(f"Deleted relationship: {rel['_key']}")
            except Exception as e:
                print(f"Error deleting relationship {rel['_key']}: {e}")
        
        # Delete the document
        try:
            vertex_collection.delete(key)
            print(f"Deleted document: {key}")
        except Exception as e:
            print(f"Error deleting document {key}: {e}")

def run_demo():
    """Run the complete demonstration."""
    print("=" * 80)
    print("DEMONSTRATION: DOCUMENT RELATIONSHIPS IN PDF EXTRACTOR")
    print("=" * 80)
    
    # Step 1: Connect to the database
    print("\n1. CONNECTING TO DATABASE:")
    print("------------------------")
    client = connect_arango()
    db = ensure_database(client)
    ensure_edge_collection(db)
    ensure_graph(db)
    print("Successfully connected to ArangoDB")
    
    # Step 2: Create test documents
    print("\n2. CREATING TEST DOCUMENTS:")
    print("-------------------------")
    doc_keys = create_test_documents(db)
    test_id = doc_keys[0].split("_")[-1]  # Extract test ID from first document key
    
    # Step 3: Search for documents
    print("\n3. SEARCHING FOR DOCUMENTS:")
    print("-------------------------")
    docs = basic_document_search(db, test_id)
    
    # Step 4: Create a relationship
    print("\n4. CREATING RELATIONSHIP:")
    print("-----------------------")
    relationship = create_demo_relationship(db, docs)
    
    # Step 5: Verify the relationship
    if relationship:
        print("\n5. VERIFYING RELATIONSHIP:")
        print("-------------------------")
        basic_doc = next((doc for doc in docs if "basics" in doc["_key"]), None)
        if basic_doc:
            verify_relationship(db, basic_doc["_key"])
    
    # Step 6: Clean up resources
    print("\n6. CLEANING UP RESOURCES:")
    print("------------------------")
    clean_up_resources(db, doc_keys)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return True

if __name__ == "__main__":
    try:
        success = run_demo()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)
