#!/usr/bin/env python3
# src/pdf_extractor/arangodb/examples/agent_with_tags.py
"""
Demonstration of an agent workflow using tag filtering
and relationship assessment with actual ArangoDB operations.
"""
import sys
import uuid
from loguru import logger
from arango.database import StandardDatabase

from complexity.arangodb.arango_setup_unknown import connect_arango, ensure_database, ensure_edge_collection, ensure_graph
from complexity.arangodb.config import COLLECTION_NAME, RELATIONSHIP_TYPE_PREREQUISITE
from complexity.arangodb.relationship_api import add_relationship, get_relationships

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

def create_test_documents(db: StandardDatabase):
    """Create test documents with specific tags in ArangoDB."""
    test_id = uuid.uuid4().hex[:8]
    
    docs = [
        {
            "_key": f"postgres_basic_{test_id}",
            "content": "Introduction to PostgreSQL database optimization including indexing fundamentals and query planning.",
            "tags": ["database", "postgresql", "optimization", "basics"],
            "problem": "Basic PostgreSQL performance tuning"
        },
        {
            "_key": f"postgres_advanced_{test_id}",
            "content": "Advanced PostgreSQL optimization techniques including partitioning, parallel queries, and workload analysis.",
            "tags": ["database", "postgresql", "optimization", "advanced"],
            "problem": "Advanced PostgreSQL performance tuning"
        },
        {
            "_key": f"mysql_basic_{test_id}",
            "content": "MySQL performance optimization fundamentals including proper indexing and query optimization.",
            "tags": ["database", "mysql", "optimization", "basics"],
            "problem": "Basic MySQL performance tuning"
        },
        {
            "_key": f"mongodb_basics_{test_id}",
            "content": "Introduction to MongoDB performance tuning including indexing strategies and query optimization.",
            "tags": ["database", "mongodb", "nosql", "optimization", "basics"],
            "problem": "Basic MongoDB performance tuning"
        },
    ]
    
    collection = db.collection(COLLECTION_NAME)
    doc_keys = []
    
    for doc in docs:
        collection.insert(doc)
        doc_keys.append(doc["_key"])
        print(f"Created document: {doc['_key']}")
        print(f"  Tags: {doc['tags']}")
        print(f"  Problem: {doc['problem']}")
        print()
    
    return doc_keys, test_id

def search_all_documents(db: StandardDatabase, test_id: str):
    """Execute a database query to find all documents matching test_id."""
    collection = db.collection(COLLECTION_NAME)
    
    aql = f"""
    FOR doc IN {COLLECTION_NAME}
    FILTER CONTAINS(doc._key, @test_id)
    RETURN {{
        "doc": doc,
        "score": 0.9
    }}
    """
    
    cursor = db.aql.execute(aql, bind_vars={"test_id": test_id})
    results = list(cursor)
    
    return {
        "results": results,
        "count": len(results)
    }

def search_with_tag_filter(db: StandardDatabase, test_id: str, tags: list):
    """Execute a database query with tag filtering to find specific documents."""
    collection = db.collection(COLLECTION_NAME)
    
    tag_conditions = []
    bind_vars = {"test_id": test_id}
    
    for i, tag in enumerate(tags):
        tag_param = f"tag{i}"
        tag_conditions.append(f"POSITION(doc.tags, @{tag_param}) != false")
        bind_vars[tag_param] = tag
    
    tag_filter = " AND ".join(tag_conditions)
    
    aql = f"""
    FOR doc IN {COLLECTION_NAME}
    FILTER CONTAINS(doc._key, @test_id)
    FILTER {tag_filter}
    RETURN {{
        "doc": doc,
        "score": 0.9
    }}
    """
    
    cursor = db.aql.execute(aql, bind_vars=bind_vars)
    results = list(cursor)
    
    return {
        "results": results,
        "count": len(results),
        "tags": tags
    }

def agent_demo(db: StandardDatabase, test_id: str):
    """
    Demonstrate an agent workflow for relationship creation with tag filtering.
    This uses actual ArangoDB operations for all steps.
    
    Args:
        db: Database connection
        test_id: ID to filter documents by
    """
    print("\n" + "="*80)
    print("AGENT WORKFLOW: TAG FILTERING AND RELATIONSHIP ASSESSMENT")
    print("="*80)
    
    # Step 1: Agent receives a query about database optimization
    query = "database optimization techniques"
    print(f"\nSTEP 1: INITIAL QUERY")
    print(f"Query: '{query}'")
    
    # Step 2: Agent performs initial broad search
    print("\nSTEP 2: INITIAL SEARCH (WITHOUT TAG FILTERING)")
    results = search_all_documents(db, test_id)
    print(f"Found {results['count']} results:")
    
    for i, result in enumerate(results.get("results", []), 1):
        doc = result.get("doc", {})
        score = result.get("score", 0)
        print(f"{i}. {doc.get('_key')} (Score: {score:.2f})")
        print(f"   Tags: {doc.get('tags', [])}")
        print(f"   Problem: {doc.get('problem', '')}")
    
    # Step 3: Agent recognizes multiple database types and decides to filter
    print("\nSTEP 3: AGENT ASSESSMENT")
    print("Analysis: Results contain multiple database types (PostgreSQL, MySQL, MongoDB)")
    print("Decision: Filter by specific database type to narrow results")
    print("Action: Add tag filter for 'postgresql'")
    
    # Step 4: Agent performs filtered search
    tag_filters = ["postgresql"]
    print(f"\nSTEP 4: FILTERED SEARCH WITH TAGS {tag_filters}")
    filtered_results = search_with_tag_filter(db, test_id, tag_filters)
    print(f"Found {filtered_results['count']} results:")
    
    for i, result in enumerate(filtered_results.get("results", []), 1):
        doc = result.get("doc", {})
        score = result.get("score", 0)
        print(f"{i}. {doc.get('_key')} (Score: {score:.2f})")
        print(f"   Tags: {doc.get('tags', [])}")
        print(f"   Problem: {doc.get('problem', '')}")
    
    # Step 5: Agent assesses relationship potential between filtered results
    print("\nSTEP 5: RELATIONSHIP ASSESSMENT")
    print("Analysis: Found a potential prerequisite relationship:")
    print("  - Basic document covers fundamental PostgreSQL optimization")
    print("  - Advanced document builds on those fundamentals")
    print("Decision: Create a PREREQUISITE relationship")
    
    # Find the basic and advanced documents
    basic_doc = None
    advanced_doc = None
    
    for result in filtered_results.get("results", []):
        doc = result.get("doc", {})
        if "basic" in doc.get("_key", "").lower():
            basic_doc = doc
        elif "advanced" in doc.get("_key", "").lower():
            advanced_doc = doc
    
    if basic_doc and advanced_doc:
        # Step 6: Agent creates the relationship
        print("\nSTEP 6: RELATIONSHIP CREATION")
        print(f"From: {basic_doc['_key']}")
        print(f"To: {advanced_doc['_key']}")
        print(f"Type: {RELATIONSHIP_TYPE_PREREQUISITE}")
        
        rationale = (
            "Understanding basic PostgreSQL optimization techniques is a prerequisite for "
            "implementing advanced optimization strategies. The basic document covers foundational "
            "concepts like indexing and query planning which are necessary before tackling advanced "
            "topics such as partitioning and workload analysis."
        )
        print(f"Rationale: {rationale[:100]}...")
        
        # Create the actual relationship in the database
        relationship = add_relationship(
            db,
            from_doc_key=basic_doc["_key"],
            to_doc_key=advanced_doc["_key"],
            rationale=rationale,
            relationship_type=RELATIONSHIP_TYPE_PREREQUISITE,
            confidence_score=1
        )
        
        if relationship:
            print("\nRelationship created successfully!")
            print("\nSTEP 7: VERIFICATION")
            
            # Verify the relationship exists in the database
            rels = get_relationships(db, basic_doc["_key"])
            if rels:
                print(f"Verified: Found {len(rels)} relationship(s) for {basic_doc['_key']}")
                for rel in rels:
                    print(f"  Type: {rel.get('type')}")
                    print(f"  From: {rel.get('_from')}")
                    print(f"  To: {rel.get('_to')}")
            else:
                print("No relationships found!")
    else:
        print("Could not find both basic and advanced documents for relationship creation.")
    
    print("\n" + "="*80)
    print("AGENT WORKFLOW DEMONSTRATION COMPLETED")
    print("="*80)

def cleanup_documents(db: StandardDatabase, doc_keys: list):
    """Clean up test documents and relationships from the database."""
    print("\nCleaning up test documents and relationships...")
    
    # Clean up relationships first
    edge_collection = db.collection("document_relationships")
    for key in doc_keys:
        # Get relationships for this document
        rels = get_relationships(db, key)
        for rel in rels:
            try:
                edge_collection.delete(rel["_key"])
                print(f"Deleted relationship: {rel['_key']}")
            except Exception as e:
                logger.error(f"Failed to delete relationship {rel['_key']}: {e}")
    
    # Then clean up documents
    collection = db.collection(COLLECTION_NAME)
    for key in doc_keys:
        try:
            collection.delete(key)
            print(f"Deleted document: {key}")
        except Exception as e:
            logger.error(f"Failed to delete document {key}: {e}")

def run_demo():
    """Main function to run the demonstration with real ArangoDB operations."""
    print("Connecting to ArangoDB...")
    client = connect_arango()
    db = ensure_database(client)
    ensure_edge_collection(db)
    ensure_graph(db)
    
    print("\nCreating test documents in ArangoDB...")
    doc_keys, test_id = create_test_documents(db)
    
    # Run the agent workflow demo with real database operations
    agent_demo(db, test_id)
    
    # Clean up test documents and relationships from the database
    cleanup_documents(db, doc_keys)
    
    print("\nDemonstration completed successfully.")
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
