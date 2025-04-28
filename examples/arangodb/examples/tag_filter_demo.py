#!/usr/bin/env python3
# src/pdf_extractor/arangodb/examples/tag_filter_demo.py
"""
Demonstration of the --tag parameter in hybrid search for filtering results.
This script shows how the agent can use tag filtering to narrow down search results
when the default search returns too many irrelevant documents.
"""
import sys
import uuid
import json
import time
from typing import Dict, Any, List, Optional

from loguru import logger
from arango.database import StandardDatabase

# Import necessary components from our implementation
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.config import COLLECTION_NAME
from pdf_extractor.arangodb.relationship_api import add_relationship, get_relationships

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

def create_test_documents(db: StandardDatabase):
    """Create diverse test documents with different tags for demonstration."""
    # Use a unique ID for this demo run to easily identify and clean up
    test_id = uuid.uuid4().hex[:8]
    
    # Define sample documents with various tags
    test_docs = [
        {
            "_key": f"python_basics_{test_id}",
            "content": "Introduction to Python programming language including variables, data types, and control structures.",
            "tags": ["python", "programming", "basics", "tutorial"],
            "problem": "Learning Python fundamentals for beginners"
        },
        {
            "_key": f"python_advanced_{test_id}",
            "content": "Advanced Python concepts including decorators, generators, metaclasses, and context managers.",
            "tags": ["python", "programming", "advanced", "optimization"],
            "problem": "Improving Python code efficiency and structure"
        },
        {
            "_key": f"java_basics_{test_id}",
            "content": "Introduction to Java programming with focus on object-oriented principles and syntax.",
            "tags": ["java", "programming", "basics", "tutorial"],
            "problem": "Understanding Java foundations"
        },
        {
            "_key": f"database_sql_{test_id}",
            "content": "SQL fundamentals for relational database management including queries, joins, and schema design.",
            "tags": ["database", "sql", "basics", "data"],
            "problem": "Learning SQL for data manipulation"
        },
        {
            "_key": f"database_nosql_{test_id}",
            "content": "NoSQL database concepts and practical applications for modern application development.",
            "tags": ["database", "nosql", "advanced", "data"],
            "problem": "Understanding NoSQL database paradigms"
        },
        {
            "_key": f"web_frontend_{test_id}",
            "content": "Web frontend development with HTML, CSS, and JavaScript for building responsive interfaces.",
            "tags": ["web", "frontend", "javascript", "tutorial"],
            "problem": "Creating modern web interfaces"
        }
    ]
    
    # Insert the documents into the collection
    collection = db.collection(COLLECTION_NAME)
    doc_keys = []
    
    for doc in test_docs:
        try:
            collection.insert(doc)
            doc_keys.append(doc["_key"])
            logger.info(f"Created document: {doc['_key']}")
            print(f"Created: {doc['_key']}")
            print(f"  Tags: {doc['tags']}")
            print(f"  Problem: {doc['problem']}")
            print()
        except Exception as e:
            logger.error(f"Failed to insert document {doc['_key']}: {e}")
    
    return doc_keys

def mock_hybrid_search(db: StandardDatabase, query_text: str, tag_filters: Optional[List[str]] = None, top_n: int = 5):
    """
    Simple mock implementation of hybrid search with tag filtering.
    
    Args:
        db: Database connection
        query_text: Search query text
        tag_filters: Optional list of tags to filter by
        top_n: Maximum number of results to return
        
    Returns:
        Search results with scores
    """
    collection = db.collection(COLLECTION_NAME)
    
    # Construct the AQL query with optional tag filtering
    tag_filter_clause = ""
    bind_vars = {"query": query_text.lower(), "limit": top_n}
    
    if tag_filters and len(tag_filters) > 0:
        tag_conditions = []
        for i, tag in enumerate(tag_filters):
            tag_param = f"tag{i}"
            tag_conditions.append(f"POSITION(doc.tags, @{tag_param}) != false")
            bind_vars[tag_param] = tag
        
        if tag_conditions:
            tag_filter_clause = f" FILTER {' AND '.join(tag_conditions)}"
    
    # Simple keyword-based query with tag filtering
    aql = f"""
    FOR doc IN {COLLECTION_NAME}
    LET score = (
        CONTAINS(LOWER(doc.content), @query) ? 0.8 : 0.0) +
        (CONTAINS(LOWER(doc.problem), @query) ? 0.9 : 0.0
    )
    FILTER score > 0 {tag_filter_clause}
    SORT score DESC
    LIMIT @limit
    RETURN {{
        "doc": doc,
        "score": score
    }}
    """
    
    # Execute the query
    cursor = db.aql.execute(aql, bind_vars=bind_vars)
    results = list(cursor)
    
    return {
        "results": results,
        "count": len(results),
        "query": query_text,
        "tag_filters": tag_filters
    }

def demo_tag_filtering(db: StandardDatabase, doc_keys: List[str]):
    """
    Demonstrate how tag filtering improves search results.
    
    Args:
        db: Database connection
        doc_keys: List of document keys to clean up after
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: HYBRID SEARCH WITH TAG FILTERING")
    print("="*80)
    
    # Step 1: Perform a generic search for "programming"
    query = "programming"
    print(f"\n1. GENERIC SEARCH QUERY: '{query}'")
    print("-" * 50)
    
    results = mock_hybrid_search(db, query)
    print(f"Found {results['count']} results:")
    for i, result in enumerate(results.get("results", []), 1):
        doc = result.get("doc", {})
        score = result.get("score", 0)
        print(f"{i}. {doc.get('_key')} (Score: {score:.2f})")
        print(f"   Tags: {doc.get('tags', [])}")
        print(f"   Problem: {doc.get('problem', '')}")
    
    # Step 2: Now filter with a specific tag to narrow the results
    tag_filter = ["python"]
    print(f"\n2. FILTERED SEARCH WITH TAG: '{query}' + tags={tag_filter}")
    print("-" * 50)
    
    filtered_results = mock_hybrid_search(db, query, tag_filters=tag_filter)
    print(f"Found {filtered_results['count']} results:")
    for i, result in enumerate(filtered_results.get("results", []), 1):
        doc = result.get("doc", {})
        score = result.get("score", 0)
        print(f"{i}. {doc.get('_key')} (Score: {score:.2f})")
        print(f"   Tags: {doc.get('tags', [])}")
        print(f"   Problem: {doc.get('problem', '')}")
    
    # Step 3: Filter with multiple tags to further narrow results
    tag_filters = ["python", "advanced"]
    print(f"\n3. FILTERED SEARCH WITH MULTIPLE TAGS: '{query}' + tags={tag_filters}")
    print("-" * 50)
    
    multi_filtered_results = mock_hybrid_search(db, query, tag_filters=tag_filters)
    print(f"Found {multi_filtered_results['count']} results:")
    for i, result in enumerate(multi_filtered_results.get("results", []), 1):
        doc = result.get("doc", {})
        score = result.get("score", 0)
        print(f"{i}. {doc.get('_key')} (Score: {score:.2f})")
        print(f"   Tags: {doc.get('tags', [])}")
        print(f"   Problem: {doc.get('problem', '')}")
    
    # Step 4: Show the agent decision process
    print("\n4. AGENT DECISION PROCESS")
    print("-" * 50)
    print("Query: 'advanced programming techniques'")
    print("Initial search returns multiple results across different programming languages")
    print("Assessment: Need to filter results to focus on specific language")
    print("Action: Add tag filter for 'python' to narrow results")
    print("CLI command: search hybrid 'advanced programming techniques' --tags python")
    print("\nThis allows the agent to:") 
    print("1. Start with broad searches")
    print("2. Add tag filters when results need refinement")
    print("3. Create targeted relationships between the most relevant documents")
    
    print("\n" + "="*80)
    print("TAG FILTERING DEMONSTRATION COMPLETED")
    print("="*80)

def cleanup_test_documents(db: StandardDatabase, doc_keys: List[str]):
    """Clean up test documents after the demo."""
    print("\nCleaning up test documents...")
    collection = db.collection(COLLECTION_NAME)
    
    for key in doc_keys:
        try:
            collection.delete(key)
            print(f"Deleted document: {key}")
        except Exception as e:
            logger.error(f"Failed to delete document {key}: {e}")

def run_demo():
    """Main demo function."""
    print("Connecting to ArangoDB...")
    client = connect_arango()
    db = ensure_database(client)
    
    print("\nCreating test documents with various tags...")
    doc_keys = create_test_documents(db)
    
    # Run the tag filtering demonstration
    demo_tag_filtering(db, doc_keys)
    
    # Clean up test documents
    cleanup_test_documents(db, doc_keys)
    
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
