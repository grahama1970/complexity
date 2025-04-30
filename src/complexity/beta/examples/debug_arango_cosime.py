#!/usr/bin/env python
"""
Debug script for ArangoDB vector similarity search issues.
This script tests different approaches to vector similarity search in ArangoDB.
"""

import sys
import os
from arango import ArangoClient
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Load configuration (simplified for this example)
CONFIG = {
    "arango": {
        "host": os.getenv("ARANGO_HOST", "http://localhost:8529"),
        "user": os.getenv("ARANGO_USER", "root"),
        "password": os.getenv("ARANGO_PASSWORD", "openSesame"),
        "db_name": os.getenv("ARANGO_DB_NAME", "memory_bank")
    },
    "search": {
        "collection_name": os.getenv("COLLECTION_NAME", "complexity")
    },
    "embedding": {
        "field": os.getenv("EMBEDDING_FIELD", "embedding"),
        "dimensions": int(os.getenv("EMBEDDING_DIM", "1024"))
    }
}

def connect_arango() -> Tuple[ArangoClient, Any]:
    """Connect to ArangoDB and return client and database objects."""
    try:
        client = ArangoClient(hosts=CONFIG["arango"]["host"])
        db = client.db(
            CONFIG["arango"]["db_name"],
            username=CONFIG["arango"]["user"],
            password=CONFIG["arango"]["password"]
        )
        print(f"Connected to ArangoDB version {db.version()}")
        return client, db
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

def test_basic_cosine_similarity(db, collection_name, embedding_field):
    """Test a basic cosine similarity query between two documents."""
    print("\n=== Testing Basic Cosine Similarity ===")
    try:
        # Get two documents
        aql = f"""
        FOR doc IN {collection_name}
            LIMIT 2
            RETURN {{ _id: doc._id, question: doc.question, embedding: doc.{embedding_field} }}
        """
        cursor = db.aql.execute(aql)
        docs = list(cursor)
        
        if len(docs) < 2:
            print("Not enough documents for test")
            return False
            
        print(f"Got 2 documents for testing:")
        print(f"  Doc1: {docs[0]['_id']} - {docs[0]['question'][:50]}...")
        print(f"  Doc2: {docs[1]['_id']} - {docs[1]['question'][:50]}...")
        
        # Test cosine similarity between them
        aql = f"""
        RETURN COSINE_SIMILARITY(@vec1, @vec2)
        """
        cursor = db.aql.execute(aql, bind_vars={
            "vec1": docs[0]["embedding"], 
            "vec2": docs[1]["embedding"]
        })
        similarity = cursor.next()
        print(f"Cosine similarity between two docs: {similarity}")
        
        # Now test it in a more structured query
        aql = f"""
        LET doc1 = DOCUMENT("{docs[0]['_id']}")
        LET doc2 = DOCUMENT("{docs[1]['_id']}")
        RETURN {{
            doc1_id: doc1._id,
            doc2_id: doc2._id,
            similarity: COSINE_SIMILARITY(doc1.{embedding_field}, doc2.{embedding_field})
        }}
        """
        cursor = db.aql.execute(aql)
        result = cursor.next()
        print(f"Structured query result: {result}")
        
        return True
    except Exception as e:
        print(f"Error in test: {str(e)}")
        return False

def test_batched_approach(db, collection_name, embedding_field, threshold=0.8, batch_size=5):
    """Test finding pairs of documents with similarity above threshold using a batched approach."""
    print(f"\n=== Testing Batched Approach (threshold={threshold}, batch_size={batch_size}) ===")
    try:
        # Get a small sample of document IDs
        aql = f"""
        FOR doc IN {collection_name}
            LIMIT 20
            RETURN doc._id
        """
        cursor = db.aql.execute(aql)
        all_ids = list(cursor)
        total_docs = len(all_ids)
        print(f"Found {total_docs} documents to process")
        
        # Process in batches
        pairs = []
        processed = 0
        
        for i in range(0, total_docs, batch_size):
            batch_ids = all_ids[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_docs+batch_size-1)//batch_size}")
            
            aql = f"""
            FOR doc1 IN {collection_name}
                FILTER doc1._id IN @batch_ids
                FOR doc2 IN {collection_name}
                    FILTER doc1._id < doc2._id  // Avoid duplicates and self-comparisons
                    LET similarity = COSINE_SIMILARITY(doc1.{embedding_field}, doc2.{embedding_field})
                    FILTER similarity >= @threshold
                    RETURN {{
                        from: doc1._id,
                        to: doc2._id,
                        similarity: similarity,
                        from_question: doc1.question,
                        to_question: doc2.question
                    }}
            """
            
            cursor = db.aql.execute(aql, bind_vars={
                "batch_ids": batch_ids,
                "threshold": threshold
            })
            
            batch_results = list(cursor)
            pairs.extend(batch_results)
            processed += len(batch_ids)
            print(f"Processed {processed}/{total_docs} documents, found {len(batch_results)} pairs in this batch")
        
        print(f"Total similar pairs found: {len(pairs)}")
        if pairs:
            print("Sample pair:")
            print(f"  From: {pairs[0]['from_question'][:50]}...")
            print(f"  To:   {pairs[0]['to_question'][:50]}...")
            print(f"  Similarity: {pairs[0]['similarity']}")
        
        return pairs
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return []

def test_nearest_neighbors(db, collection_name, embedding_field, k=5, threshold=0.8, limit=10):
    """Test finding k-nearest neighbors for each document using the vector index."""
    print(f"\n=== Testing K-Nearest Neighbors (k={k}, threshold={threshold}, limit={limit}) ===")
    try:
        aql = f"""
        FOR doc1 IN {collection_name}
            LIMIT {limit}
            LET neighbors = (
                FOR doc2 IN {collection_name}
                    FILTER doc1._id != doc2._id
                    LET similarity = COSINE_SIMILARITY(doc1.{embedding_field}, doc2.{embedding_field})
                    FILTER similarity >= @threshold
                    SORT similarity DESC
                    LIMIT @k
                    RETURN {{
                        id: doc2._id,
                        similarity: similarity,
                        question: doc2.question
                    }}
            )
            FILTER LENGTH(neighbors) > 0
            RETURN {{
                from: doc1._id,
                from_question: doc1.question,
                neighbors: neighbors
            }}
        """
        
        cursor = db.aql.execute(aql, bind_vars={"threshold": threshold, "k": k})
        results = list(cursor)
        
        print(f"Found neighbors for {len(results)} documents")
        
        # Flatten the results to get pairs
        pairs = []
        for doc in results:
            for neighbor in doc["neighbors"]:
                pairs.append({
                    "from": doc["from"],
                    "to": neighbor["id"],
                    "similarity": neighbor["similarity"],
                    "from_question": doc["from_question"],
                    "to_question": neighbor["question"]
                })
        
        print(f"Total similar pairs found: {len(pairs)}")
        if pairs:
            print("Sample pair:")
            print(f"  From: {pairs[0]['from_question'][:50]}...")
            print(f"  To:   {pairs[0]['to_question'][:50]}...")
            print(f"  Similarity: {pairs[0]['similarity']}")
        
        return pairs
    except Exception as e:
        print(f"Error finding nearest neighbors: {str(e)}")
        return []

def test_approximate_near_cosine(db, collection_name, embedding_field, threshold=0.8, limit=10):
    """Test if APPROX_NEAR_COSINE works in a simple context."""
    print(f"\n=== Testing APPROX_NEAR_COSINE (threshold={threshold}, limit={limit}) ===")
    try:
        # First check if we can get a single similarity value
        aql = f"""
        FOR doc1 IN {collection_name}
            LIMIT 1
            FOR doc2 IN {collection_name}
                FILTER doc1._id != doc2._id
                LIMIT 1
                RETURN APPROX_NEAR_COSINE(doc1.{embedding_field}, doc2.{embedding_field})
        """
        
        try:
            cursor = db.aql.execute(aql)
            result = cursor.next()
            print(f"APPROX_NEAR_COSINE single test result: {result}")
            
            # Now try to use it in a query
            aql = f"""
            FOR doc1 IN {collection_name}
                LIMIT {limit}
                LET neighbors = (
                    FOR doc2 IN {collection_name}
                        FILTER doc1._id != doc2._id
                        LET similarity = APPROX_NEAR_COSINE(doc1.{embedding_field}, doc2.{embedding_field})
                        RETURN {{
                            id: doc2._id,
                            similarity: similarity,
                            question: doc2.question
                        }}
                )
                RETURN {{
                    from: doc1._id,
                    from_question: doc1.question,
                    neighbors: neighbors
                }}
            """
            
            cursor = db.aql.execute(aql)
            results = list(cursor)
            print(f"APPROX_NEAR_COSINE query worked, returned {len(results)} documents with neighbor lists")
            return True
        except Exception as e:
            print(f"APPROX_NEAR_COSINE failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error in APPROX_NEAR_COSINE test: {str(e)}")
        return False

def test_python_similarity(db, collection_name, embedding_field, threshold=0.8, sample_size=10):
    """Test calculating similarity in Python instead of ArangoDB."""
    print(f"\n=== Testing Python-based Similarity (threshold={threshold}, sample_size={sample_size}) ===")
    try:
        # Get sample documents with embeddings
        aql = f"""
        FOR doc IN {collection_name}
            LIMIT {sample_size}
            RETURN {{
                id: doc._id,
                question: doc.question,
                embedding: doc.{embedding_field}
            }}
        """
        cursor = db.aql.execute(aql)
        docs = list(cursor)
        
        print(f"Retrieved {len(docs)} documents for similarity calculation")
        
        # Calculate similarities in Python
        pairs = []
        for i in range(len(docs)):
            for j in range(i+1, len(docs)):
                doc1 = docs[i]
                doc2 = docs[j]
                
                # Convert to numpy arrays
                v1 = np.array(doc1["embedding"])
                v2 = np.array(doc2["embedding"])
                
                # Calculate cosine similarity
                dot_product = np.dot(v1, v2)
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                similarity = dot_product / (norm1 * norm2)
                
                if similarity >= threshold:
                    pairs.append({
                        "from": doc1["id"],
                        "to": doc2["id"],
                        "similarity": float(similarity),
                        "from_question": doc1["question"],
                        "to_question": doc2["question"]
                    })
        
        print(f"Found {len(pairs)} similar pairs above threshold {threshold}")
        if pairs:
            print("Sample pair:")
            print(f"  From: {pairs[0]['from_question'][:50]}...")
            print(f"  To:   {pairs[0]['to_question'][:50]}...")
            print(f"  Similarity: {pairs[0]['similarity']}")
            
            # Verify with ArangoDB calculation
            aql = f"""
            LET doc1 = DOCUMENT("{pairs[0]['from']}")
            LET doc2 = DOCUMENT("{pairs[0]['to']}")
            RETURN COSINE_SIMILARITY(doc1.{embedding_field}, doc2.{embedding_field})
            """
            cursor = db.aql.execute(aql)
            db_similarity = cursor.next()
            print(f"  Verification via ArangoDB: {db_similarity}")
            print(f"  Difference: {abs(pairs[0]['similarity'] - db_similarity)}")
        
        return pairs
    except Exception as e:
        print(f"Error in Python similarity: {str(e)}")
        return []

def main():
    """Run tests to debug ArangoDB vector similarity issues."""
    print("Starting ArangoDB vector similarity debugging")
    
    client, db = connect_arango()
    collection_name = CONFIG["search"]["collection_name"]
    embedding_field = CONFIG["embedding"]["field"]
    
    # Run tests in sequence
    print(f"\nCollection: {collection_name}")
    print(f"Embedding field: {embedding_field}")
    
    # First check if basic cosine similarity works
    if not test_basic_cosine_similarity(db, collection_name, embedding_field):
        print("CRITICAL: Basic cosine similarity test failed!")
        return
    
    # Try test with different thresholds
    for threshold in [0.9, 0.8, 0.7, 0.5]:
        test_batched_approach(db, collection_name, embedding_field, threshold=threshold)
    
    # Try nearest neighbors approach
    test_nearest_neighbors(db, collection_name, embedding_field)
    
    # Test APPROX_NEAR_COSINE
    test_approximate_near_cosine(db, collection_name, embedding_field)
    
    # Test Python-based similarity
    test_python_similarity(db, collection_name, embedding_field)
    
    print("\nVector similarity debugging complete!")

if __name__ == "__main__":
    main()