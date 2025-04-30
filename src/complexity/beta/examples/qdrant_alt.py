import os
import sys
import numpy as np
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from arango import ArangoClient
from arango.exceptions import ArangoError, ServerConnectionError
from loguru import logger
from tqdm import tqdm, trange

# Configure loguru
log_path = "logs"
os.makedirs(log_path, exist_ok=True)

# Remove default logger and add custom configuration
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(
    f"{log_path}/qdrant_similarity.log", 
    rotation="10 MB", 
    retention="1 week", 
    level="DEBUG", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def find_similar_documents(db_name, collection_name, embedding_field, threshold=0.8, 
                           arango_host="http://localhost:8529", username="root", password="openSesame",
                           qdrant_host="localhost", qdrant_port=6333):
    """
    Find similar document pairs using Qdrant instead of FAISS.
    
    Parameters:
    -----------
    db_name: str
        Name of the ArangoDB database
    collection_name: str
        Name of the collection containing documents with embeddings
    embedding_field: str
        Name of the field containing vector embeddings
    threshold: float
        Similarity threshold (0.0-1.0)
    arango_host: str
        ArangoDB host URL
    username: str
        ArangoDB username
    password: str
        ArangoDB password
    qdrant_host: str
        Qdrant host address
    qdrant_port: int
        Qdrant port
        
    Returns:
    --------
    list of dicts
        Each dict contains source document and its similar neighbors
    """
    results = []
    
    try:
        logger.info(f"Connecting to ArangoDB at {arango_host}")
        # Connect to ArangoDB
        client = ArangoClient(hosts=arango_host)
        
        try:
            db = client.db(db_name, username=username, password=password)
            logger.info(f"Successfully connected to database '{db_name}'")
        except ArangoError as e:
            logger.error(f"Failed to connect to database '{db_name}': {str(e)}")
            return []
        
        # Verify collection exists
        try:
            if not db.has_collection(collection_name):
                logger.error(f"Collection '{collection_name}' does not exist in database '{db_name}'")
                return []
            logger.info(f"Found collection '{collection_name}'")
        except ArangoError as e:
            logger.error(f"Error checking collection '{collection_name}': {str(e)}")
            return []
        
        # Retrieve all documents
        try:
            logger.debug(f"Querying for documents with embedding field '{embedding_field}'")
            aql = f"""
            FOR doc IN {collection_name}
                RETURN {{
                    _id: doc._id,
                    question: doc.question,
                    {embedding_field}: doc.{embedding_field}
                }}
            """
            cursor = db.aql.execute(aql)
            
            # Use tqdm to show progress during document retrieval
            logger.info(f"Retrieving documents from collection '{collection_name}'")
            documents = []
            with tqdm(desc="Retrieving documents", unit="docs") as pbar:
                for doc in cursor:
                    documents.append(doc)
                    pbar.update(1)
            
            logger.info(f"Retrieved {len(documents)} documents from collection")
            
            if not documents:
                logger.warning(f"No documents found in collection '{collection_name}'")
                return []
                
            # Check if first document has embedding field
            if embedding_field not in documents[0]:
                logger.error(f"Embedding field '{embedding_field}' not found in documents")
                return []
                
        except ArangoError as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
            
        try:
            # Extract embeddings and metadata
            logger.debug("Extracting embeddings and metadata from documents")
            embeddings = []
            ids = []
            questions = []
            
            # Show progress while extracting embeddings
            with tqdm(desc="Processing documents", total=len(documents), unit="docs") as pbar:
                for doc in documents:
                    try:
                        # Ensure embedding exists and has proper format
                        if embedding_field not in doc or not isinstance(doc[embedding_field], list):
                            logger.warning(f"Document {doc['_id']} has invalid or missing embedding, skipping")
                            pbar.update(1)
                            continue
                            
                        embeddings.append(doc[embedding_field])
                        ids.append(doc["_id"])
                        questions.append(doc["question"])
                        pbar.update(1)
                    except KeyError as e:
                        logger.warning(f"Document {doc.get('_id', 'unknown')} is missing field: {str(e)}")
                        pbar.update(1)
                        continue
            
            # Verify we have documents with valid embeddings
            if not embeddings:
                logger.error("No valid embeddings found in documents")
                return []
                
            # Convert to numpy array for normalization
            embeddings_np = np.array(embeddings, dtype=np.float32)
            dimension = embeddings_np.shape[1]
            logger.info(f"Extracted {len(embeddings_np)} valid embeddings of dimension {dimension}")
            
            # Normalize vectors for cosine similarity
            logger.debug("Normalizing vectors for cosine similarity")
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            normalized_embeddings = embeddings_np / norms
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {str(e)}")
            return []
            
        try:
            # Connect to Qdrant
            logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
            qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            
            # Create temporary collection for similarity search
            # Using a unique name based on timestamp to avoid conflicts
            import time
            temp_collection_name = f"temp_similarity_{int(time.time())}"
            
            # Create collection - using cosine distance since we normalized vectors
            logger.info(f"Creating temporary Qdrant collection '{temp_collection_name}'")
            qdrant_client.create_collection(
                collection_name=temp_collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
            
            # Prepare points for Qdrant
            logger.info("Preparing vectors for Qdrant insertion")
            points = []
            for i, embedding in enumerate(normalized_embeddings):
                points.append(PointStruct(
                    id=i,  # Use position as ID for easy reference
                    vector=embedding.tolist(),
                    payload={
                        "doc_id": ids[i],
                        "question": questions[i]
                    }
                ))
            
            # Upload to Qdrant in batches
            batch_size = 100  # Adjust based on your requirements
            total_batches = (len(points) + batch_size - 1) // batch_size
            
            logger.info(f"Uploading vectors to Qdrant in {total_batches} batches")
            for i in trange(0, len(points), batch_size, desc="Uploading to Qdrant"):
                batch = points[i:i+batch_size]
                qdrant_client.upload_points(
                    collection_name=temp_collection_name,
                    points=batch
                )
            
            logger.info(f"Successfully uploaded {len(points)} vectors to Qdrant")
            
        except Exception as e:
            logger.error(f"Error setting up Qdrant: {str(e)}")
            return []
        
        try:
            # Search for similar pairs
            # We'll search for similar documents for each document
            k = min(100, len(documents))  # Don't search for more neighbors than we have documents
            logger.info(f"Searching for up to {k} similar documents per document with threshold {threshold}")
            
            # Structure to hold the results
            results = []
            
            # For each document, find its similar documents
            total_neighbors = 0
            
            # Use tqdm to show progress during similarity search
            for i in trange(len(normalized_embeddings), desc="Searching for similar documents", unit="docs"):
                try:
                    # Get the embedding vector for this document
                    query_vector = normalized_embeddings[i].tolist()
                    query_id = ids[i]
                    
                    # Search in Qdrant - excluding the document itself
                    search_results = qdrant_client.search(
                        collection_name=temp_collection_name,
                        query_vector=query_vector,
                        limit=k,
                        score_threshold=threshold,
                        with_payload=True
                    )
                    
                    # Filter out self-match and process results
                    neighbors = []
                    for hit in search_results:
                        # Skip self (should be the first result with score=1.0)
                        if hit.id == i:
                            continue
                            
                        # Add to neighbors list
                        neighbors.append({
                            'id': hit.payload['doc_id'],
                            'similarity': hit.score,
                            'question': hit.payload['question']
                        })
                    
                    # Sort neighbors by similarity (highest first)
                    neighbors.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Add to results if we found any neighbors
                    if neighbors:
                        results.append({
                            'from': query_id,
                            'from_question': questions[i],
                            'neighbors': neighbors
                        })
                        total_neighbors += len(neighbors)
                
                except Exception as e:
                    logger.warning(f"Error processing document {ids[i]}: {str(e)}")
                    continue
            
            logger.info(f"Completed search: found {len(results)} documents with similar pairs, {total_neighbors} total connections")
            
            # Clean up - delete temporary collection
            logger.info(f"Cleaning up: deleting temporary Qdrant collection '{temp_collection_name}'")
            qdrant_client.delete_collection(collection_name=temp_collection_name)
            
        except Exception as e:
            logger.error(f"Error searching for similar documents: {str(e)}")
            # Try to clean up even if there was an error
            try:
                qdrant_client.delete_collection(collection_name=temp_collection_name)
            except:
                pass
            return []
            
    except ServerConnectionError as e:
        logger.error(f"Failed to connect to ArangoDB server: {str(e)}")
        return []
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return []
        
    return results


if __name__ == "__main__":
    try:
        logger.info("Starting similarity search process using Qdrant")
        
        # Example usage
        results = find_similar_documents(
            db_name='memory_bank',  # Replace with your database name
            collection_name='complexity',  # Replace with your collection name
            embedding_field='embedding',  # Replace with your embedding field name
            threshold=0.8
        )

        # Print summary
        logger.info(f"Found {len(results)} documents with similar pairs")
        
        if results:
            doc = results[0]
            logger.info(f"Example: Document '{doc['from']}' has {len(doc['neighbors'])} similar documents")
            if doc['neighbors']:
                neighbor = doc['neighbors'][0]
                logger.info(f"  Most similar: '{neighbor['id']}' with similarity {neighbor['similarity']:.4f}")
        
        # Optional: Save results
        try:
            output_file = 'similar_documents.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {str(e)}")
            
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Process failed with error: {str(e)}")
        sys.exit(1)