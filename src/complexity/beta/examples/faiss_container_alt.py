import os
import sys
import numpy as np
import json
import requests
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
    f"{log_path}/faiss_similarity.log", 
    rotation="10 MB", 
    retention="1 week", 
    level="DEBUG", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def find_similar_documents(db_name, collection_name, embedding_field, threshold=0.8, 
                           arango_host="http://localhost:8529", username="root", password="openSesame",
                           faiss_url="http://localhost:5001"):
    """
    Find similar document pairs using FAISS-GPU container.
    
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
    faiss_url: str
        URL of the FAISS service
        
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
            # Create an index file and ID mapping file for FAISS-GPU container
            logger.info(f"Setting up index for FAISS-GPU service at {faiss_url}")
            
            # Generate unique filenames for this operation
            import time
            import uuid
            index_name = f"temp_index_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Create a directory for the index files if it doesn't exist
            index_dir = "faiss_indexes"
            os.makedirs(index_dir, exist_ok=True)
            
            # Path to index and ID mapping files
            index_path = os.path.join(index_dir, f"{index_name}.index")
            id_mapping_path = os.path.join(index_dir, f"{index_name}.json")
            
            # Create ID mapping for lookups
            id_mapping = {str(i): {"doc_id": ids[i], "question": questions[i]} for i in range(len(ids))}
            
            # Save ID mapping to disk
            logger.info(f"Saving ID mapping to {id_mapping_path}")
            with open(id_mapping_path, "w") as f:
                json.dump(id_mapping, f)
            
            logger.info(f"Building FAISS index locally and saving to disk")
            
            # Build the index locally and save it to disk
            import faiss
            
            # Create a simple flat index
            index = faiss.IndexFlatL2(dimension)
            
            # Add normalized vectors
            faiss.normalize_L2(normalized_embeddings)
            index.add(normalized_embeddings)
            
            # Save to disk
            faiss.write_index(index, index_path)
            logger.info(f"Successfully saved FAISS index to {index_path}")
            
        except Exception as e:
            logger.error(f"Error creating FAISS index files: {str(e)}")
            return []
        
        try:
            # Check FAISS service directly
            logger.info(f"Checking FAISS service at {faiss_url}")
            
            # Try a basic query to the service root to see what it returns
            try:
                root_response = requests.get(faiss_url)
                logger.info(f"FAISS service root response: {root_response.status_code}")
                logger.debug(f"FAISS service root content: {root_response.text[:100]}...")
            except Exception as e:
                logger.warning(f"Error checking FAISS service root: {str(e)}")
            
            # Structure to hold the results
            results = []
            
            # For each document, find its similar documents
            k = min(100, len(normalized_embeddings))  # Don't search for more neighbors than we have docs
            logger.info(f"Searching for up to {k} similar documents per document with threshold {threshold}")
            
            total_neighbors = 0
            
            # Try both common possible endpoints
            endpoints = [
                "/search",
                "/api/search",
                "/search_by_id",
                "/knn"
            ]
            
            # Find one document's neighbors to determine the correct endpoint
            i = 0  # Use the first document
            successful_endpoint = None
            
            for endpoint in endpoints:
                try:
                    logger.info(f"Trying endpoint: {faiss_url}{endpoint}")
                    
                    # Prepare the query data - adjust based on possible API formats
                    # Format 1: Expects index file and query vector
                    query_data = {
                        "query": normalized_embeddings[i].tolist(),
                        "k": 5,
                        "index_file": f"{index_name}.index",
                        "id_mapping_file": f"{index_name}.json"
                    }
                    
                    # Try the endpoint
                    response = requests.post(f"{faiss_url}{endpoint}", json=query_data, timeout=30)
                    
                    if response.status_code == 200:
                        successful_endpoint = endpoint
                        logger.info(f"Found working endpoint: {endpoint}")
                        
                        # Log response to understand the format
                        logger.debug(f"Response format: {response.json()}")
                        break
                    else:
                        logger.warning(f"Endpoint {endpoint} returned status {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"Error testing endpoint {endpoint}: {str(e)}")
                    
            if not successful_endpoint:
                logger.error("Could not find a working endpoint for FAISS service")
                return []
                
            # Use tqdm to show progress during similarity search
            for i in trange(len(normalized_embeddings), desc="Searching for similar documents", unit="docs"):
                try:
                    # Get the query vector for this document
                    query_vector = normalized_embeddings[i].tolist()
                    query_id = ids[i]
                    
                    # Prepare search request for the working endpoint
                    search_data = {
                        "query": query_vector,
                        "k": k,
                        "index_file": f"{index_name}.index",
                        "id_mapping_file": f"{index_name}.json"
                    }
                    
                    # Perform search
                    search_response = requests.post(f"{faiss_url}{successful_endpoint}", json=search_data)
                    
                    if search_response.status_code != 200:
                        logger.warning(f"Search failed for document {ids[i]}: {search_response.text}")
                        continue
                        
                    # Parse results - adapt based on actual response format
                    search_results = search_response.json()
                    
                    # Filter out self-match and process results
                    # Response format might be {"results": [{"id": "0", "score": 1.0}, ...]}
                    neighbors = []
                    
                    # Try to adapt to different possible response formats
                    if "results" in search_results:
                        # Format: {"results": [{"id": "0", "score": 1.0}, ...]}
                        for hit in search_results["results"]:
                            hit_id = int(hit["id"]) if "id" in hit else int(hit.get("index", -1))
                            score = hit.get("score", 0.0)
                            
                            # Skip self-match
                            if hit_id == i:
                                continue
                                
                            # Only keep results above threshold
                            if score >= threshold:
                                neighbors.append({
                                    'id': ids[hit_id],
                                    'similarity': float(score),
                                    'question': questions[hit_id]
                                })
                    
                    elif "indices" in search_results and ("distances" in search_results or "similarities" in search_results):
                        # Format: {"indices": [0, 1, 2], "distances": [0.0, 0.2, 0.3]}
                        indices = search_results["indices"]
                        
                        # Some APIs return distances (lower is better), others return similarities (higher is better)
                        if "distances" in search_results:
                            # Convert L2 distances to similarities for normalized vectors
                            distances = search_results["distances"]
                            scores = [1.0 - (dist/2.0) if dist < 2.0 else 0.0 for dist in distances]
                        else:
                            # Directly use similarities
                            scores = search_results["similarities"]
                        
                        for idx, score in zip(indices, scores):
                            # Skip self-match
                            if idx == i:
                                continue
                                
                            # Only keep results above threshold
                            if score >= threshold:
                                neighbors.append({
                                    'id': ids[idx],
                                    'similarity': float(score),
                                    'question': questions[idx]
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
            
            # Clean up - delete temporary index files
            try:
                logger.info(f"Cleaning up: deleting temporary FAISS index files")
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(id_mapping_path):
                    os.remove(id_mapping_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error searching for similar documents: {str(e)}")
            # Try to clean up even if there was an error
            try:
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(id_mapping_path):
                    os.remove(id_mapping_path)
            except:
                pass
            return []
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return []
        
    return results


if __name__ == "__main__":
    try:
        logger.info("Starting similarity search process using FAISS-GPU")
        
        # Get FAISS service URL from environment or use default
        faiss_url = os.environ.get("FAISS_SERVICE_URL", "http://localhost:5001")
        
        # Example usage
        results = find_similar_documents(
            db_name='memory_bank',  # Replace with your database name
            collection_name='complexity',  # Replace with your collection name
            embedding_field='embedding',  # Replace with your embedding field name
            threshold=0.8,
            faiss_url=faiss_url
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