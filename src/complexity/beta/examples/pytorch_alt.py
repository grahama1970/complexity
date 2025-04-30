import os
import sys
import numpy as np
import json
import torch  # Add PyTorch import
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
    f"{log_path}/similarity_search.log", 
    rotation="10 MB", 
    retention="1 week", 
    level="DEBUG", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)



def find_similar_documents(db_name, collection_name, embedding_field, threshold=0.8, 
                           arango_host="http://localhost:8529", username="root", password="openSesame"):
    """
    Find similar document pairs using PyTorch GPU acceleration.
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
                
            # Convert to numpy array
            embeddings = np.array(embeddings, dtype=np.float32)
            logger.info(f"Extracted {len(embeddings)} valid embeddings of dimension {embeddings.shape[1]}")
            
            # Normalize vectors for cosine similarity
            logger.debug("Normalizing vectors for cosine similarity")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            logger.info(f"Normalized {len(normalized_embeddings)} vectors for cosine similarity")
            
            # Perform GPU-accelerated similarity search
            results = gpu_similarity_search(
                embeddings=normalized_embeddings,
                ids=ids,
                questions=questions,
                threshold=threshold,
                batch_size=128  # Adjust based on your GPU memory
            )
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
            
    except ServerConnectionError as e:
        logger.error(f"Failed to connect to ArangoDB server: {str(e)}")
        return []
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
        
    return results


def gpu_similarity_search(embeddings, ids, questions, threshold=0.8, batch_size=128):
    """
    Perform GPU-accelerated similarity search using PyTorch batching for better performance.
    
    Parameters:
    -----------
    embeddings: numpy.ndarray
        The normalized embedding vectors to search within
    ids: list
        List of document IDs corresponding to embeddings
    questions: list
        List of questions corresponding to embeddings
    threshold: float
        Similarity threshold (0.0-1.0)
    batch_size: int
        Batch size for processing queries (to avoid GPU memory issues)
        
    Returns:
    --------
    list of dicts
        Each dict contains source document and its similar neighbors
    """
    # Determine if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Convert numpy array to PyTorch tensor and move to device
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    # Structure to hold the results
    results = []
    total_neighbors = 0
    k = min(100, len(embeddings))
    
    # Process in batches to avoid GPU memory issues
    num_batches = (len(embeddings) + batch_size - 1) // batch_size
    
    with tqdm(total=len(embeddings), desc="Searching for similar documents", unit="docs") as pbar:
        for batch_idx in range(num_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(embeddings))
            batch_indices = list(range(start_idx, end_idx))
            
            # Create batch query tensor
            query_tensor = embeddings_tensor[batch_indices]
            
            # Compute similarities for all pairs in the batch at once
            # (batch_size x embedding_dim) @ (embedding_dim x total_embeddings)
            similarities = torch.mm(query_tensor, embeddings_tensor.t())
            
            # Get top-k for each query
            top_similarities, top_indices = torch.topk(similarities, k=min(k, len(embeddings)))
            
            # Process each query in the batch
            for i, query_idx in enumerate(batch_indices):
                query_id = ids[query_idx]
                query_question = questions[query_idx]
                
                # Get similarities and indices for this query
                sims = top_similarities[i].cpu().numpy()
                inds = top_indices[i].cpu().numpy()
                
                # Filter out self-matches and apply threshold
                neighbors = []
                for j, (idx, similarity) in enumerate(zip(inds, sims)):
                    # Skip self-match
                    if idx == query_idx:
                        continue
                        
                    # Apply threshold
                    if similarity >= threshold:
                        neighbors.append({
                            'id': ids[idx],
                            'similarity': float(similarity),
                            'question': questions[idx]
                        })
                
                # Sort neighbors by similarity (highest first)
                neighbors.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Add to results if we found any neighbors
                if neighbors:
                    results.append({
                        'from': query_id,
                        'from_question': query_question,
                        'neighbors': neighbors
                    })
                    total_neighbors += len(neighbors)
            
            # Update progress bar
            pbar.update(len(batch_indices))
    
    logger.info(f"Completed search: found {len(results)} documents with similar pairs, {total_neighbors} total connections")
    return results


if __name__ == "__main__":
    try:
        logger.info("Starting GPU-accelerated similarity search process")
        
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