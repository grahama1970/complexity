import os
import sys
import numpy as np
import json
import torch 
import faiss
import time
import requests
import typer
from typing import Optional
from arango import ArangoClient
from arango.exceptions import ArangoError, ServerConnectionError
from loguru import logger
from tqdm import tqdm, trange
from tabulate import tabulate

# Create Typer app
app = typer.Typer(help="Compare FAISS-CPU and PyTorch for similarity search")

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

def load_documents_from_arango(db_name, collection_name, embedding_field,
                               arango_host="http://localhost:8529", 
                               username="root", password="openSesame",
                               batch_size=1000, max_documents=None):
    """
    Optimized function to load documents from ArangoDB with batching support.
    
    Parameters:
    -----------
    db_name: str
        Name of the ArangoDB database
    collection_name: str
        Name of the collection containing documents with embeddings
    embedding_field: str
        Name of the field containing vector embeddings
    arango_host: str
        ArangoDB host URL
    username: str
        ArangoDB username
    password: str
        ArangoDB password
    batch_size: int
        Size of batches when retrieving documents
    max_documents: int or None
        Maximum number of documents to retrieve (None for all)
        
    Returns:
    --------
    tuple (embeddings, ids, questions, dimension)
        embeddings: numpy.ndarray of normalized embeddings
        ids: list of document IDs
        questions: list of document questions
        dimension: dimension of the embeddings
    """
    try:
        # Connect to ArangoDB
        start_connection = time.time()
        logger.info(f"Connecting to ArangoDB at {arango_host}")
        client = ArangoClient(hosts=arango_host)
        
        try:
            db = client.db(db_name, username=username, password=password)
            logger.info(f"Successfully connected to database '{db_name}'")
        except ArangoError as e:
            logger.error(f"Failed to connect to database '{db_name}': {str(e)}")
            return None, None, None, None
        
        connection_time = time.time() - start_connection
        logger.info(f"Connection established in {connection_time:.2f} seconds")
        
        # Verify collection exists
        try:
            collections = db.collections()
            collection_names = [col['name'] for col in collections]
            if collection_name not in collection_names:
                logger.error(f"Collection '{collection_name}' does not exist in database '{db_name}'")
                return None, None, None, None
            logger.info(f"Found collection '{collection_name}'")
            
            # Get collection count for progress tracking
            count = db.collection(collection_name).count()
            logger.info(f"Collection contains {count} documents")
            
            # Limit number of documents if specified
            if max_documents is not None and max_documents > 0:
                logger.info(f"Will process at most {max_documents} documents (out of {count})")
                expected_docs = min(count, max_documents)
            else:
                expected_docs = count
                
        except ArangoError as e:
            logger.error(f"Error checking collection '{collection_name}': {str(e)}")
            return None, None, None, None
        
        # Extract embeddings and metadata
        embeddings = []
        ids = []
        questions = []
        
        load_start_time = time.time()
        
        # Count total number of documents with valid embeddings for progress tracking
        start_query = time.time()
        aql_count = f"""
        RETURN LENGTH(
            FOR doc IN {collection_name}
            FILTER HAS(doc, "{embedding_field}")
            {f"LIMIT {max_documents}" if max_documents is not None else ""}
            RETURN 1
        )
        """
        valid_count = db.aql.execute(aql_count).next()
        count_time = time.time() - start_query
        
        logger.info(f"Found {valid_count} documents with '{embedding_field}' field (count query: {count_time:.2f}s)")
        
        # Fetch documents in batches
        total_processed = 0
        with tqdm(total=valid_count, desc="Retrieving documents", unit="docs") as pbar:
            while (max_documents is None or total_processed < max_documents):
                # Calculate how many documents to retrieve in this batch
                if max_documents is not None:
                    current_batch_size = min(batch_size, max_documents - total_processed)
                else:
                    current_batch_size = batch_size
                
                if current_batch_size <= 0:
                    break
                
                # Query to get a batch of documents
                aql = f"""
                FOR doc IN {collection_name}
                FILTER HAS(doc, "{embedding_field}")
                LIMIT {total_processed}, {current_batch_size}
                RETURN {{
                    _id: doc._id,
                    question: doc.question,
                    {embedding_field}: doc.{embedding_field}
                }}
                """
                
                # Time the query execution
                batch_start = time.time()
                cursor = db.aql.execute(aql)
                batch_docs = list(cursor)
                batch_time = time.time() - batch_start
                
                if not batch_docs:
                    logger.info("No more documents to process")
                    break
                
                # Log batch performance
                logger.debug(f"Batch {total_processed // batch_size + 1}: Retrieved {len(batch_docs)} docs in {batch_time:.2f}s ({len(batch_docs)/batch_time:.1f} docs/s)")
                
                # Process batch
                processed_in_batch = 0
                for doc in batch_docs:
                    try:
                        # Ensure embedding exists and has proper format
                        if not isinstance(doc[embedding_field], list):
                            continue
                            
                        embeddings.append(doc[embedding_field])
                        ids.append(doc["_id"])
                        questions.append(doc["question"])
                        processed_in_batch += 1
                        
                    except KeyError:
                        continue
                
                total_processed += processed_in_batch
                pbar.update(processed_in_batch)
                
                # Stop if we've processed enough documents
                if max_documents is not None and total_processed >= max_documents:
                    break
        
        fetch_time = time.time() - load_start_time
        docs_per_second = len(embeddings)/fetch_time if fetch_time > 0 else 0
        logger.info(f"Document fetching completed in {fetch_time:.2f} seconds ({docs_per_second:.1f} docs/s)")
        
        # Verify we have documents with valid embeddings
        if not embeddings:
            logger.error("No valid embeddings found in documents")
            return None, None, None, None
            
        # Convert to numpy array
        start_time = time.time()
        embeddings_np = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_np.shape[1]
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / norms
        normalize_time = time.time() - start_time
        
        logger.info(f"Extracted and normalized {len(embeddings_np)} valid embeddings of dimension {dimension} in {normalize_time:.2f}s")
        
        return normalized_embeddings, ids, questions, dimension
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None


def load_documents_optimized(db_name, collection_name, embedding_field,
                            arango_host="http://localhost:8529", 
                            username="root", password="openSesame",
                            batch_size=1000, max_documents=None):
    """
    Optimized version of the original load_documents_from_arango function.
    """
    try:
        # Connect to ArangoDB
        start_connection = time.time()
        logger.info(f"Connecting to ArangoDB at {arango_host}")
        client = ArangoClient(hosts=arango_host)
        
        try:
            db = client.db(db_name, username=username, password=password)
            logger.info(f"Successfully connected to database '{db_name}'")
        except ArangoError as e:
            logger.error(f"Failed to connect to database '{db_name}': {str(e)}")
            return None, None, None, None
        
        connection_time = time.time() - start_connection
        logger.info(f"Connection established in {connection_time:.2f} seconds")
        
        # Verify collection exists
        try:
            collections = db.collections()
            collection_names = [col['name'] for col in collections]
            if collection_name not in collection_names:
                logger.error(f"Collection '{collection_name}' does not exist in database '{db_name}'")
                return None, None, None, None
            logger.info(f"Found collection '{collection_name}'")
            
            # Get collection count - directly use a count-only query for efficiency
            count_aql = f"RETURN COLLECTION_COUNT('{collection_name}')"
            count = db.aql.execute(count_aql).next()
            logger.info(f"Collection contains {count} documents")
            
            # Limit number of documents if specified
            if max_documents is not None and max_documents > 0:
                logger.info(f"Will process at most {max_documents} documents (out of {count})")
                expected_docs = min(count, max_documents)
            else:
                expected_docs = count
                
        except ArangoError as e:
            logger.error(f"Error checking collection '{collection_name}': {str(e)}")
            return None, None, None, None
        
        # Extract embeddings and metadata
        embeddings = []
        ids = []
        questions = []
        
        load_start_time = time.time()
        
        # Optimization: Use a more efficient query that pre-filters documents
        # Directly get only the documents with embeddings in a single query
        # Use a larger batch size (2000 instead of 1000) for fewer trips to the database
        limit_clause = f"LIMIT {max_documents}" if max_documents is not None else ""
        
        aql = f"""
        FOR doc IN {collection_name}
        FILTER HAS(doc, "{embedding_field}")
        {limit_clause}
        RETURN {{
            _id: doc._id,
            question: doc.question,
            embedding: doc.{embedding_field}
        }}
        """
        
        # Execute the query with a larger batch size
        cursor = db.aql.execute(aql, batch_size=batch_size)
        
        # Process all documents
        with tqdm(total=expected_docs, desc="Retrieving documents", unit="docs") as pbar:
            for doc in cursor:
                try:
                    # Skip invalid embeddings
                    if not isinstance(doc["embedding"], list):
                        continue
                        
                    embeddings.append(doc["embedding"])
                    ids.append(doc["_id"])
                    questions.append(doc["question"])
                    pbar.update(1)
                    
                except KeyError:
                    continue
        
        fetch_time = time.time() - load_start_time
        docs_per_second = len(embeddings)/fetch_time if fetch_time > 0 else 0
        logger.info(f"Document fetching completed in {fetch_time:.2f} seconds ({docs_per_second:.1f} docs/s)")
        
        # Verify we have documents with valid embeddings
        if not embeddings:
            logger.error("No valid embeddings found in documents")
            return None, None, None, None
            
        # Convert to numpy array - use optimized operations
        start_time = time.time()
        embeddings_np = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_np.shape[1]
        
        # Normalize vectors for cosine similarity - use optimized operations
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / norms
        normalize_time = time.time() - start_time
        
        logger.info(f"Extracted and normalized {len(embeddings_np)} valid embeddings of dimension {dimension} in {normalize_time:.2f}s")
        
        return normalized_embeddings, ids, questions, dimension
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None





def pytorch_similarity_search(embeddings, ids, questions, threshold=0.8, batch_size=128, 
                              fp16=True, cuda_streams=True):
    """
    Optimized GPU-accelerated similarity search using PyTorch.
    
    Parameters:
    -----------
    embeddings: numpy.ndarray
        Normalized embedding vectors 
    ids: list
        Document IDs
    questions: list
        Document questions
    threshold: float
        Similarity threshold (0.0-1.0)
    batch_size: int
        Batch size for processing
    fp16: bool
        Whether to use FP16 (half precision) for faster computation
    cuda_streams: bool
        Whether to use CUDA streams for concurrent processing
        
    Returns:
    --------
    tuple (results, elapsed_time)
        results: list of similarity results
        elapsed_time: time taken for the search
    """
    # Record start time
    start_time = time.time()
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set up CUDA streams if needed
    stream = torch.cuda.Stream() if cuda_streams and torch.cuda.is_available() else None
    
    # Configure precision
    dtype = torch.float16 if fp16 and device.type == 'cuda' else torch.float32
    logger.info(f"Using {dtype} precision")
    
    # Convert embeddings to tensor with appropriate precision
    with torch.cuda.stream(stream) if stream else nullcontext():
        embeddings_tensor = torch.tensor(embeddings, dtype=dtype).to(device)
    
    if stream:
        torch.cuda.current_stream().wait_stream(stream)
    
    # Structure to hold results
    results = []
    total_neighbors = 0
    k = min(100, len(embeddings))
    
    # Calculate number of batches
    num_batches = (len(embeddings) + batch_size - 1) // batch_size
    
    with tqdm(total=len(embeddings), desc="PyTorch similarity search", unit="docs") as pbar:
        for batch_idx in range(num_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(embeddings))
            batch_indices = list(range(start_idx, end_idx))
            
            # Extract batch queries
            with torch.cuda.stream(stream) if stream else nullcontext():
                query_tensor = embeddings_tensor[batch_indices]
                
                # Compute similarities for all pairs in the batch at once
                # Use torch.matmul for better performance with half precision
                similarities = torch.matmul(query_tensor, embeddings_tensor.t())
                
                # Get top-k for each query
                top_similarities, top_indices = torch.topk(similarities, k=min(k, len(embeddings)))
            
            if stream:
                torch.cuda.current_stream().wait_stream(stream)
            
            # Process results
            for i, query_idx in enumerate(batch_indices):
                query_id = ids[query_idx]
                query_question = questions[query_idx]
                
                # Get similarities and indices for this query
                sims = top_similarities[i].cpu().numpy()
                inds = top_indices[i].cpu().numpy()
                
                # Filter results
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
                
                # Sort neighbors
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
    
    elapsed_time = time.time() - start_time
    logger.info(f"PyTorch search completed in {elapsed_time:.2f} seconds")
    logger.info(f"Found {len(results)} documents with similar pairs, {total_neighbors} total connections")
    
    return results, elapsed_time

def pytorch_enhanced_search(embeddings, ids, questions, threshold=0.8, batch_size=128, 
                           fp16=True, cuda_streams=True, use_ann=True, nlist=100, nprobe=10):
    """
    Enhanced GPU-accelerated similarity search using PyTorch with FAISS-like features.
    
    Parameters:
    -----------
    embeddings: numpy.ndarray
        Normalized embedding vectors 
    ids: list
        Document IDs
    questions: list
        Document questions
    threshold: float
        Similarity threshold (0.0-1.0)
    batch_size: int
        Batch size for processing
    fp16: bool
        Whether to use FP16 (half precision) for faster computation
    cuda_streams: bool
        Whether to use CUDA streams for concurrent processing
    use_ann: bool
        Whether to use approximate nearest neighbor search (IVFPQ-like)
    nlist: int
        Number of clusters for ANN index (if use_ann=True)
    nprobe: int
        Number of clusters to search in ANN index (if use_ann=True)
        
    Returns:
    --------
    tuple (results, elapsed_time)
        results: list of similarity results
        elapsed_time: time taken for the search
    """
    # Record start time
    start_time = time.time()
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get dimensions
    num_vectors, dimension = embeddings.shape
    
    # Configure precision
    dtype = torch.float16 if fp16 and device.type == 'cuda' else torch.float32
    logger.info(f"Using {dtype} precision")
    
    # Create CUDA streams if needed
    main_stream = torch.cuda.Stream() if cuda_streams and torch.cuda.is_available() else None
    
    # ANN preparation - clustering (if use_ann is True)
    if use_ann and num_vectors > nlist * 39:  # Only use ANN if we have enough vectors
        logger.info(f"Using approximate nearest neighbor search with {nlist} clusters")
        
        # Convert embeddings to tensor
        embeddings_tensor = torch.tensor(embeddings, dtype=dtype).to(device)
        
        # Perform clustering using k-means (simulating FAISS IVF)
        with torch.cuda.stream(main_stream) if main_stream else nullcontext():
            # Initialize centroids randomly from dataset
            perm = torch.randperm(num_vectors, device=device)
            centroids = embeddings_tensor[perm[:nlist]].clone()
            
            # Run k-means for a few iterations
            max_iter = 10
            logger.info(f"Training clustering with {max_iter} iterations")
            with tqdm(total=max_iter, desc="Training clusters", unit="iter") as pbar:
                for i in range(max_iter):
                    # Compute distances to centroids
                    distances = torch.cdist(embeddings_tensor, centroids, p=2.0)
                    
                    # Assign vectors to nearest centroid
                    assignments = torch.argmin(distances, dim=1)
                    
                    # Update centroids
                    new_centroids = torch.zeros_like(centroids)
                    counts = torch.zeros(nlist, dtype=torch.int, device=device)
                    
                    # This is a simplified update; FAISS has more sophisticated methods
                    for j in range(num_vectors):
                        centroid_idx = assignments[j]
                        new_centroids[centroid_idx] += embeddings_tensor[j]
                        counts[centroid_idx] += 1
                    
                    # Avoid division by zero
                    mask = counts > 0
                    for j in range(nlist):
                        if mask[j]:
                            new_centroids[j] /= counts[j]
                        else:
                            # If a centroid has no points, reinitialize it
                            idx = torch.randint(0, num_vectors, (1,), device=device)
                            new_centroids[j] = embeddings_tensor[idx]
                    
                    # Update centroids
                    centroids = new_centroids
                    pbar.update(1)
        
        # Create inverted lists (mapping from cluster to vector indices)
        inverted_lists = [[] for _ in range(nlist)]
        with torch.cuda.stream(main_stream) if main_stream else nullcontext():
            # Compute final assignments
            distances = torch.cdist(embeddings_tensor, centroids, p=2.0)
            assignments = torch.argmin(distances, dim=1).cpu().numpy()
            
            # Build inverted lists
            for i in range(num_vectors):
                inverted_lists[assignments[i]].append(i)
        
        if main_stream:
            torch.cuda.current_stream().wait_stream(main_stream)
        
        # Convert inverted lists to tensor format for faster access
        inverted_list_lengths = [len(lst) for lst in inverted_lists]
        max_list_length = max(inverted_list_lengths)
        inverted_lists_tensor = torch.full((nlist, max_list_length), -1, 
                                          dtype=torch.int, device=device)
        
        for i, lst in enumerate(inverted_lists):
            if lst:
                inverted_lists_tensor[i, :len(lst)] = torch.tensor(lst, dtype=torch.int, device=device)
        
        # Keep assignments for search
        cluster_sizes = torch.tensor(inverted_list_lengths, device=device)
        logger.info(f"Clustering completed. Min/Avg/Max cluster size: {cluster_sizes.min().item()}/{cluster_sizes.float().mean().item():.1f}/{cluster_sizes.max().item()}")
        
        cluster_done_time = time.time()
        logger.info(f"Clustering time: {cluster_done_time - start_time:.2f} seconds")
    else:
        use_ann = False
        logger.info("Using exact search (not enough vectors for efficient ANN)")
        embeddings_tensor = torch.tensor(embeddings, dtype=dtype).to(device)
    
    # Structure to hold results
    results = []
    total_neighbors = 0
    
    # Number of neighbors to find
    k = min(100, num_vectors)
    
    # Search
    search_start_time = time.time()
    
    # Batch processing for search
    if use_ann:
        # ANN search
        with tqdm(total=num_vectors, desc="ANN similarity search", unit="docs") as pbar:
            # Process in batches
            for batch_start in range(0, num_vectors, batch_size):
                batch_end = min(batch_start + batch_size, num_vectors)
                batch_size_actual = batch_end - batch_start
                batch_queries = embeddings_tensor[batch_start:batch_end]
                
                # Find nprobe closest clusters for each query
                with torch.cuda.stream(main_stream) if main_stream else nullcontext():
                    # Compute distances to centroids
                    centroid_distances = torch.matmul(batch_queries, centroids.t())
                    
                    # Get top nprobe clusters
                    _, cluster_indices = torch.topk(centroid_distances, k=min(nprobe, nlist))
                
                # Search within selected clusters
                batch_results = []
                batch_indices = []
                batch_distances = []
                
                for i in range(batch_size_actual):
                    query_idx = batch_start + i
                    query_results = []
                    
                    # Collect all candidate indices from selected clusters
                    selected_clusters = cluster_indices[i].cpu().numpy()
                    
                    # Get candidates from inverted lists
                    candidates = []
                    for cluster_idx in selected_clusters:
                        # Get vectors in this cluster
                        cluster_vectors = inverted_lists_tensor[cluster_idx]
                        valid_indices = cluster_vectors[cluster_vectors >= 0]
                        candidates.extend(valid_indices.cpu().numpy())
                    
                    if not candidates:
                        continue
                    
                    # Remove duplicates
                    candidates = list(set(candidates))
                    
                    # Compute similarities with candidates
                    candidates_tensor = torch.tensor(candidates, dtype=torch.int, device=device)
                    candidate_embeddings = embeddings_tensor[candidates_tensor]
                    
                    # Compute similarities
                    similarities = torch.matmul(batch_queries[i:i+1], candidate_embeddings.t())
                    
                    # Get top k
                    if len(candidates) > k:
                        top_similarities, top_indices = torch.topk(similarities[0], k=min(k, len(candidates)))
                        top_candidate_indices = candidates_tensor[top_indices].cpu().numpy()
                        top_similarities = top_similarities.cpu().numpy()
                    else:
                        top_candidate_indices = candidates_tensor.cpu().numpy()
                        top_similarities = similarities[0].cpu().numpy()
                    
                    # Add to results
                    query_id = ids[query_idx]
                    query_question = questions[query_idx]
                    
                    # Filter out self and apply threshold
                    neighbors = []
                    for idx, similarity in zip(top_candidate_indices, top_similarities):
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
                    
                    # Sort by similarity
                    neighbors.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Add to results
                    if neighbors:
                        results.append({
                            'from': query_id,
                            'from_question': query_question,
                            'neighbors': neighbors
                        })
                        total_neighbors += len(neighbors)
                
                # Update progress
                pbar.update(batch_size_actual)
    else:
        # Exact search using batch processing
        with tqdm(total=num_vectors, desc="Exact similarity search", unit="docs") as pbar:
            for batch_start in range(0, num_vectors, batch_size):
                batch_end = min(batch_start + batch_size, num_vectors)
                batch_size_actual = batch_end - batch_start
                
                # Extract batch queries
                with torch.cuda.stream(main_stream) if main_stream else nullcontext():
                    query_tensor = embeddings_tensor[batch_start:batch_end]
                    
                    # Compute similarities for all pairs in the batch at once
                    similarities = torch.matmul(query_tensor, embeddings_tensor.t())
                    
                    # Get top-k for each query
                    top_similarities, top_indices = torch.topk(similarities, k=min(k, num_vectors))
                
                if main_stream:
                    torch.cuda.current_stream().wait_stream(main_stream)
                
                # Process results
                for i in range(batch_size_actual):
                    query_idx = batch_start + i
                    query_id = ids[query_idx]
                    query_question = questions[query_idx]
                    
                    # Get similarities and indices for this query
                    sims = top_similarities[i].cpu().numpy()
                    inds = top_indices[i].cpu().numpy()
                    
                    # Filter results
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
                pbar.update(batch_size_actual)
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    search_time = time.time() - search_start_time
    
    logger.info(f"Search completed in {search_time:.2f} seconds ({search_time/num_vectors*1000:.2f} ms per vector)")
    logger.info(f"Total time (including preprocessing): {elapsed_time:.2f} seconds")
    logger.info(f"Found {len(results)} documents with similar pairs, {total_neighbors} total connections")
    
    return results, elapsed_time

def faiss_similarity_search(embeddings, ids, questions, threshold=0.8):
    """
    Similarity search using FAISS-CPU.
    
    Parameters:
    -----------
    embeddings: numpy.ndarray
        Normalized embedding vectors
    ids: list
        Document IDs
    questions: list
        Document questions
    threshold: float
        Similarity threshold (0.0-1.0)
        
    Returns:
    --------
    tuple (results, elapsed_time)
        results: list of similarity results
        elapsed_time: time taken for the search
    """
    # Record start time
    start_time = time.time()
    
    # Get dimensions
    num_vectors, dimension = embeddings.shape
    logger.info(f"Creating FAISS index for {num_vectors} vectors of dimension {dimension}")
    
    # Create a flat index for inner product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to the index
    index.add(embeddings)
    index_build_time = time.time() - start_time
    logger.info(f"FAISS index built in {index_build_time:.2f} seconds")
    
    # Structure to hold results
    results = []
    total_neighbors = 0
    
    # Number of neighbors to find
    k = min(100, num_vectors)
    
    # Search for each vector
    search_start_time = time.time()
    with tqdm(total=num_vectors, desc="FAISS similarity search", unit="docs") as pbar:
        for i in range(num_vectors):
            # Query vector
            query_vector = embeddings[i:i+1]
            
            # Search for similar vectors
            similarities, indices = index.search(query_vector, k)
            
            # Process results
            neighbors = []
            for j, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
                # Skip self-match
                if idx == i:
                    continue
                    
                # Apply threshold
                if similarity >= threshold:
                    neighbors.append({
                        'id': ids[idx],
                        'similarity': float(similarity),
                        'question': questions[idx]
                    })
            
            # Sort neighbors
            neighbors.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Add to results if we found any neighbors
            if neighbors:
                results.append({
                    'from': ids[i],
                    'from_question': questions[i],
                    'neighbors': neighbors
                })
                total_neighbors += len(neighbors)
            
            # Update progress bar
            pbar.update(1)
    
    search_time = time.time() - search_start_time
    total_time = time.time() - start_time
    logger.info(f"FAISS search completed in {search_time:.2f} seconds ({search_time/num_vectors*1000:.2f} ms per vector)")
    logger.info(f"Total FAISS time (index build + search): {total_time:.2f} seconds")
    logger.info(f"Found {len(results)} documents with similar pairs, {total_neighbors} total connections")
    
    return results, total_time

def compare_methods(db_name, collection_name, embedding_field, threshold=0.8, 
                    arango_host="http://localhost:8529", 
                    username="root", password="openSesame",
                    max_documents=None, batch_size=1000,
                    run_faiss=True, run_pytorch=True, run_pytorch_opt=True, run_pytorch_enh=False):
    """
    Compare PyTorch and FAISS-CPU for similarity search.
    
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
    max_documents: int or None
        Maximum number of documents to process (None for all)
    batch_size: int
        Size of batches for processing documents from ArangoDB
    run_faiss: bool
        Whether to run FAISS-CPU test
    run_pytorch: bool
        Whether to run PyTorch (Standard) test
    run_pytorch_opt: bool
        Whether to run PyTorch (Optimized) test
    run_pytorch_enh: bool
        Whether to run PyTorch (Enhanced) test
        
    Returns:
    --------
    dict
        Comparison results
    """
    logger.info("=== Starting Performance Comparison ===")
    
    # Load data
    start_time = time.time()
    embeddings, ids, questions, dimension = load_documents_optimized(
        db_name, collection_name, embedding_field,
        arango_host, username, password,
        batch_size=batch_size, max_documents=max_documents
    )
    data_load_time = time.time() - start_time
    
    if embeddings is None:
        logger.error("Failed to load data. Aborting comparison.")
        return None
    
    logger.info(f"Data loaded in {data_load_time:.2f} seconds")
    logger.info(f"Dataset size: {len(embeddings)} vectors of dimension {dimension}")
    
    # Initialize results
    results = {
        "dataset_size": len(embeddings),
        "dimension": dimension,
        "data_load_time": data_load_time
    }
    
    # Run FAISS-CPU
    faiss_results, faiss_time, faiss_count = None, 0, 0
    if run_faiss:
        logger.info("Running FAISS-CPU similarity search...")
        faiss_results, faiss_time = faiss_similarity_search(embeddings, ids, questions, threshold)
        faiss_count = len(faiss_results)
        faiss_vps = len(embeddings) / faiss_time if faiss_time > 0 else 0
        
        results["faiss_cpu"] = {
            "time": faiss_time,
            "results_count": faiss_count,
            "vectors_per_second": faiss_vps
        }
    
    # Run PyTorch (Standard)
    pytorch_results, pytorch_time, pytorch_count = None, 0, 0
    if run_pytorch:
        logger.info("Running PyTorch (Standard) similarity search...")
        pytorch_results, pytorch_time = pytorch_similarity_search(
            embeddings, ids, questions, threshold, 
            batch_size=128, fp16=False, cuda_streams=False
        )
        pytorch_count = len(pytorch_results)
        pytorch_vps = len(embeddings) / pytorch_time if pytorch_time > 0 else 0
        
        results["pytorch_standard"] = {
            "time": pytorch_time,
            "results_count": pytorch_count,
            "vectors_per_second": pytorch_vps
        }
        
        if run_faiss and faiss_time > 0:
            results["pytorch_standard"]["speedup_vs_faiss"] = faiss_time / pytorch_time
    
    # Run PyTorch (Optimized)
    pytorch_opt_results, pytorch_opt_time, pytorch_opt_count = None, 0, 0
    if run_pytorch_opt:
        logger.info("Running PyTorch (Optimized) similarity search...")
        pytorch_opt_results, pytorch_opt_time = pytorch_similarity_search(
            embeddings, ids, questions, threshold, 
            batch_size=128, fp16=True, cuda_streams=True
        )
        pytorch_opt_count = len(pytorch_opt_results)
        pytorch_opt_vps = len(embeddings) / pytorch_opt_time if pytorch_opt_time > 0 else 0
        
        results["pytorch_optimized"] = {
            "time": pytorch_opt_time,
            "results_count": pytorch_opt_count,
            "vectors_per_second": pytorch_opt_vps
        }
        
        if run_faiss and faiss_time > 0:
            results["pytorch_optimized"]["speedup_vs_faiss"] = faiss_time / pytorch_opt_time
            
        if run_pytorch and pytorch_time > 0:
            results["pytorch_optimized"]["speedup_vs_pytorch_standard"] = pytorch_time / pytorch_opt_time
    
    # Run PyTorch (Enhanced with FAISS-like features)
    pytorch_enh_results, pytorch_enh_time, pytorch_enh_count = None, 0, 0
    if run_pytorch_enh:
        logger.info("Running PyTorch (Enhanced) similarity search...")
        pytorch_enh_results, pytorch_enh_time = pytorch_enhanced_search(
            embeddings, ids, questions, threshold, 
            batch_size=128, fp16=True, cuda_streams=True,
            use_ann=(len(embeddings) > 5000),  # Only use ANN for larger datasets
            nlist=min(4096, len(embeddings) // 39),  # FAISS rule of thumb
            nprobe=min(256, len(embeddings) // 39 // 4)  # FAISS rule of thumb
        )
        pytorch_enh_count = len(pytorch_enh_results)
        pytorch_enh_vps = len(embeddings) / pytorch_enh_time if pytorch_enh_time > 0 else 0
        
        results["pytorch_enhanced"] = {
            "time": pytorch_enh_time,
            "results_count": pytorch_enh_count,
            "vectors_per_second": pytorch_enh_vps
        }
        
        if run_faiss and faiss_time > 0:
            results["pytorch_enhanced"]["speedup_vs_faiss"] = faiss_time / pytorch_enh_time
            
        if run_pytorch_opt and pytorch_opt_time > 0:
            results["pytorch_enhanced"]["speedup_vs_pytorch_optimized"] = pytorch_opt_time / pytorch_enh_time
    
    # Create table for display
    table_data = [["Metric"]]
    if run_faiss:
        table_data[0].append("FAISS-CPU")
    if run_pytorch:
        table_data[0].append("PyTorch (Standard)")
    if run_pytorch_opt:
        table_data[0].append("PyTorch (Optimized)")
    if run_pytorch_enh:
        table_data[0].append("PyTorch (Enhanced)")
    
    # Time row
    time_row = ["Time (seconds)"]
    if run_faiss:
        time_row.append(f"{faiss_time:.2f}")
    if run_pytorch:
        time_row.append(f"{pytorch_time:.2f}")
    if run_pytorch_opt:
        time_row.append(f"{pytorch_opt_time:.2f}")
    if run_pytorch_enh:
        time_row.append(f"{pytorch_enh_time:.2f}")
    table_data.append(time_row)
    
    # Similar pairs found row
    pairs_row = ["Similar pairs found"]
    if run_faiss:
        pairs_row.append(faiss_count)
    if run_pytorch:
        pairs_row.append(pytorch_count)
    if run_pytorch_opt:
        pairs_row.append(pytorch_opt_count)
    if run_pytorch_enh:
        pairs_row.append(pytorch_enh_count)
    table_data.append(pairs_row)
    
    # Vectors per second row
    vps_row = ["Vectors per second"]
    if run_faiss:
        vps_row.append(f"{results['faiss_cpu']['vectors_per_second']:.1f}")
    if run_pytorch:
        vps_row.append(f"{results['pytorch_standard']['vectors_per_second']:.1f}")
    if run_pytorch_opt:
        vps_row.append(f"{results['pytorch_optimized']['vectors_per_second']:.1f}")
    if run_pytorch_enh:
        vps_row.append(f"{results['pytorch_enhanced']['vectors_per_second']:.1f}")
    table_data.append(vps_row)
    
    # Speedup vs FAISS row
    if run_faiss:
        speedup_faiss_row = ["Speedup vs FAISS"]
        speedup_faiss_row.append("1.00x")
        if run_pytorch:
            speedup_faiss_row.append(f"{results['pytorch_standard']['speedup_vs_faiss']:.2f}x")
        if run_pytorch_opt:
            speedup_faiss_row.append(f"{results['pytorch_optimized']['speedup_vs_faiss']:.2f}x")
        if run_pytorch_enh:
            speedup_faiss_row.append(f"{results['pytorch_enhanced']['speedup_vs_faiss']:.2f}x")
        table_data.append(speedup_faiss_row)
    
    # Speedup vs PyTorch (Standard) row
    if run_pytorch:
        speedup_pytorch_row = ["Speedup vs PyTorch (Standard)"]
        if run_faiss:
            speedup_pytorch_row.append("-")
        speedup_pytorch_row.append("1.00x")
        if run_pytorch_opt:
            speedup_pytorch_row.append(f"{results['pytorch_optimized']['speedup_vs_pytorch_standard']:.2f}x")
        if run_pytorch_enh:
            speedup = pytorch_time / pytorch_enh_time if pytorch_time > 0 and pytorch_enh_time > 0 else 0
            speedup_pytorch_row.append(f"{speedup:.2f}x")
        table_data.append(speedup_pytorch_row)
    
    # Speedup vs PyTorch (Optimized) row
    if run_pytorch_opt and run_pytorch_enh:
        speedup_pytorch_opt_row = ["Speedup vs PyTorch (Optimized)"]
        if run_faiss:
            speedup_pytorch_opt_row.append("-")
        if run_pytorch:
            speedup_pytorch_opt_row.append("-")
        speedup_pytorch_opt_row.append("1.00x")
        speedup = pytorch_opt_time / pytorch_enh_time if pytorch_opt_time > 0 and pytorch_enh_time > 0 else 0
        speedup_pytorch_opt_row.append(f"{speedup:.2f}x")
        table_data.append(speedup_pytorch_opt_row)
    
    # Display table
    table = tabulate(table_data, headers="firstrow", tablefmt="grid")
    print("\n=== Performance Comparison Results ===")
    print(f"Dataset: {len(embeddings)} vectors of dimension {dimension}")
    print(f"Data loading time: {data_load_time:.2f} seconds")
    print(table)
    
    # Report results in log
    logger.info("=== Performance Comparison Results ===")
    logger.info(f"Dataset: {len(embeddings)} vectors of dimension {dimension}")
    logger.info(f"Data loading time: {data_load_time:.2f} seconds")
    
    if run_faiss:
        logger.info(f"FAISS-CPU time: {faiss_time:.2f} seconds, found {faiss_count} similar pairs")
    if run_pytorch:
        logger.info(f"PyTorch (Standard) time: {pytorch_time:.2f} seconds, found {pytorch_count} similar pairs")
    if run_pytorch_opt:
        logger.info(f"PyTorch (Optimized) time: {pytorch_opt_time:.2f} seconds, found {pytorch_opt_count} similar pairs")
    if run_pytorch_enh:
        logger.info(f"PyTorch (Enhanced) time: {pytorch_enh_time:.2f} seconds, found {pytorch_enh_count} similar pairs")
    
    if run_faiss and run_pytorch:
        logger.info(f"PyTorch (Standard) speedup vs FAISS: {results['pytorch_standard']['speedup_vs_faiss']:.2f}x")
    if run_faiss and run_pytorch_opt:
        logger.info(f"PyTorch (Optimized) speedup vs FAISS: {results['pytorch_optimized']['speedup_vs_faiss']:.2f}x")
    if run_faiss and run_pytorch_enh:
        logger.info(f"PyTorch (Enhanced) speedup vs FAISS: {results['pytorch_enhanced']['speedup_vs_faiss']:.2f}x")
    
    # Save detailed comparison to file
    try:
        comparison_file = 'performance_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed comparison saved to {comparison_file}")
    except Exception as e:
        logger.error(f"Failed to save comparison results to file: {str(e)}")
    
    return results

# Add nullcontext for Python < 3.7
class nullcontext:
    def __enter__(self):
        return None
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

@app.command()
def run(
    db_name: str = typer.Option("memory_bank", help="ArangoDB database name"),
    collection_name: str = typer.Option("complexity", help="ArangoDB collection name"),
    embedding_field: str = typer.Option("embedding", help="Field containing embeddings"),
    threshold: float = typer.Option(0.8, help="Similarity threshold (0.0-1.0)"),
    arango_host: str = typer.Option("http://localhost:8529", help="ArangoDB host URL"),
    username: str = typer.Option("root", help="ArangoDB username"),
    password: str = typer.Option("openSesame", help="ArangoDB password"),
    max_docs: Optional[int] = typer.Option(None, help="Maximum number of documents to process (default: all)"),
    batch_size: int = typer.Option(1000, help="Batch size for retrieving documents from ArangoDB"),
    methods: list[str] = typer.Option(
        ["faiss", "pytorch", "pytorch-opt"], 
        help="Methods to run: faiss, pytorch, pytorch-opt, pytorch-enh (default: all except pytorch-enh)"
    )
):
    """Compare FAISS-CPU and PyTorch for similarity search"""
    try:
        logger.info("Starting similarity search performance comparison")
        
        # Determine which methods to run
        run_faiss = "faiss" in methods
        run_pytorch = "pytorch" in methods
        run_pytorch_opt = "pytorch-opt" in methods
        run_pytorch_enh = "pytorch-enh" in methods
        
        # Run the comparison
        compare_methods(
            db_name=db_name,
            collection_name=collection_name,
            embedding_field=embedding_field,
            threshold=threshold,
            arango_host=arango_host,
            username=username, 
            password=password,
            max_documents=max_docs,
            batch_size=batch_size,
            run_faiss=run_faiss,
            run_pytorch=run_pytorch,
            run_pytorch_opt=run_pytorch_opt,
            run_pytorch_enh=run_pytorch_enh
        )
        
        logger.info("Performance comparison completed successfully")
        
    except Exception as e:
        logger.error(f"Process failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    app()