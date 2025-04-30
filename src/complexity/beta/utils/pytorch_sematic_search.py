# src/complexity/beta/utils/pytorch_semantic_search.py
"""
Module Description:
Provides optimized GPU-accelerated similarity search functionality using PyTorch.
Contains utility functions to load documents from ArangoDB and perform efficient
semantic similarity search using a FAISS-like approach with PyTorch.

Links:
- PyTorch: https://pytorch.org/
- ArangoDB: https://www.arangodb.com/docs/stable/

Sample Input/Output:

- load_documents_from_arango(db, "complexity", "embedding"):
  - Input: Database connection, collection name, embedding field
  - Output: Tuple (embeddings, ids, questions, dimension)

- pytorch_enhanced_search(embeddings, ids, questions, threshold=0.8):
  - Input: Embeddings array, document IDs, document questions, threshold
  - Output: Tuple (results, elapsed_time)
"""

import time
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import List, Dict, Any, Tuple, Optional, Union
from arango.database import StandardDatabase


class nullcontext:
    """Context manager that does nothing, for Python < 3.7 compatibility."""
    def __enter__(self):
        return None
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def load_documents_from_arango(db: StandardDatabase, collection_name: str, 
                              embedding_field: str, batch_size: int = 1000, 
                              max_documents: Optional[int] = None,
                              filter_conditions: Optional[str] = None) -> Tuple:
    """
    Load documents with embeddings from ArangoDB.
    
    Parameters:
    -----------
    db: StandardDatabase
        ArangoDB database connection
    collection_name: str
        Name of the collection containing documents with embeddings
    embedding_field: str
        Name of the field containing vector embeddings
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
        # Get collection count
        count = db.collection(collection_name).count()
        logger.info(f"Collection contains {count} documents")
        
        # Limit number of documents if specified
        if max_documents is not None and max_documents > 0:
            logger.info(f"Will process at most {max_documents} documents (out of {count})")
            expected_docs = min(count, max_documents)
        else:
            expected_docs = count
            
        # Extract embeddings and metadata
        embeddings = []
        ids = []
        questions = []
        
        load_start_time = time.time()
        
        # Optimization: Use a more efficient query that pre-filters documents
        limit_clause = f"LIMIT {max_documents}" if max_documents is not None else ""
        filter_clause = f"FILTER HAS(doc, \"{embedding_field}\")"
        
        # Add additional filter conditions if provided
        if filter_conditions:
            filter_clause += f" AND {filter_conditions}"
        
        aql = f"""
        FOR doc IN {collection_name}
        {filter_clause}
        {limit_clause}
        RETURN {{
            _id: doc._id,
            question: doc.question,
            embedding: doc.{embedding_field},
            metadata: doc.metadata
        }}
        """
        
        # Execute the query with batch size
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


def pytorch_enhanced_search(embeddings: np.ndarray, ids: List[str], 
                           questions: List[str], threshold: float = 0.8, 
                           batch_size: int = 128, fp16: bool = True, 
                           cuda_streams: bool = True, use_ann: bool = True, 
                           nlist: int = 100, nprobe: int = 10) -> Tuple[List[Dict], float]:
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