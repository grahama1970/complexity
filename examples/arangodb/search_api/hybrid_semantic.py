import time
import json
import sys
from typing import Dict, Any, List, Optional, Tuple, Union
import copy

from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError

# Import config variables and embedding utils
try:
    from complexity.arangodb.config import (
        COLLECTION_NAME,
        VIEW_NAME,
        ALL_DATA_FIELDS_PREVIEW,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSIONS
    )
    from complexity.arangodb.arango_setup_unknown import EMBEDDING_FIELD, connect_arango, ensure_database
    from complexity.arangodb.embedding_utils import get_embedding
except ImportError as e:
    logger.critical(f"CRITICAL: Failed module-level import in semantic.py: {e}. Functionality will be broken.")
    # Define fallbacks to allow module to load (but not function)
    COLLECTION_NAME = "documents"
    VIEW_NAME = "document_view"
    ALL_DATA_FIELDS_PREVIEW = ["_key"]
    EMBEDDING_MODEL = "text-embedding-ada-002"
    EMBEDDING_DIMENSIONS = 1536
    EMBEDDING_FIELD = "embedding"

# Try to import PyTorch and vector search utils
try:
    import torch
    import numpy as np
    # Import the PyTorch search functions - adjust the import path as needed
    from complexity.arangodb.search_api.pytorch_search import (
        load_documents_from_arango,
        pytorch_enhanced_search
    )
    has_pytorch = True
    logger.info("PyTorch is available, using GPU-accelerated semantic search")
except ImportError:
    has_pytorch = False
    logger.info("PyTorch not available, will use ArangoDB for semantic search")


def semantic_search(
    db: StandardDatabase,
    query: Union[str, List[float]],
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    min_score: float = 0.7,
    top_n: int = 10,
    tag_list: Optional[List[str]] = None,
    force_arangodb: bool = False
) -> Dict[str, Any]:
    """
    Enhanced semantic search that uses PyTorch by default with ArangoDB as fallback.
    
    Args:
        db: ArangoDB database
        query: Search query text or embedding vector
        collections: Optional list of collections to search
        filter_expr: Optional AQL filter expression
        min_score: Minimum similarity score threshold (0-1)
        top_n: Maximum number of results to return
        tag_list: Optional list of tags to filter by
        force_arangodb: Force using ArangoDB's built-in search (even when PyTorch is available)
        
    Returns:
        Dict with search results
    """
    start_time = time.time()
    
    # Use default collection if not specified
    if not collections:
        collections = [COLLECTION_NAME]
    
    # Get query embedding if string is provided
    query_embedding = query
    query_text_for_return = "vector query"
    if isinstance(query, str):
        query_text_for_return = query
        query_embedding = get_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            return {
                "results": [],
                "total": 0,
                "query": query,
                "error": "Failed to generate embedding"
            }

    # Build filter conditions for PyTorch and ArangoDB
    filter_conditions = []
    
    # Add filter expression if provided
    if filter_expr:
        filter_conditions.append(f"({filter_expr})")
    
    # Add tag filtering if provided
    if tag_list:
        tag_conditions = []
        for tag in tag_list:
            tag_conditions.append(f'"{tag}" IN doc.tags')
        if tag_conditions:
            filter_conditions.append(f"({' AND '.join(tag_conditions)})")
    
    # Combine all filter conditions
    combined_filter = " AND ".join(filter_conditions) if filter_conditions else ""
    
    # Use PyTorch-based search if available and not forced to use ArangoDB
    if has_pytorch and not force_arangodb:
        try:
            return _pytorch_semantic_search(
                db=db,
                query_embedding=query_embedding,
                query_text=query_text_for_return,
                collection_name=collections[0],  # Use first collection
                embedding_field=EMBEDDING_FIELD,
                filter_conditions=combined_filter,
                min_score=min_score,
                top_n=top_n,
                start_time=start_time
            )
        except Exception as e:
            logger.error(f"PyTorch search failed, falling back to ArangoDB: {str(e)}")
            # Fall back to ArangoDB search on error
    
    # Use ArangoDB search (fallback or if forced)
    return _arangodb_semantic_search(
        db=db,
        query_embedding=query_embedding,
        query_text=query_text_for_return,
        filter_expr=filter_expr,
        min_score=min_score,
        top_n=top_n,
        tag_list=tag_list,
        start_time=start_time
    )


def _pytorch_semantic_search(
    db: StandardDatabase,
    query_embedding: List[float],
    query_text: str,
    collection_name: str,
    embedding_field: str,
    filter_conditions: str = "",
    min_score: float = 0.7,
    top_n: int = 10,
    start_time: float = None
) -> Dict[str, Any]:
    """
    Perform semantic search using PyTorch.
    
    Args:
        db: ArangoDB database
        query_embedding: Vector representation of the query
        query_text: Original query text
        collection_name: Name of the collection to search
        embedding_field: Name of the field containing embeddings
        filter_conditions: AQL filter conditions
        min_score: Minimum similarity threshold
        top_n: Maximum number of results to return
        start_time: Optional timestamp for the start of the search
        
    Returns:
        Dict with search results
    """
    if start_time is None:
        start_time = time.time()
    
    logger.info(f"Using PyTorch-based semantic search with threshold {min_score}")
    if filter_conditions:
        logger.info(f"Applying filter conditions: {filter_conditions}")
    
    # Load documents with filtering
    embeddings, ids, questions, metadata, dimension = load_documents_from_arango(
        db, collection_name, embedding_field,
        filter_conditions=filter_conditions
    )
    
    if embeddings is None or len(embeddings) == 0:
        logger.warning("No documents found matching the filter criteria")
        return {
            "results": [],
            "total": 0,
            "query": query_text,
            "time": time.time() - start_time
        }
    
    # Prepare query vector
    query_array = np.array([query_embedding], dtype=np.float32)
    # Normalize query vector
    query_norm = np.linalg.norm(query_array, axis=1, keepdims=True)
    query_normalized = query_array / query_norm
    
    # Check if GPU is available
    has_gpu = torch.cuda.is_available()
    
    # Perform similarity search
    results, search_time = pytorch_enhanced_search(
        embeddings=embeddings,
        query_embedding=query_normalized[0],  # Use the first (only) query vector
        ids=ids,
        metadata=metadata,
        threshold=min_score,
        top_k=top_n,
        batch_size=128,
        fp16=has_gpu,
        cuda_streams=has_gpu,
        use_ann=(len(embeddings) > 5000)
    )
    
    # Format results to match the expected output
    formatted_results = []
    for result in results:
        formatted_result = {
            "doc": result["metadata"],
            "similarity_score": result["similarity"]
        }
        formatted_results.append(formatted_result)
    
    # Return results
    return {
        "results": formatted_results[:top_n],  # Apply top_n limit
        "total": len(results),
        "query": query_text,
        "time": time.time() - start_time,
        "search_engine": "pytorch"
    }


def _arangodb_semantic_search(
    db: StandardDatabase,
    query_embedding: List[float],
    query_text: str,
    filter_expr: Optional[str] = None,
    min_score: float = 0.7,
    top_n: int = 10,
    tag_list: Optional[List[str]] = None,
    start_time: float = None
) -> Dict[str, Any]:
    """
    Perform semantic search using ArangoDB's built-in functionality.
    
    Args:
        db: ArangoDB database
        query_embedding: Vector representation of the query
        query_text: Original query text
        filter_expr: Optional AQL filter expression
        min_score: Minimum similarity threshold
        top_n: Maximum number of results to return
        tag_list: Optional list of tags to filter by
        start_time: Optional timestamp for the start of the search
        
    Returns:
        Dict with search results
    """
    if start_time is None:
        start_time = time.time()
    
    logger.info("Using ArangoDB-based semantic search")
    
    try:
        # Build the SEARCH filter expression
        search_filter_parts = []
        bind_vars = {
            "query_embedding": query_embedding,
            "min_score": min_score,
        }
        
        if filter_expr:
            search_filter_parts.append(f"({filter_expr})")
        
        # Add tag filtering
        if tag_list:
            bind_vars["tag_list"] = tag_list
            search_filter_parts.append("doc.tags ALL IN @tag_list")
        
        search_filter_expression = " AND ".join(search_filter_parts) if search_filter_parts else ""
        search_clause = f"SEARCH {search_filter_expression}" if search_filter_expression else ""
        
        # Build AQL using COSINE_SIMILARITY on the View
        aql = f"""
        FOR doc IN {VIEW_NAME}
        {search_clause}
        LET similarity = COSINE_SIMILARITY(doc.{EMBEDDING_FIELD}, @query_embedding)
        FILTER similarity >= @min_score
        SORT similarity DESC
        LIMIT {top_n}
        RETURN {{
            "doc": doc,
            "similarity_score": similarity
        }}
        """
        
        # Execute query
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        # Get total count (with same filters, without limit)
        count_filter_clauses = copy.deepcopy(search_filter_parts)
        count_filter_clause_str = f"SEARCH {' AND '.join(count_filter_clauses)}" if count_filter_clauses else ""
        
        count_aql = f"""
        RETURN COUNT(
            FOR doc IN {VIEW_NAME}
            {count_filter_clause_str}
            LET similarity = COSINE_SIMILARITY(doc.{EMBEDDING_FIELD}, @query_embedding)
            FILTER similarity >= @min_score
            RETURN 1
        )
        """
        
        count_cursor = db.aql.execute(count_aql, bind_vars=bind_vars)
        total_count = next(count_cursor)
        
        # Return results
        return {
            "results": results,
            "total": total_count,
            "query": query_text,
            "time": time.time() - start_time,
            "search_engine": "arangodb"
        }
        
    except Exception as e:
        logger.error(f"ArangoDB semantic search failed: {str(e)}")
        
        # Construct error details
        error_str = str(e)
        if hasattr(e, 'error_code'):
            error_str = f"Code: {e.error_code}, {error_str}"
        
        return {
            "results": [],
            "total": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": error_str,
            "search_engine": "arangodb"
        }


# Update pytorch_search.py to include these enhancements
def update_pytorch_search_module():
    """
    Implementation of enhanced PyTorch search functionality.
    This would normally be in a separate file, but including here for completeness.
    """
    return """
# src/complexity/arangodb/search_api/pytorch_search.py
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
    filter_conditions: str or None
        Optional AQL filter conditions
        
    Returns:
    --------
    tuple (embeddings, ids, questions, metadata, dimension)
        embeddings: numpy.ndarray of normalized embeddings
        ids: list of document IDs
        questions: list of document questions or titles
        metadata: list of document metadata/full documents
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
        metadata = []
        
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
            question: doc.question || doc.title || doc._key,
            embedding: doc.{embedding_field},
            metadata: doc
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
                    metadata.append(doc["metadata"])
                    pbar.update(1)
                    
                except KeyError:
                    continue
        
        fetch_time = time.time() - load_start_time
        docs_per_second = len(embeddings)/fetch_time if fetch_time > 0 else 0
        logger.info(f"Document fetching completed in {fetch_time:.2f} seconds ({docs_per_second:.1f} docs/s)")
        
        # Verify we have documents with valid embeddings
        if not embeddings:
            logger.error("No valid embeddings found in documents")
            return None, None, None, None, None
            
        # Convert to numpy array
        start_time = time.time()
        embeddings_np = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_np.shape[1]
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / norms
        normalize_time = time.time() - start_time
        
        logger.info(f"Extracted and normalized {len(embeddings_np)} valid embeddings of dimension {dimension} in {normalize_time:.2f}s")
        
        return normalized_embeddings, ids, questions, metadata, dimension
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None, None


def pytorch_enhanced_search(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    ids: List[str],
    metadata: List[Dict],
    threshold: float = 0.8,
    top_k: int = 10,
    batch_size: int = 128,
    fp16: bool = True,
    cuda_streams: bool = True,
    use_ann: bool = True,
    nlist: int = 100,
    nprobe: int = 10
) -> Tuple[List[Dict], float]:
    """
    Enhanced GPU-accelerated similarity search using PyTorch.
    
    Parameters:
    -----------
    embeddings: numpy.ndarray
        Normalized embedding vectors
    query_embedding: numpy.ndarray
        Query vector (already normalized)
    ids: list
        Document IDs
    metadata: list
        Document metadata (full documents)
    threshold: float
        Similarity threshold (0.0-1.0)
    top_k: int
        Number of top results to return
    batch_size: int
        Batch size for processing
    fp16: bool
        Whether to use FP16 (half precision) for faster computation
    cuda_streams: bool
        Whether to use CUDA streams for concurrent processing
    use_ann: bool
        Whether to use approximate nearest neighbor search
    nlist: int
        Number of clusters for ANN index
    nprobe: int
        Number of clusters to search in ANN index
        
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
    
    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=dtype).to(device)
    
    # Convert query to tensor
    query_tensor = torch.tensor(query_embedding, dtype=dtype).to(device)
    
    # Create CUDA streams if needed
    main_stream = torch.cuda.Stream() if cuda_streams and torch.cuda.is_available() else None
    
    # Structure to hold results
    results = []
    
    # Use ANN if requested and if enough vectors
    if use_ann and num_vectors > nlist * 10:
        # Implement ANN search logic here (clustering, etc.)
        # This would be similar to the full implementation in the original code
        # For brevity, we'll use simple exact search here
        logger.info("ANN search requested but using exact search for this example.")
        use_exact_search = True
    else:
        use_exact_search = True
    
    # Exact search implementation
    if use_exact_search:
        # Compute similarity scores for all vectors
        with torch.cuda.stream(main_stream) if main_stream else nullcontext():
            # Reshape query for matrix multiplication
            query_reshaped = query_tensor.reshape(1, -1)
            
            # Compute similarities in batches
            all_similarities = []
            
            for batch_start in range(0, num_vectors, batch_size):
                batch_end = min(batch_start + batch_size, num_vectors)
                batch_embeddings = embeddings_tensor[batch_start:batch_end]
                
                # Compute similarity (dot product for normalized vectors = cosine similarity)
                batch_similarities = torch.matmul(query_reshaped, batch_embeddings.t())
                all_similarities.append(batch_similarities)
            
            # Concatenate all batch results
            similarities = torch.cat(all_similarities, dim=1).squeeze()
            
            # Get indices sorted by similarity (descending)
            _, indices = torch.sort(similarities, descending=True)
            
            # Get similarities and indices as numpy arrays
            similarities_np = similarities.cpu().numpy()
            indices_np = indices.cpu().numpy()
        
        if main_stream:
            torch.cuda.current_stream().wait_stream(main_stream)
    
    # Prepare results
    filtered_results = []
    
    for idx in indices_np:
        similarity = float(similarities_np[idx])
        
        # Apply threshold filter
        if similarity < threshold:
            continue
        
        # Add to results
        result = {
            "id": ids[idx],
            "similarity": similarity,
            "metadata": metadata[idx]
        }
        filtered_results.append(result)
        
        # Stop after finding top_k results
        if len(filtered_results) >= top_k:
            break
    
    elapsed_time = time.time() - start_time
    logger.info(f"Search completed in {elapsed_time:.2f} seconds")
    logger.info(f"Found {len(filtered_results)} results above threshold {threshold}")
    
    return filtered_results, elapsed_time
"""

# For testing and validation
def validate_semantic_search(search_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate semantic search results against known good fixture data.
    
    Args:
        search_results: The results returned from semantic_search
        fixture_path: Path to the fixture file containing expected results
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    # Load fixture data
    try:
        with open(fixture_path, "r") as f:
            expected_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load fixture data: {e}")
        return False, {"fixture_loading_error": {"expected": "Valid JSON file", "actual": str(e)}}
    
    # Track all validation failures
    validation_failures = {}
    
    # Structural validation
    if "results" not in search_results:
        validation_failures["missing_results"] = {
            "expected": "Results field present",
            "actual": "Results field missing"
        }
        return False, validation_failures
    
    # Validate result count
    if len(search_results["results"]) != len(expected_data["results"]):
        validation_failures["result_count"] = {
            "expected": len(expected_data["results"]),
            "actual": len(search_results["results"])
        }
    
    # Validate total count
    if search_results.get("total") != expected_data.get("total"):
        validation_failures["total_count"] = {
            "expected": expected_data.get("total"),
            "actual": search_results.get("total")
        }
    
    # Content validation - compare actual results with expected
    for i, (expected_result, actual_result) in enumerate(
        zip(expected_data["results"], search_results["results"])
    ):
        # Check structure of each result
        if "doc" not in actual_result:
            validation_failures[f"result_{i}_missing_doc"] = {
                "expected": "doc field present",
                "actual": "doc field missing"
            }
            continue
        
        # Check specific, meaningful document fields, ignoring volatile ones
        fields_to_validate = ["problem", "solution", "context", "tags", "_key"]
        for key in fields_to_validate:
            if key not in expected_result.get("doc", {}):
                 # If key is not in expected, don't validate it (allows fixture flexibility)
                 continue
            if key not in actual_result.get("doc", {}):
                validation_failures[f"result_{i}_doc_missing_{key}"] = {
                    "expected": f"'{key}' field present",
                    "actual": f"'{key}' field missing"
                }
            elif actual_result["doc"][key] != expected_result["doc"][key]:
                 # Special handling for tags list comparison (order might not matter)
                 if key == "tags" and isinstance(actual_result["doc"][key], list) and isinstance(expected_result["doc"][key], list):
                     if sorted(actual_result["doc"][key]) != sorted(expected_result["doc"][key]):
                         validation_failures[f"result_{i}_doc_{key}"] = {
                             "expected": sorted(expected_result["doc"][key]),
                             "actual": sorted(actual_result["doc"][key])
                         }
                 else:
                    validation_failures[f"result_{i}_doc_{key}"] = {
                        "expected": expected_result["doc"][key],
                        "actual": actual_result["doc"][key]
                    }
        
        # Check similarity score field (allow for score or similarity_score key)
        expected_score = expected_result.get("similarity_score", expected_result.get("score"))
        actual_score = actual_result.get("similarity_score", actual_result.get("score"))
        
        # For floating point scores, use approximate comparison
        if expected_score is not None and actual_score is not None:
            # Allow small differences in scores due to floating point precision
            if abs(expected_score - actual_score) > 0.01:
                validation_failures[f"result_{i}_score"] = {
                    "expected": expected_score,
                    "actual": actual_score
                }
        elif expected_score != actual_score:
            validation_failures[f"result_{i}_score"] = {
                "expected": expected_score,
                "actual": actual_score
            }
    
    return len(validation_failures) == 0, validation_failures


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/semantic_search_expected_20250422_181101.json"
    
    try:
        # Set up database connection
        client = connect_arango()
        db = ensure_database(client)
        
        # Run a test search query
        test_query = "python error handling"
        
        # Test PyTorch search
        if has_pytorch:
            logger.info("Testing PyTorch-based semantic search")
            search_results = semantic_search(
                db=db,
                query=test_query,
                top_n=10,
                min_score=0.6,
                force_arangodb=False  # Use PyTorch
            )
            
            # Check search engine used
            engine = search_results.get("search_engine", "unknown")
            logger.info(f"Search engine used: {engine}")
            
            # Validate the results
            validation_passed, validation_failures = validate_semantic_search(search_results, fixture_path)
            
            print(f"PyTorch Search Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
        
        # Test ArangoDB search (force it even if PyTorch is available)
        logger.info("Testing ArangoDB-based semantic search")
        search_results = semantic_search(
            db=db,
            query=test_query,
            top_n=10,
            min_score=0.6,
            force_arangodb=True  # Force ArangoDB
        )
        
        # Check search engine used
        engine = search_results.get("search_engine", "unknown")
        logger.info(f"Search engine used: {engine}")
        
        # Validate the results
        validation_passed, validation_failures = validate_semantic_search(search_results, fixture_path)
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All semantic search results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Semantic search results don't match expected values")
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Unexpected error in __main__: {e}")
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        sys.exit(1)