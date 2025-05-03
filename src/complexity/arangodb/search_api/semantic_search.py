"""
# Hybrid Semantic Search Module

This module provides optimized vector similarity search capabilities using ArangoDB,
with multiple search strategies (direct, filtered, and PyTorch-based) for different use cases.

## Third-Party Packages:
- python-arango: https://python-driver.arangodb.com/3.10.0/index.html (v3.10.0)
- tenacity: https://tenacity.readthedocs.io/en/latest/ (v8.2.3)
- loguru: https://github.com/Delgan/loguru (v0.7.2)
- torch: https://pytorch.org/docs/stable/index.html (v2.1.0)

## Sample Input:
Query text or embedding vector with optional filters:
```python
query = "primary color"
filter_expr = "doc.label == 1"
tag_list = ["color", "science"]
```

## Expected Output:
```json
{
  "results": [
    {
      "doc": {"_id": "complexity/12345", "_key": "12345", "question": "What are primary colors?", ...},
      "similarity_score": 0.92
    },
    ...
  ],
  "total": 150,
  "query": "primary color",
  "time": 0.125,
  "search_engine": "arangodb-direct"
}
```
"""

import sys
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple

from loguru import logger
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError
from colorama import init, Fore, Style

# Import config variables and embedding utils
from complexity.arangodb.config import (
    COLLECTION_NAME,
    VIEW_NAME,
    ALL_DATA_FIELDS_PREVIEW,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS, 
    EMBEDDING_FIELD
)
from complexity.arangodb.arango_setup import connect_arango, ensure_database
from complexity.arangodb.embedding_utils import get_embedding
from complexity.arangodb.log_utils import truncate_large_value
from complexity.arangodb.display_utils import print_search_results as display_results


# Try to import PyTorch search utilities
try:
    from complexity.arangodb.pytorch_search_utils import (
        has_pytorch_available,
        pytorch_semantic_search
    )
    has_pytorch = has_pytorch_available()
    logger.debug(f"PyTorch availability: {has_pytorch}")
except ImportError:
    has_pytorch = False
    logger.debug("PyTorch search utilities not available")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def execute_aql_with_retry(
    db: StandardDatabase, 
    query: str, 
    bind_vars: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Execute AQL query with retry logic for transient errors.
    
    Args:
        db: ArangoDB database
        query: AQL query string
        bind_vars: Query bind variables
        
    Returns:
        Query cursor
        
    Raises:
        Exception: Re-raises the last exception after retries are exhausted
    """
    try:
        return db.aql.execute(query, bind_vars=bind_vars)
    except (AQLQueryExecuteError, ArangoServerError) as e:
        # Log the error before retry
        logger.warning(f"ArangoDB query failed, retrying: {str(e)}")
        # Re-raise to let @retry handle it
        raise


@lru_cache(maxsize=100)
def get_cached_vector_results(
    db: StandardDatabase,
    collection_name: str, 
    embedding_field: str, 
    query_embedding_tuple: Tuple[float, ...],
    limit: int
) -> List[Dict[str, Any]]:
    """
    Get vector search results with caching.
    
    This function caches vector search results to avoid redundant queries
    for the same embedding vector, improving performance for repeated searches.
    
    Args:
        db: ArangoDB database
        collection_name: Name of the collection to search
        embedding_field: Field containing embedding vectors
        query_embedding_tuple: Query embedding as a tuple (for hashability)
        limit: Number of results to return
        
    Returns:
        List of vector search results
    """
    # Convert tuple back to list for ArangoDB query
    query_embedding = list(query_embedding_tuple)
    
    vector_query = f"""
    FOR doc IN {collection_name}
    LET score = APPROX_NEAR_COSINE(doc.{embedding_field}, @query_embedding)
    SORT score DESC
    LIMIT {limit}
    RETURN {{
        "id": doc._id,
        "similarity_score": score
    }}
    """
    
    logger.debug(f"Vector search query (cached): {vector_query}")
    cursor = execute_aql_with_retry(db, vector_query, bind_vars={"query_embedding": query_embedding})
    return list(cursor)


def semantic_search(
    db: StandardDatabase,
    query: Union[str, List[float]],
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    min_score: float = 0.7,
    top_n: int = 10,
    tag_list: Optional[List[str]] = None,
    force_pytorch: bool = False,
    force_direct: bool = False,
    force_twostage: bool = False,
    output_format: str = "table"  # Added this parameter
) -> Dict[str, Any]:

    """
    Optimized semantic search that uses the appropriate approach based on query type.
    
    Args:
        db: ArangoDB database
        query: Search query text or embedding vector
        collections: Optional list of collections to search
        filter_expr: Optional AQL filter expression
        min_score: Minimum similarity score threshold (0-1)
        top_n: Maximum number of results to return
        tag_list: Optional list of tags to filter by
        force_pytorch: Force using PyTorch (needed for relationship building)
        force_direct: Force using direct ArangoDB approach (for testing)
        force_twostage: Force using two-stage ArangoDB approach (for testing)
        
    Returns:
        Dict with search results
    """
    start_time = time.time()
    
    # Use default collection if not specified
    if not collections:
        collections = [COLLECTION_NAME]
    
    logger.info(f"Searching in collections: {collections}")
    collection_name = collections[0]  # Use first collection
    
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

    # Check if this is a nesting-type query or if PyTorch is forced
    if force_pytorch and has_pytorch:
        logger.info("Using PyTorch approach (required for nesting/relationship building)")
        try:
            # Build combined filter for PyTorch
            filter_parts = []
            if filter_expr:
                filter_parts.append(f"({filter_expr})")
            if tag_list:
                tag_conditions = []
                for tag in tag_list:
                    tag_conditions.append(f'"{tag}" IN doc.tags')
                if tag_conditions:
                    filter_parts.append(f"({' AND '.join(tag_conditions)})")
            combined_filter = " AND ".join(filter_parts) if filter_parts else ""
            
            # Import and call PyTorch search
            return pytorch_semantic_search(
                db=db,
                query_embedding=query_embedding,
                query_text=query_text_for_return,
                collection_name=collection_name,
                embedding_field=EMBEDDING_FIELD,
                filter_conditions=combined_filter,
                min_score=min_score,
                top_n=top_n,
                start_time=start_time
            )
        except Exception as e:
            logger.error(f"PyTorch approach failed: {str(e)}")
            return {
                "results": [],
                "total": 0,
                "query": query_text_for_return,
                "time": time.time() - start_time,
                "error": str(e),
                "search_engine": "pytorch-failed"
            }
    
    # Check if filtering is required
    needs_filtering = filter_expr is not None or tag_list is not None
    
    # Handle based on whether filtering is needed or forced
    if force_twostage or (needs_filtering and not force_direct):
        logger.info("Filtering required - using two-stage ArangoDB approach")
        return _filtered_semantic_search(
            db=db,
            query_embedding=query_embedding,
            query_text=query_text_for_return,
            collection_name=collection_name,
            embedding_field=EMBEDDING_FIELD,
            filter_expr=filter_expr,
            tag_list=tag_list,
            min_score=min_score,
            top_n=top_n,
            start_time=start_time,
            output_format=output_format
        )
    else:
        logger.info("No filtering required - using direct ArangoDB approach")
        return _direct_semantic_search(
            db=db,
            query_embedding=query_embedding,
            query_text=query_text_for_return,
            collection_name=collection_name,
            embedding_field=EMBEDDING_FIELD,
            min_score=min_score,
            top_n=top_n,
            start_time=start_time,
            output_format=output_format
        )


def _direct_semantic_search(
    db: StandardDatabase,
    query_embedding: List[float],
    query_text: str,
    collection_name: str,
    embedding_field: str,
    min_score: float = 0.7,
    top_n: int = 10,
    start_time: float = None,
    output_format: str = "table"
) -> Dict[str, Any]:
    """
    Direct semantic search without filtering using ArangoDB.
    
    Args:
        db: ArangoDB database
        query_embedding: Query embedding vector
        query_text: Original query text
        collection_name: Collection to search
        embedding_field: Field containing embeddings
        min_score: Minimum similarity threshold
        top_n: Maximum results to return
        start_time: Start time for timing
        output_format: Output format (table or json)
        
    Returns:
        Search results dictionary
    """
    if start_time is None:
        start_time = time.time()
    
    try:
        # Use cached vector search results
        query_embedding_tuple = tuple(query_embedding)
        
        vector_results = get_cached_vector_results(
            db,
            collection_name,
            embedding_field,
            query_embedding_tuple,
            top_n * 2  # Get extra results to ensure enough after filtering by score
        )
        
        # Filter results by minimum score
        results = []
        for result in vector_results:
            # Skip if below threshold
            if result["similarity_score"] < min_score:
                continue
                
            # Get the full document for each result
            doc_id = result["id"]
            
            # Use retry for document fetching - FIX THE CURSOR HANDLING HERE
            try:
                cursor = execute_aql_with_retry(
                    db,
                    f"RETURN DOCUMENT('{doc_id}')"
                )
                # Convert cursor to list and get the first item
                try:
                    doc_list = list(cursor)
                    # Handle various possible return types
                    if isinstance(doc_list, list) and doc_list and len(doc_list) > 0:
                        doc = doc_list[0]
                    else:
                        logger.warning(f"Document {doc_id} not found (empty result)")
                        continue
                except Exception as e:
                    logger.warning(f"Error processing document {doc_id}: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to fetch document {doc_id}: {e}")
                continue
                
            # Add to results
            results.append({
                "doc": doc,
                "similarity_score": result["similarity_score"]
            })
                
            # Stop once we have enough results
            if len(results) >= top_n:
                break
        
        # Get total count of documents in collection for context
        count_cursor = execute_aql_with_retry(
            db,
            f"RETURN LENGTH({collection_name})"
        )
        total_list = list(count_cursor)
        total_count = total_list[0] if total_list else 0
        
        search_time = time.time() - start_time
        
        return {
            "results": results,
            "total": total_count,
            "query": query_text,
            "time": search_time,
            "search_engine": "arangodb-direct",
            "format": output_format
        }
    
    except Exception as e:
        logger.error(f"Direct ArangoDB search failed: {str(e)}")
        logger.exception("Detailed error information:")
        return {
            "results": [],
            "total": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e),
            "search_engine": "arangodb-direct-failed",
            "format": output_format
        }

def _filtered_semantic_search(
    db: StandardDatabase,
    query_embedding: List[float],
    query_text: str,
    collection_name: str,
    embedding_field: str,
    filter_expr: Optional[str] = None,
    tag_list: Optional[List[str]] = None,
    min_score: float = 0.7,
    top_n: int = 10,
    start_time: float = None,
    output_format: str = "table"  # Added this parameter
) -> Dict[str, Any]:
    """ 
    Two-stage semantic search with filtering using ArangoDB.
    
    This function implements a two-stage approach to work around ArangoDB's 
    limitation that prevents combining filtering with vector search.
    
    Args:
        db: ArangoDB database
        query_embedding: Query embedding vector
        query_text: Original query text
        collection_name: Collection to search
        embedding_field: Field containing embeddings
        filter_expr: AQL filter expression
        tag_list: List of tags to filter by
        min_score: Minimum similarity threshold
        top_n: Maximum results to return
        start_time: Start time for timing
        
    Returns:
        Dict with search results
    """
    if start_time is None:
        start_time = time.time()
    
    try:
        # --------------------------------------------------------------------
        # STAGE 1: Get documents that match filter criteria
        # --------------------------------------------------------------------
        stage1_time = time.time()
        
        # Build the filter expression from the filter_expr and tag_list
        filter_parts = []
        bind_vars = {}
        
        if filter_expr:
            filter_parts.append(f"({filter_expr})")
        
        if tag_list:
            bind_vars["tag_list"] = tag_list
            filter_parts.append("doc.tags ALL IN @tag_list")
        
        filter_clause = ""
        if filter_parts:
            filter_clause = f"FILTER {' AND '.join(filter_parts)}"
        
        # Optimize query to only fetch essential fields
        filtered_query = f"""
        FOR doc IN {collection_name}
        {filter_clause}
        RETURN {{
            _id: doc._id,
            _key: doc._key,
            _rev: doc._rev,
            question: doc.question,
            label: doc.label,
            validated: doc.validated
        }}
        """
        
        logger.debug(f"Stage 1 Query: {filtered_query}")
        
        # Add retry logic for transient errors
        filtered_docs_cursor = execute_aql_with_retry(db, filtered_query, bind_vars=bind_vars)
        filtered_docs = list(filtered_docs_cursor)
        
        stage1_elapsed = time.time() - stage1_time
        logger.debug(f"Stage 1 completed in {stage1_elapsed:.3f}s")
        
        # Early return if no documents match the filter criteria
        if not filtered_docs:
            logger.info("No documents match the filter criteria")
            return {
                "results": [],
                "total": 0,
                "query": query_text,
                "time": time.time() - start_time,
                "search_engine": "arangodb-filtered"
            }
        
        logger.info(f"Stage 1: Found {len(filtered_docs)} documents matching filters")
        
        # Create a lookup map for O(1) access to filtered documents by ID
        filtered_ids_map = {doc["_id"]: doc for doc in filtered_docs}
        
        # --------------------------------------------------------------------
        # STAGE 2: Perform unrestricted vector search (no filtering)
        # --------------------------------------------------------------------
        stage2_time = time.time()
        
        # Add caching for vector search results
        query_embedding_tuple = tuple(query_embedding)
        vector_results = get_cached_vector_results(
            db, 
            collection_name, 
            embedding_field, 
            query_embedding_tuple, 
            top_n * 10
        )
        
        stage2_elapsed = time.time() - stage2_time
        logger.debug(f"Stage 2 completed in {stage2_elapsed:.3f}s")
        
        # --------------------------------------------------------------------
        # STAGE 3: Join results in Python
        # --------------------------------------------------------------------
        stage3_time = time.time()
        
        # Combine vector search results with filtered documents
        results = []
        filtered_count = 0
        for result in vector_results:
            doc_id = result["id"]
            score = result["similarity_score"]
            
            # Skip documents that don't meet the minimum score
            if score < min_score:
                continue
            
            # Skip documents that didn't match our filter criteria
            if doc_id not in filtered_ids_map:
                filtered_count += 1
                continue
            
            # Add matching document to results
            results.append({
                "doc": filtered_ids_map[doc_id],
                "similarity_score": score
            })
            
            # Stop once we have enough results
            if len(results) >= top_n:
                break
        
        stage3_elapsed = time.time() - stage3_time
        logger.debug(f"Stage 3 completed in {stage3_elapsed:.3f}s")
        
        logger.debug(f"Stage 3: Filtered {filtered_count} docs that didn't match criteria")
        logger.info(f"Final results: {len(results)} documents with similarity >= {min_score}")
        
        # Calculate total search time
        search_time = time.time() - start_time
        
        return {
            "results": results,
            "total": len(filtered_docs),  # Total refers to docs that matched filter criteria
            "query": query_text,
            "time": search_time,
            "search_engine": "arangodb-filtered",
            "timings": {
                "stage1": stage1_elapsed,
                "stage2": stage2_elapsed,
                "stage3": stage3_elapsed,
                "total": search_time
            }
        }
    
    except Exception as e:
        logger.error(f"Filtered ArangoDB search failed: {str(e)}")
        logger.exception("Detailed error information:")
        return {
            "results": [],
            "total": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e),
            "search_engine": "arangodb-filtered-failed"
        }


async def validate_results(search_results: Dict[str, Any], expected_results: Dict[str, Any]) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate search results against expected results.
    
    Args:
        search_results: Actual search results
        expected_results: Expected results for validation
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    # Check search engine type
    if search_results.get("search_engine") != expected_results.get("expected_engine"):
        validation_failures["search_engine"] = {
            "expected": expected_results.get("expected_engine"),
            "actual": search_results.get("search_engine")
        }
    
    # Check result count
    results_count = len(search_results.get("results", []))
    min_expected = expected_results.get("min_results", 0)
    if results_count < min_expected:
        validation_failures["results_count"] = {
            "expected": f">= {min_expected}",
            "actual": results_count
        }
    
    # Check if error is present when not expected
    if not expected_results.get("has_error", False) and "error" in search_results:
        validation_failures["unexpected_error"] = {
            "expected": "No error",
            "actual": search_results.get("error")
        }
    
    return len(validation_failures) == 0, validation_failures


def print_search_results(search_results: Dict[str, Any], max_width: int = 120) -> None:
    """
    Print search results in the specified format (table or JSON).
    
    Args:
        search_results: The search results to display
        max_width: Maximum width for text fields in characters (used for table format)
    """
    # Get the requested output format
    output_format = search_results.get("format", "table").lower()
    
    # For JSON output, just print the JSON
    if output_format == "json":
        json_results = {
            "results": search_results.get("results", []),
            "total": search_results.get("total", 0),
            "query": search_results.get("query", ""),
            "time": search_results.get("time", 0),
            "search_engine": search_results.get("search_engine", "semantic")
        }
        print(json.dumps(json_results, indent=2))
        return
    
    # Initialize colorama for cross-platform colored terminal output
    init(autoreset=True)
    
    # Print basic search metadata
    result_count = len(search_results.get("results", []))
    total_count = search_results.get("total", 0)
    query = search_results.get("query", "")
    search_time = search_results.get("time", 0)
    search_engine = search_results.get("search_engine", "semantic")
    
    print(f"{Fore.CYAN}{'═' * 80}{Style.RESET_ALL}")
    print(f"Found {Fore.GREEN}{result_count}{Style.RESET_ALL} results for query '{Fore.YELLOW}{query}{Style.RESET_ALL}'")
    print(f"Engine: {Fore.MAGENTA}{search_engine}{Style.RESET_ALL}, Time: {Fore.CYAN}{search_time:.3f}s{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")
    
    # Use common display utility for consistent formatting across search modes
    display_results(
        search_results,
        max_width=max_width,
        title_field="Content",
        id_field="_key",
        score_field="similarity_score",
        score_name="Similarity Score",
        table_title="Semantic Search Results"
    )
    
    # Print detailed info for first result if there are results
    results = search_results.get("results", [])
    if results:
        print_result_details(results[0])


def print_result_details(result: Dict[str, Any]) -> None:
    """
    Print beautifully formatted details about a search result.
    
    Args:
        result: Search result to display
    """
    # Initialize colorama for cross-platform colored terminal output
    init(autoreset=True)
    
    doc = result.get("doc", {})
    score = result.get("similarity_score", 0)
    
    # Print document header with key
    key = doc.get("_key", "N/A")
    header = f"{Fore.GREEN}{'═' * 80}{Style.RESET_ALL}"
    print(f"\n{header}")
    print(f"{Fore.GREEN}  DOCUMENT: {Fore.YELLOW}{key}{Style.RESET_ALL}  ")
    print(f"{header}")
    
    # Get fields to display (excluding internal fields and tags)
    display_fields = [f for f in doc.keys() if f not in ["_key", "_id", "tags", "_rev", "embedding"]]
    
    # Print all fields with truncation
    for field in display_fields:
        if field in doc and doc[field]:
            field_title = field.title()
            # Truncate large field values
            safe_value = truncate_large_value(doc[field], max_str_len=100)
            print(f"{Fore.YELLOW}{field_title}:{Style.RESET_ALL} {safe_value}")
    
    # Print score with color coding based on value
    if score > 0.9:
        score_str = f"{Fore.GREEN}{score:.2f}{Style.RESET_ALL}"
    elif score > 0.7:
        score_str = f"{Fore.YELLOW}{score:.2f}{Style.RESET_ALL}"
    else:
        score_str = f"{Fore.WHITE}{score:.2f}{Style.RESET_ALL}"
    print(f"\n{Fore.CYAN}Similarity Score:{Style.RESET_ALL} {score_str}")
    
    # Print tags in a special section if present with truncation
    if "tags" in doc and isinstance(doc["tags"], list) and doc["tags"]:
        tags = doc["tags"]
        print(f"\n{Fore.BLUE}Tags:{Style.RESET_ALL}")
        
        # Truncate tag list if it's very long
        safe_tags = truncate_large_value(tags, max_list_elements_shown=10)
        
        if isinstance(safe_tags, str):  # It's already a summary string
            print(f"  {safe_tags}")
        else:  # It's still a list
            tag_colors = [Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.GREEN, Fore.YELLOW]
            for i, tag in enumerate(safe_tags):
                color = tag_colors[i % len(tag_colors)]  # Cycle through colors
                print(f"  • {color}{tag}{Style.RESET_ALL}")
    
    # Print additional timings if available (specific to this module)
    if "timings" in result:
        timings = result.get("timings", {})
        print(f"\n{Fore.MAGENTA}Performance Details:{Style.RESET_ALL}")
        for stage, time_value in timings.items():
            print(f"  {Fore.CYAN}{stage}:{Style.RESET_ALL} {time_value:.3f}s")
    
    # Print footer
    print(f"{header}\n")

if __name__ == "__main__":
    """
    Test the semantic search functionality with validation against expected results.
    """
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{message}</cyan>",
        colorize=True
    )
    
    # Parse command line arguments
    output_format = "table"
    query_text = "primary color"
    min_score = 0.5
    top_n = 5
    use_filter = False
    use_pytorch = False

    # Parse command line arguments
    for i, arg in enumerate(sys.argv):
        if arg == "--format" and i+1 < len(sys.argv):
            output_format = sys.argv[i+1]
        elif arg == "--json":
            output_format = "json"
        elif arg == "--query" and i+1 < len(sys.argv):
            query_text = sys.argv[i+1]
        elif arg == "--threshold" and i+1 < len(sys.argv):
            min_score = float(sys.argv[i+1])
        elif arg == "--top" and i+1 < len(sys.argv):
            top_n = int(sys.argv[i+1])
        elif arg == "--filter":
            use_filter = True
        elif arg == "--pytorch":
            use_pytorch = True
    
    # Expected outputs based on known test cases
    expected_outputs = {
        "direct_search": {
            "expected_engine": "arangodb-direct",
            "min_results": 1,  # At least one result expected
            "has_error": False
        },
        "filtered_search": {
            "expected_engine": "arangodb-filtered",
            "min_results": 1,  # At least one result expected
            "has_error": False
        },
        "pytorch_search": {
            "expected_engine": "pytorch",
            "min_results": 1,  # At least one result expected
            "has_error": False
        }
    }
    
    validation_failures = {}
    
    # Test the semantic search functionality
    try:
        # Set up database connection
        client = connect_arango()
        db = ensure_database(client)
        
        # Import necessary display utilities
        from complexity.arangodb.display_utils import print_search_results as display_results
        from complexity.arangodb.log_utils import truncate_large_value
        
        # Initialize colorama
        init(autoreset=True)
        
        # Create search options based on arguments
        search_options = {
            "db": db,
            "query": query_text,
            "top_n": top_n,
            "min_score": min_score,
            "output_format": output_format
        }
        
        if use_filter:
            search_options["filter_expr"] = "doc.label == 1"
            expected_key = "filtered_search"
            logger.info(f"Running filtered search for '{query_text}'")
        elif use_pytorch:
            search_options["force_pytorch"] = True
            expected_key = "pytorch_search"
            logger.info(f"Running PyTorch search for '{query_text}'")
        else:
            expected_key = "direct_search"
            logger.info(f"Running direct search for '{query_text}'")
        
        # Run the search
        search_results = semantic_search(**search_options)
        
        # Print metadata headers
        print(f"{Fore.CYAN}{'═' * 80}{Style.RESET_ALL}")
        print(f"Found {Fore.GREEN}{len(search_results.get('results', []))}{Style.RESET_ALL} results for query '{Fore.YELLOW}{query_text}{Style.RESET_ALL}'")
        print(f"Engine: {Fore.MAGENTA}{search_results.get('search_engine', 'semantic')}{Style.RESET_ALL}, Time: {Fore.CYAN}{search_results.get('time', 0):.3f}s{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")
        
        # Use the common display utility
        display_results(
            search_results,
            max_width=120,
            title_field="Content",
            id_field="_key",
            score_field="similarity_score",
            score_name="Similarity Score",
            table_title="Semantic Search Results"
        )
        
        # Print detailed info for first result if there are results
        results = search_results.get("results", [])
        if results:
            print_result_details(results[0])
        
        # Validate results
        import asyncio
        validation_passed, current_failures = asyncio.run(
            validate_results(search_results, expected_outputs[expected_key])
        )
        
        if not validation_passed:
            validation_failures.update(current_failures)
        
        # Report validation results
        if not validation_failures:
            print(f"{Fore.GREEN}✅ VALIDATION PASSED - Search results match expected patterns{Style.RESET_ALL}")
            sys.exit(0)
        else:
            print(f"{Fore.RED}❌ VALIDATION FAILED - Results don't match expected values{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}FAILURE DETAILS:{Style.RESET_ALL}")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        print(f"{Fore.RED}❌ VALIDATION FAILED - Search test failed with error: {e}{Style.RESET_ALL}")
        sys.exit(1)