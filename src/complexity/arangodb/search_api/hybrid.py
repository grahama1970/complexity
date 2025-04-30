"""
# Hybrid Search Module for PDF Extractor

This module provides a combined search approach that leverages BM25 text search,
semantic vector search, tag filtering, and optionally graph traversal capabilities,
delivering the best results from multiple search paradigms using Reciprocal Rank Fusion (RRF).

## Features:
- Combined BM25 and semantic search results with RRF re-ranking
- Tag filtering capabilities
- Optional graph traversal integration
- Customizable weighting between search types
- Configurable thresholds and result counts
- Multiple output formats (JSON, table)
- Dedicated validation functionality

## Third-Party Packages:
- python-arango: https://python-driver.arangodb.com/ (v3.10.0)
- loguru: https://github.com/Delgan/loguru (v0.7.2)
- colorama: For colored terminal output
- tabulate: For table formatting
- rich: For enhanced terminal output (optional)

## Sample Input:
```python
query_text = "python error handling"
tag_list = ["python", "error-handling"]
filter_expr = "doc.label == 1"
min_score = {
    "bm25": 0.1,
    "semantic": 0.7
}
weights = {
    "bm25": 0.4,
    "semantic": 0.6
}
```

## Expected Output:
```json
{
  "results": [
    {
      "doc": {
        "_key": "doc1",
        "problem": "Python error when processing JSON data",
        "solution": "Use try/except blocks to handle JSON parsing exceptions",
        "context": "Error handling in data processing",
        "tags": ["python", "json", "error-handling"]
      },
      "bm25_score": 0.85,
      "semantic_score": 0.92,
      "hybrid_score": 0.89
    }
  ],
  "total": 1,
  "query": "python error handling",
  "time": 0.245,
  "search_engine": "hybrid-bm25-semantic",
  "weights": {"bm25": 0.4, "semantic": 0.6},
  "format": "json"
}
```
"""

import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError
from colorama import init, Fore, Style
from tabulate import tabulate

# Import config variables and module utilities
from complexity.arangodb.config import (
    COLLECTION_NAME,
    SEARCH_FIELDS,
    ALL_DATA_FIELDS_PREVIEW,
    TEXT_ANALYZER,
    VIEW_NAME,
    EMBEDDING_FIELD,
    GRAPH_NAME,
    EDGE_COLLECTION_NAME
)
from complexity.arangodb.arango_setup import connect_arango, ensure_database
from complexity.arangodb.embedding_utils import get_embedding
from complexity.arangodb.display_utils import print_search_results
from complexity.arangodb.log_utils import truncate_large_value

# Import search modules
from complexity.arangodb.search_api.bm25_search import bm25_search
from complexity.arangodb.search_api.semantic_search import semantic_search
from complexity.arangodb.search_api.tag_search import tag_search
from complexity.arangodb.search_api.graph_traverse import graph_rag_search


def hybrid_search(
    db: StandardDatabase,
    query_text: str,
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    tag_list: Optional[List[str]] = None,
    min_score: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    top_n: int = 10,
    initial_k: int = 20,
    rrf_k: int = 60,
    output_format: str = "table",
    fields_to_return: Optional[List[str]] = None,
    require_all_tags: bool = False,
    use_graph: bool = False,
    graph_min_depth: int = 1,
    graph_max_depth: int = 1,
    graph_direction: str = "ANY",
    relationship_types: Optional[List[str]] = None,
    edge_collection_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs hybrid search by combining BM25, Semantic search, and optionally
    graph traversal results using Reciprocal Rank Fusion (RRF) for re-ranking.

    Args:
        db: ArangoDB database connection
        query_text: The user's search query
        collections: Optional list of collections to search in
        filter_expr: Optional AQL filter expression
        tag_list: Optional list of tags to filter results
        min_score: Dictionary of minimum scores for each search type (bm25, semantic)
        weights: Dictionary of weights for each search type (bm25, semantic, graph)
        top_n: The final number of ranked results to return
        initial_k: Number of results to initially fetch from BM25 and Semantic searches
        rrf_k: Constant used in the RRF calculation (default 60)
        output_format: Output format ("table" or "json")
        fields_to_return: Fields to include in the result
        require_all_tags: Whether all tags must be present (for tag filtering)
        use_graph: Whether to include graph traversal in the hybrid search
        graph_min_depth: Minimum traversal depth for graph search
        graph_max_depth: Maximum traversal depth for graph search
        graph_direction: Direction of traversal (OUTBOUND, INBOUND, ANY)
        relationship_types: Optional list of relationship types to filter
        edge_collection_name: Custom edge collection name (uses default if None)

    Returns:
        A dictionary containing the ranked 'results', 'total' unique documents found,
        the 'query' for reference, and other metadata.
    """
    start_time = time.time()
    logger.info(f"Hybrid search for query: '{query_text}'")
    
    # Input validation
    if not query_text or query_text.strip() == "":
        error_msg = "Query text cannot be empty"
        logger.error(error_msg)
        return {
            "results": [],
            "total": 0,
            "query": "",
            "time": time.time() - start_time,
            "format": output_format,
            "error": error_msg,
            "search_engine": "hybrid-failed"
        }
    
    # Use default collection if not specified
    if not collections:
        collections = [COLLECTION_NAME]
    
    # Default fields to return if not provided
    if not fields_to_return:
        fields_to_return = ["_key", "_id", "question", "problem", "solution", "context", "tags", "label", "validated"]
    
    # Default minimum scores if not provided
    if not min_score:
        min_score = {
            "bm25": 0.1,
            "semantic": 0.7,
            "graph": 0.5
        }
    elif isinstance(min_score, float):
        min_score = {
            "bm25": min_score,
            "semantic": min_score,
            "graph": min_score
        }
    
    # Default weights if not provided
    if not weights:
        if use_graph:
            weights = {
                "bm25": 0.3,
                "semantic": 0.5,
                "graph": 0.2
            }
        else:
            weights = {
                "bm25": 0.5,
                "semantic": 0.5
            }
    
    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    if total_weight != 1.0:
        logger.warning(f"Weights do not sum to 1.0 ({total_weight}), normalizing...")
        for key in weights:
            weights[key] = weights[key] / total_weight
    
    try:
        # STEP 1: Run tag search if tags are provided
        tag_filtered_ids = None
        if tag_list and len(tag_list) > 0:
            logger.info(f"Filtering by tags: {tag_list} (require_all_tags={require_all_tags})")
            tag_search_results = tag_search(
                db=db,
                tags=tag_list,
                collections=collections,
                filter_expr=filter_expr,
                require_all_tags=require_all_tags,
                limit=initial_k * 2,  # Get extra results to ensure enough after filtering
                output_format="json",
                fields_to_return=["_id", "_key"]  # Minimal fields for efficiency
            )
            
            if tag_search_results.get("results", []):
                tag_filtered_ids = {r["doc"]["_id"] for r in tag_search_results.get("results", [])}
                logger.info(f"Tag search found {len(tag_filtered_ids)} matching documents")
                # If no documents matched the tag criteria, return empty results
                if not tag_filtered_ids:
                    logger.warning(f"No documents matched the specified tags: {tag_list}")
                    return {
                        "results": [],
                        "total": 0,
                        "query": query_text,
                        "time": time.time() - start_time,
                        "search_engine": "hybrid-tag-filtered",
                        "weights": weights,
                        "format": output_format,
                        "tags": tag_list,
                        "require_all_tags": require_all_tags
                    }
        
        # STEP 2: Run BM25 search
        bm25_time_start = time.time()
        bm25_results = bm25_search(
            db=db,
            query_text=query_text,
            collections=collections,
            filter_expr=filter_expr,
            min_score=min_score.get("bm25", 0.1),
            top_n=initial_k,
            tag_list=tag_list,
            output_format="json"
        )
        bm25_time = time.time() - bm25_time_start
        
        # Extract BM25 candidates
        bm25_candidates = bm25_results.get("results", [])
        logger.info(f"BM25 search found {len(bm25_candidates)} candidates in {bm25_time:.3f}s")
        
        # STEP 3: Run semantic search
        semantic_time_start = time.time()
        
        # Get embedding for query
        query_embedding = get_embedding(query_text)
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            return {
                "results": [],
                "total": 0,
                "query": query_text,
                "time": time.time() - start_time,
                "error": "Failed to generate embedding for semantic search",
                "search_engine": "hybrid-failed",
                "format": output_format
            }
        
        semantic_results = semantic_search(
            db=db,
            query=query_text,
            collections=collections,
            filter_expr=filter_expr,
            min_score=min_score.get("semantic", 0.7),
            top_n=initial_k,
            tag_list=tag_list,
            output_format="json"
        )
        semantic_time = time.time() - semantic_time_start
        
        # Extract semantic candidates
        semantic_candidates = semantic_results.get("results", [])
        logger.info(f"Semantic search found {len(semantic_candidates)} candidates in {semantic_time:.3f}s")
        
        # STEP 4: Run graph traversal search if enabled
        graph_candidates = []
        graph_time = 0
        
        if use_graph:
            logger.info(f"Running graph traversal search (depth: {graph_min_depth}-{graph_max_depth}, direction: {graph_direction})")
            graph_time_start = time.time()
            
            # Use default or custom edge collection
            edge_col = edge_collection_name or EDGE_COLLECTION_NAME
            
            # Check if edge collection exists
            if not db.has_collection(edge_col):
                logger.warning(f"Edge collection '{edge_col}' does not exist, skipping graph search")
            else:
                # Run graph traversal search
                graph_results = graph_rag_search(
                    db=db,
                    query_text=query_text,
                    min_depth=graph_min_depth,
                    max_depth=graph_max_depth,
                    direction=graph_direction,
                    relationship_types=relationship_types,
                    min_score=min_score.get("graph", 0.5),
                    top_n=initial_k,
                    output_format="json",
                    fields_to_return=fields_to_return,
                    edge_collection_name=edge_col
                )
                
                # Extract graph candidates
                graph_candidates = graph_results.get("results", [])
                
                # Process related documents to add them as candidates
                for result in graph_candidates:
                    # Add related documents as candidates
                    related_docs = result.get("related", [])
                    for related in related_docs:
                        vertex = related.get("vertex", {})
                        if vertex:
                            # Add as a graph candidate with the score from the original document
                            graph_candidates.append({
                                "doc": vertex,
                                "score": result.get("score", 0) * 0.8  # Slightly reduce score for related docs
                            })
                
                logger.info(f"Graph search found {len(graph_candidates)} candidates (including related docs)")
            
            graph_time = time.time() - graph_time_start
        
        # STEP 5: Apply tag filtering if needed
        if tag_filtered_ids is not None:
            logger.info(f"Applying tag filtering to search results...")
            
            # Filter BM25 results
            filtered_bm25_candidates = []
            for result in bm25_candidates:
                doc_id = result.get("doc", {}).get("_id")
                if doc_id in tag_filtered_ids:
                    filtered_bm25_candidates.append(result)
            
            # Filter semantic results
            filtered_semantic_candidates = []
            for result in semantic_candidates:
                doc_id = result.get("doc", {}).get("_id")
                if doc_id in tag_filtered_ids:
                    filtered_semantic_candidates.append(result)
            
            # Filter graph results
            filtered_graph_candidates = []
            for result in graph_candidates:
                doc_id = result.get("doc", {}).get("_id")
                if doc_id in tag_filtered_ids:
                    filtered_graph_candidates.append(result)
            
            # Update candidates
            bm25_candidates = filtered_bm25_candidates
            semantic_candidates = filtered_semantic_candidates
            graph_candidates = filtered_graph_candidates
            
            logger.info(f"After tag filtering: BM25 has {len(bm25_candidates)}, Semantic has {len(semantic_candidates)}, Graph has {len(graph_candidates)}")
        
        # STEP 6: Combine results using weighted RRF
        combined_weights = weights
        logger.info(f"Combining results with weights: {combined_weights}")
        
        if use_graph and graph_candidates:
            combined_results = weighted_reciprocal_rank_fusion_with_graph(
                bm25_candidates=bm25_candidates,
                semantic_candidates=semantic_candidates,
                graph_candidates=graph_candidates,
                weights=combined_weights,
                rrf_k=rrf_k
            )
        else:
            combined_results = weighted_reciprocal_rank_fusion(
                bm25_candidates=bm25_candidates,
                semantic_candidates=semantic_candidates,
                weights=combined_weights,
                rrf_k=rrf_k
            )
        
        # STEP 7: Limit to top_n results
        final_results = combined_results[:top_n]
        logger.info(f"Final results: {len(final_results)} documents")
        
        search_time = time.time() - start_time
        
        # Build the response
        response = {
            "results": final_results,
            "total": len(combined_results),
            "query": query_text,
            "time": search_time,
            "bm25_time": bm25_time,
            "semantic_time": semantic_time,
            "search_engine": "hybrid-bm25-semantic",
            "weights": weights,
            "format": output_format,
            "tags": tag_list,
            "require_all_tags": require_all_tags if tag_list else None
        }
        
        # Add graph search info if used
        if use_graph:
            response["graph_time"] = graph_time
            response["search_engine"] = "hybrid-bm25-semantic-graph"
            response["graph_params"] = {
                "min_depth": graph_min_depth,
                "max_depth": graph_max_depth,
                "direction": graph_direction,
                "relationship_types": relationship_types
            }
        
        return response
    
    except Exception as e:
        logger.exception(f"Hybrid search error: {e}")
        return {
            "results": [],
            "total": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e),
            "search_engine": "hybrid-failed",
            "format": output_format
        }


def weighted_reciprocal_rank_fusion(
    bm25_candidates: List[Dict[str, Any]],
    semantic_candidates: List[Dict[str, Any]],
    weights: Dict[str, float] = None,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines multiple result lists using Weighted Reciprocal Rank Fusion.

    Args:
        bm25_candidates: Results from BM25 search
        semantic_candidates: Results from semantic search
        weights: Dictionary of weights for each search type
        rrf_k: Constant for the RRF formula (default: 60)

    Returns:
        A combined list of results, sorted by hybrid score
    """
    if weights is None:
        weights = {
            "bm25": 0.5,
            "semantic": 0.5
        }
    
    # Create a dictionary to track document keys and their rankings
    doc_scores = {}

    # Process BM25 results
    bm25_weight = weights.get("bm25", 0.5)
    for rank, result in enumerate(bm25_candidates, 1):
        doc_key = result.get("doc", {}).get("_key", "")
        if not doc_key:
            continue

        # Initialize if not seen before
        if doc_key not in doc_scores:
            doc_scores[doc_key] = {
                "doc": result.get("doc", {}),
                "bm25_rank": rank,
                "bm25_score": result.get("score", 0),
                "semantic_rank": len(semantic_candidates) + 1,  # Default to worst possible rank
                "semantic_score": 0,
                "hybrid_score": 0
            }
        else:
            # Update BM25 rank info
            doc_scores[doc_key]["bm25_rank"] = rank
            doc_scores[doc_key]["bm25_score"] = result.get("score", 0)

    # Process semantic results
    semantic_weight = weights.get("semantic", 0.5)
    for rank, result in enumerate(semantic_candidates, 1):
        doc_key = result.get("doc", {}).get("_key", "")
        if not doc_key:
            continue

        # Initialize if not seen before
        if doc_key not in doc_scores:
            doc_scores[doc_key] = {
                "doc": result.get("doc", {}),
                "bm25_rank": len(bm25_candidates) + 1,  # Default to worst possible rank
                "bm25_score": 0,
                "semantic_rank": rank,
                "semantic_score": result.get("similarity_score", 0),
                "hybrid_score": 0
            }
        else:
            # Update semantic rank info
            doc_scores[doc_key]["semantic_rank"] = rank
            doc_scores[doc_key]["semantic_score"] = result.get("similarity_score", 0)

    # Calculate weighted RRF scores
    for doc_key, scores in doc_scores.items():
        # Calculate individual RRF scores
        bm25_rrf = 1 / (rrf_k + scores["bm25_rank"])
        semantic_rrf = 1 / (rrf_k + scores["semantic_rank"])
        
        # Apply weights
        weighted_bm25 = bm25_rrf * bm25_weight
        weighted_semantic = semantic_rrf * semantic_weight
        
        # Calculate hybrid score
        scores["hybrid_score"] = weighted_bm25 + weighted_semantic

    # Convert to list and sort by hybrid score (descending)
    result_list = [v for k, v in doc_scores.items()]
    result_list.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return result_list


def weighted_reciprocal_rank_fusion_with_graph(
    bm25_candidates: List[Dict[str, Any]],
    semantic_candidates: List[Dict[str, Any]],
    graph_candidates: List[Dict[str, Any]],
    weights: Dict[str, float] = None,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines multiple result lists including graph traversal using Weighted RRF.

    Args:
        bm25_candidates: Results from BM25 search
        semantic_candidates: Results from semantic search
        graph_candidates: Results from graph traversal search
        weights: Dictionary of weights for each search type
        rrf_k: Constant for the RRF formula (default: 60)

    Returns:
        A combined list of results, sorted by hybrid score
    """
    if weights is None:
        weights = {
            "bm25": 0.4,
            "semantic": 0.4,
            "graph": 0.2
        }
    
    # Create a dictionary to track document keys and their rankings
    doc_scores = {}

    # Process BM25 results
    bm25_weight = weights.get("bm25", 0.4)
    for rank, result in enumerate(bm25_candidates, 1):
        doc_key = result.get("doc", {}).get("_key", "")
        if not doc_key:
            continue

        # Initialize if not seen before
        if doc_key not in doc_scores:
            doc_scores[doc_key] = {
                "doc": result.get("doc", {}),
                "bm25_rank": rank,
                "bm25_score": result.get("score", 0),
                "semantic_rank": len(semantic_candidates) + 1,  # Default to worst possible rank
                "semantic_score": 0,
                "graph_rank": len(graph_candidates) + 1,  # Default to worst possible rank
                "graph_score": 0,
                "hybrid_score": 0
            }
        else:
            # Update BM25 rank info
            doc_scores[doc_key]["bm25_rank"] = rank
            doc_scores[doc_key]["bm25_score"] = result.get("score", 0)

    # Process semantic results
    semantic_weight = weights.get("semantic", 0.4)
    for rank, result in enumerate(semantic_candidates, 1):
        doc_key = result.get("doc", {}).get("_key", "")
        if not doc_key:
            continue

        # Initialize if not seen before
        if doc_key not in doc_scores:
            doc_scores[doc_key] = {
                "doc": result.get("doc", {}),
                "bm25_rank": len(bm25_candidates) + 1,  # Default to worst possible rank
                "bm25_score": 0,
                "semantic_rank": rank,
                "semantic_score": result.get("similarity_score", 0),
                "graph_rank": len(graph_candidates) + 1,  # Default to worst possible rank
                "graph_score": 0,
                "hybrid_score": 0
            }
        else:
            # Update semantic rank info
            doc_scores[doc_key]["semantic_rank"] = rank
            doc_scores[doc_key]["semantic_score"] = result.get("similarity_score", 0)
    
    # Process graph results
    graph_weight = weights.get("graph", 0.2)
    for rank, result in enumerate(graph_candidates, 1):
        doc_key = result.get("doc", {}).get("_key", "")
        if not doc_key:
            continue

        # Initialize if not seen before
        if doc_key not in doc_scores:
            doc_scores[doc_key] = {
                "doc": result.get("doc", {}),
                "bm25_rank": len(bm25_candidates) + 1,  # Default to worst possible rank
                "bm25_score": 0,
                "semantic_rank": len(semantic_candidates) + 1,  # Default to worst possible rank
                "semantic_score": 0,
                "graph_rank": rank,
                "graph_score": result.get("score", 0),
                "hybrid_score": 0
            }
        else:
            # Update graph rank info
            doc_scores[doc_key]["graph_rank"] = rank
            doc_scores[doc_key]["graph_score"] = result.get("score", 0)

    # Calculate weighted RRF scores
    for doc_key, scores in doc_scores.items():
        # Calculate individual RRF scores
        bm25_rrf = 1 / (rrf_k + scores["bm25_rank"])
        semantic_rrf = 1 / (rrf_k + scores["semantic_rank"])
        graph_rrf = 1 / (rrf_k + scores["graph_rank"])
        
        # Apply weights
        weighted_bm25 = bm25_rrf * bm25_weight
        weighted_semantic = semantic_rrf * semantic_weight
        weighted_graph = graph_rrf * graph_weight
        
        # Calculate hybrid score
        scores["hybrid_score"] = weighted_bm25 + weighted_semantic + weighted_graph

    # Convert to list and sort by hybrid score (descending)
    result_list = [v for k, v in doc_scores.items()]
    result_list.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return result_list


def validate_hybrid_search(
    search_results: Dict[str, Any],
    fixture_path: str
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate hybrid search results against known good fixture data.

    Args:
        search_results: The results returned from hybrid_search
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

    # Check for top-level errors first
    if "error" in search_results:
        validation_failures["search_error"] = {
            "expected": "No error",
            "actual": search_results["error"]
        }
        return False, validation_failures

    # Structural validation
    if "results" not in search_results:
        validation_failures["missing_results"] = {
            "expected": "Results field present",
            "actual": "Results field missing"
        }
        return False, validation_failures

    # Validate result count
    actual_results = search_results.get("results", [])
    if "count" in expected_data and len(actual_results) != expected_data.get("count"):
        validation_failures["result_count"] = {
            "expected": expected_data.get("count"),
            "actual": len(actual_results)
        }

    # Validate total count
    if "total" in expected_data and search_results.get("total") != expected_data.get("total"):
        validation_failures["total_count"] = {
            "expected": expected_data.get("total"),
            "actual": search_results.get("total")
        }

    # Validate query text
    if "query" in expected_data and search_results.get("query") != expected_data.get("query"):
        validation_failures["query"] = {
            "expected": expected_data.get("query"),
            "actual": search_results.get("query")
        }

    # Validate search engine type
    if "search_engine" in expected_data and search_results.get("search_engine") != expected_data.get("search_engine"):
        validation_failures["search_engine"] = {
            "expected": expected_data.get("search_engine"),
            "actual": search_results.get("search_engine")
        }

    # Validate individual results if expected results list exists
    expected_results = expected_data.get("results", [])
    if expected_results and len(actual_results) > 0:
        # Only validate up to the minimum length of both lists
        min_length = min(len(expected_results), len(actual_results))
        
        for i in range(min_length):
            expected = expected_results[i]
            actual = actual_results[i]
            
            # Check document keys match
            expected_key = expected.get("doc", {}).get("_key", "")
            actual_key = actual.get("doc", {}).get("_key", "")
            
            if expected_key and expected_key != actual_key:
                validation_failures[f"result_{i}_doc_key"] = {
                    "expected": expected_key,
                    "actual": actual_key
                }
            
            # Check hybrid scores are close
            if "hybrid_score" in expected and "hybrid_score" in actual:
                expected_score = expected["hybrid_score"]
                actual_score = actual["hybrid_score"]
                
                # Allow small differences in scores due to floating point precision
                if abs(expected_score - actual_score) > 0.01:
                    validation_failures[f"result_{i}_hybrid_score"] = {
                        "expected": expected_score,
                        "actual": actual_score
                    }

    return len(validation_failures) == 0, validation_failures


def print_hybrid_search_results(search_results: Dict[str, Any], max_width: int = 120) -> None:
    """
    Print hybrid search results in the specified format (table or JSON).
    
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
            "bm25_time": search_results.get("bm25_time", 0),
            "semantic_time": search_results.get("semantic_time", 0),
            "search_engine": search_results.get("search_engine", "hybrid"),
            "weights": search_results.get("weights", {"bm25": 0.5, "semantic": 0.5})
        }
        
        # Add graph info if present
        if "graph_time" in search_results:
            json_results["graph_time"] = search_results.get("graph_time", 0)
            json_results["graph_params"] = search_results.get("graph_params", {})
            
        print(json.dumps(json_results, indent=2))
        return
    
    # For table output, use the common display utility with hybrid-specific settings
    # Initialize colorama for cross-platform colored terminal output
    init(autoreset=True)
    
    # Print basic search metadata
    result_count = len(search_results.get("results", []))
    total_count = search_results.get("total", 0)
    query = search_results.get("query", "")
    search_time = search_results.get("time", 0)
    bm25_time = search_results.get("bm25_time", 0)
    semantic_time = search_results.get("semantic_time", 0)
    weights = search_results.get("weights", {"bm25": 0.5, "semantic": 0.5})
    
    print(f"{Fore.CYAN}{'═' * 80}{Style.RESET_ALL}")
    print(f"Found {Fore.GREEN}{result_count}{Style.RESET_ALL} results for query '{Fore.YELLOW}{query}{Style.RESET_ALL}'")
    print(f"Engine: {Fore.MAGENTA}{search_results.get('search_engine', 'hybrid')}{Style.RESET_ALL}, Time: {Fore.CYAN}{search_time:.3f}s{Style.RESET_ALL}")
    
    # Show weights based on search mode
    if "graph_time" in search_results:
        print(f"Weights: BM25={Fore.GREEN}{weights.get('bm25', 0.3):.2f}{Style.RESET_ALL}, " +
              f"Semantic={Fore.GREEN}{weights.get('semantic', 0.5):.2f}{Style.RESET_ALL}, " +
              f"Graph={Fore.GREEN}{weights.get('graph', 0.2):.2f}{Style.RESET_ALL}")
        
        # Show graph parameters
        graph_params = search_results.get("graph_params", {})
        min_depth = graph_params.get("min_depth", 1)
        max_depth = graph_params.get("max_depth", 1)
        direction = graph_params.get("direction", "ANY")
        rel_types = graph_params.get("relationship_types", [])
        rel_types_str = ", ".join(rel_types) if rel_types else "Any"
        
        print(f"Graph: Depth={Fore.BLUE}{min_depth}-{max_depth}{Style.RESET_ALL}, " +
              f"Direction={Fore.BLUE}{direction}{Style.RESET_ALL}, " +
              f"Relationships={Fore.BLUE}{rel_types_str}{Style.RESET_ALL}")
    else:
        print(f"Weights: BM25={Fore.GREEN}{weights.get('bm25', 0.5):.2f}{Style.RESET_ALL}, " +
              f"Semantic={Fore.GREEN}{weights.get('semantic', 0.5):.2f}{Style.RESET_ALL}")
    
    # If tag filtering was applied, show tag information
    if search_results.get("tags"):
        tags = search_results.get("tags", [])
        tag_mode = "ALL" if search_results.get("require_all_tags", False) else "ANY"
        print(f"Tags: {Fore.YELLOW}{', '.join(tags)}{Style.RESET_ALL} ({Fore.CYAN}{tag_mode}{Style.RESET_ALL})")
    
    print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")
    
    # Use common display utility for consistent formatting across search modes
    print_search_results(
        search_results,
        max_width=max_width,
        title_field="Content",
        id_field="_key",
        score_field="hybrid_score",
        score_name="Hybrid Score",
        table_title="Hybrid Search Results"
    )
    
    # Print detailed info for first result if there are results
    results = search_results.get("results", [])
    if results:
        print_result_details(results[0])


def print_result_details(result: Dict[str, Any]) -> None:
    """
    Print beautifully formatted details about a hybrid search result.
    
    Args:
        result: Search result to display
    """
    # Initialize colorama for cross-platform colored terminal output
    init(autoreset=True)
    
    doc = result.get("doc", {})
    bm25_score = result.get("bm25_score", 0)
    semantic_score = result.get("semantic_score", 0)
    graph_score = result.get("graph_score", 0)
    hybrid_score = result.get("hybrid_score", 0)
    
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
    
    # Print all scores with color coding
    print(f"\n{Fore.MAGENTA}Search Scores:{Style.RESET_ALL}")
    
    # BM25 score
    if bm25_score > 7.0:
        bm25_str = f"{Fore.GREEN}{bm25_score:.2f}{Style.RESET_ALL}"
    elif bm25_score > 5.0:
        bm25_str = f"{Fore.YELLOW}{bm25_score:.2f}{Style.RESET_ALL}"
    else:
        bm25_str = f"{Fore.WHITE}{bm25_score:.2f}{Style.RESET_ALL}"
    print(f"  • {Fore.CYAN}BM25 Score:{Style.RESET_ALL} {bm25_str}")
    
    # Semantic score
    if semantic_score > 0.9:
        semantic_str = f"{Fore.GREEN}{semantic_score:.2f}{Style.RESET_ALL}"
    elif semantic_score > 0.7:
        semantic_str = f"{Fore.YELLOW}{semantic_score:.2f}{Style.RESET_ALL}"
    else:
        semantic_str = f"{Fore.WHITE}{semantic_score:.2f}{Style.RESET_ALL}"
    print(f"  • {Fore.CYAN}Semantic Score:{Style.RESET_ALL} {semantic_str}")
    
    # Graph score (if available)
    if graph_score > 0:
        if graph_score > 0.9:
            graph_str = f"{Fore.GREEN}{graph_score:.2f}{Style.RESET_ALL}"
        elif graph_score > 0.7:
            graph_str = f"{Fore.YELLOW}{graph_score:.2f}{Style.RESET_ALL}"
        else:
            graph_str = f"{Fore.WHITE}{graph_score:.2f}{Style.RESET_ALL}"
        print(f"  • {Fore.CYAN}Graph Score:{Style.RESET_ALL} {graph_str}")
    
    # Hybrid score
    if hybrid_score > 0.9:
        hybrid_str = f"{Fore.GREEN}{hybrid_score:.2f}{Style.RESET_ALL}"
    elif hybrid_score > 0.7:
        hybrid_str = f"{Fore.YELLOW}{hybrid_score:.2f}{Style.RESET_ALL}"
    else:
        hybrid_str = f"{Fore.WHITE}{hybrid_score:.2f}{Style.RESET_ALL}"
    print(f"  • {Fore.CYAN}Hybrid Score:{Style.RESET_ALL} {hybrid_str}")
    
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
    
    # Print footer
    print(f"{header}\n")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    )
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/hybrid_search_expected.json"
    
    # Parse command line arguments
    output_format = "table"
    query_text = "python error"
    min_score_values = {"bm25": 0.1, "semantic": 0.7, "graph": 0.5}
    weight_values = {"bm25": 0.5, "semantic": 0.5}
    top_n = 5
    tag_list = None
    require_all_tags = False
    use_graph = False
    graph_min_depth = 1
    graph_max_depth = 1
    graph_direction = "ANY"
    relationship_types = None

    # Parse command line arguments
    for i, arg in enumerate(sys.argv):
        if arg == "--format" and i+1 < len(sys.argv):
            output_format = sys.argv[i+1]
        elif arg == "--json":
            output_format = "json"
        elif arg == "--query" and i+1 < len(sys.argv):
            query_text = sys.argv[i+1]
        elif arg == "--bm25-threshold" and i+1 < len(sys.argv):
            min_score_values["bm25"] = float(sys.argv[i+1])
        elif arg == "--semantic-threshold" and i+1 < len(sys.argv):
            min_score_values["semantic"] = float(sys.argv[i+1])
        elif arg == "--graph-threshold" and i+1 < len(sys.argv):
            min_score_values["graph"] = float(sys.argv[i+1])
        elif arg == "--bm25-weight" and i+1 < len(sys.argv):
            weight_values["bm25"] = float(sys.argv[i+1])
            weight_values["semantic"] = 1.0 - weight_values["bm25"]
        elif arg == "--top" and i+1 < len(sys.argv):
            top_n = int(sys.argv[i+1])
        elif arg == "--tags" and i+1 < len(sys.argv):
            tag_list = sys.argv[i+1].split(',')
        elif arg == "--require-all-tags":
            require_all_tags = True
        elif arg == "--graph":
            use_graph = True
            # Update weights for graph
            weight_values = {"bm25": 0.3, "semantic": 0.5, "graph": 0.2}
        elif arg == "--graph-depth" and i+1 < len(sys.argv):
            depths = sys.argv[i+1].split('-')
            if len(depths) == 1:
                graph_min_depth = int(depths[0])
                graph_max_depth = int(depths[0])
            elif len(depths) == 2:
                graph_min_depth = int(depths[0])
                graph_max_depth = int(depths[1])
        elif arg == "--graph-direction" and i+1 < len(sys.argv):
            graph_direction = sys.argv[i+1].upper()
        elif arg == "--relationship-types" and i+1 < len(sys.argv):
            relationship_types = sys.argv[i+1].split(',')
    
    try:
        # Set up database connection
        client = connect_arango()
        db = ensure_database(client)
        
        # Run hybrid search
        logger.info(f"Running hybrid search for '{query_text}'")
        search_results = hybrid_search(
            db=db,
            query_text=query_text,
            min_score=min_score_values,
            weights=weight_values,
            top_n=top_n,
            output_format=output_format,
            tag_list=tag_list,
            require_all_tags=require_all_tags,
            use_graph=use_graph,
            graph_min_depth=graph_min_depth,
            graph_max_depth=graph_max_depth,
            graph_direction=graph_direction,
            relationship_types=relationship_types
        )
        
        # Print the results
        print_hybrid_search_results(search_results)
        
        # Load or create fixture
        try:
            with open(fixture_path, 'r') as f:
                expected_data = json.load(f)
            logger.info(f"Loaded expected results from {fixture_path}")
            
            # Validate the results
            validation_passed, validation_failures = validate_hybrid_search(search_results, fixture_path)
            
            # Report validation status
            if validation_passed:
                print(f"{Fore.GREEN}✅ VALIDATION PASSED - Hybrid search results match expected values{Style.RESET_ALL}")
                sys.exit(0)
            else:
                print(f"{Fore.RED}❌ VALIDATION FAILED - Hybrid search results don't match expected values{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}FAILURE DETAILS:{Style.RESET_ALL}")
                for field, details in validation_failures.items():
                    print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
                print(f"Total errors: {len(validation_failures)} fields mismatched")
                sys.exit(1)
        except FileNotFoundError:
            logger.warning(f"Fixture file not found at {fixture_path}, creating...")
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(fixture_path), exist_ok=True)
            
            # Create fixture with current search results
            expected_data = {
                "query": query_text,
                "count": len(search_results.get("results", [])),
                "total": search_results.get("total", 0),
                "search_engine": search_results.get("search_engine", "hybrid"),
                "results": search_results.get("results", [])
            }
            
            with open(fixture_path, 'w') as f:
                json.dump(expected_data, f, indent=2)
            
            logger.info(f"Created fixture at {fixture_path}")
            print(f"{Fore.GREEN}✅ FIXTURE CREATED - Run again to validate{Style.RESET_ALL}")
            sys.exit(0)
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"{Fore.RED}❌ ERROR: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)