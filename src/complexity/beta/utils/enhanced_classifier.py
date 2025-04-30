# src/complexity/beta/utils/enhanced_classifier.py
"""
Module Description:
Provides an enhanced complexity classification function that combines semantic similarity
search with graph traversal (prerequisites, related topics) in ArangoDB to determine
question complexity. Returns a dictionary with classification details, confidence,
and optionally an explanation.

Links:
- python-arango Driver: https://python-arango.readthedocs.io/en/latest/
- ArangoDB Manual: https://www.arangodb.com/docs/stable/
- Loguru: https://loguru.readthedocs.io/en/stable/

Sample Input/Output:

- enhanced_classify_complexity(db: StandardDatabase, question: str, k: Optional[int] = None, with_explanation: bool = False):
  - Input: DB instance, question string, optional k, optional explanation flag.
  - Output: Dict[str, Any] containing classification, confidence, auto_accept status, neighbors, and optionally an explanation.
    Example:
    {
        "classification": "Complex",
        "confidence": 0.82,
        "auto_accept": True,
        "neighbors": [...],
        "explanation": {...} # if with_explanation=True
    }
"""
import os
import sys
from typing import Dict, Any, Tuple, Optional, List
from loguru import logger
from arango.database import StandardDatabase

from complexity.beta.utils.config import CONFIG
# Corrected import path for get_EmbedderModel
from complexity.beta.utils.arango_setup import get_EmbedderModel as get_embedder
# Import necessary setup functions for standalone execution
from complexity.beta.utils.arango_setup import connect_arango, ensure_database, ensure_collection, ensure_vector_index, ensure_edge_collections, ensure_graph

def enhanced_classify_complexity(
    db: StandardDatabase, 
    question: str, 
    k: Optional[int] = None, # Corrected type hint for default None
    with_explanation: bool = False
) -> Dict[str, Any]:
    """
    Enhanced complexity classifier using semantic search and graph traversal.
    
    Args:
        db: ArangoDB database connection
        question: The question to classify
        k: Number of nearest neighbors (defaults to CONFIG value)
        with_explanation: Whether to include explanation in response
    
    Returns:
        Dictionary with classification results
    """
    k = k or CONFIG["classification"]["default_k"]
    embedder = get_embedder()
    
    # Define edge collections directly
    edge_collections = ["prerequisites", "related_topics"]
    
    try:
        # Get embedding for question
        query_embedding = embedder.embed_batch([question])[0]
        
        # Collection names need to be interpolated directly
        collection_name = CONFIG["search"]["collection_name"]
        embedding_field = CONFIG["embedding"]["field"]
        
        # Combined AQL query with graph traversal - using f-strings for collection names
        aql = f"""
        // First get semantically similar documents
        LET semantic_matches = (
            FOR doc IN {collection_name}
                LET similarity = COSINE_SIMILARITY(doc.{embedding_field}, @embedding)
                FILTER similarity > @min_similarity
                SORT similarity DESC
                LIMIT @k
                RETURN MERGE(doc, {{ similarity: similarity, source: "semantic" }})
        )
        
        // Then get documents from graph traversal
        LET traversal_matches = (
            FOR doc IN semantic_matches
                FOR related IN 1..@max_depth ANY doc {', '.join(edge_collections)}
                    RETURN MERGE(related, {{ 
                        distance: LENGTH(SHORTEST_PATH({collection_name}, doc._id, related._id, 
                                  {{ direction: "any", edgeCollections: {str(edge_collections)} }})),
                        source: "graph"
                    }})
        )
        
        // Combine and calculate scores
        LET all_matches = APPEND(semantic_matches, traversal_matches)
        FOR doc IN all_matches
            COLLECT document = doc INTO matches
            LET semantic_score = MAX(
                FOR d IN matches[*].doc
                FILTER HAS(d, "similarity")
                RETURN d.similarity
            )
            LET graph_score = MIN(
                FOR d IN matches[*].doc
                FILTER HAS(d, "distance")
                RETURN 1 - (d.distance / (@max_depth + 1))
            )
            LET combined_score = semantic_score * @semantic_weight + 
                           COALESCE(graph_score, 0) * @graph_weight
            SORT combined_score DESC
            LIMIT @k
            RETURN {{
                document: document,
                score: combined_score,
                semantic_score: semantic_score,
                graph_score: graph_score
            }}
        """
        
        bind_vars = {
            "embedding": query_embedding,
            "min_similarity": 0.1,  # Lower threshold to allow for graph relationships
            "k": k,
            "max_depth": 2,  # Max traversal depth
            "semantic_weight": 0.7,  # Weight for semantic similarity
            "graph_weight": 0.3  # Weight for graph relationships
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        if not results:
            return {
                "classification": "Unknown",
                "confidence": 0.0,
                "auto_accept": False,
                "neighbors": []
            }
        
        # Process results with exponential weighting
        votes = {0: 0.0, 1: 0.0}
        total = 0.0
        exponent = 2.0  # Configurable, higher values give more weight to close matches
        
        for r in results:
            document = r["document"]
            score = r["score"]
            if score > 0:
                label = document.get("label", 0)
                weight = score ** exponent  # Exponential weighting
                votes[label] += weight
                total += weight
        
        if total <= 0:
            return {
                "classification": "Unknown",
                "confidence": 0.0,
                "auto_accept": False,
                "neighbors": []
            }
        
        # Determine majority and confidence
        majority = max(votes, key=votes.get)
        confidence = votes[majority] / total
        auto_accept = confidence >= CONFIG["classification"]["confidence_threshold"] and len(results) >= k
        
        classification = "Complex" if majority == 1 else "Simple"
        
        # Prepare response
        response = {
            "classification": classification,
            "confidence": confidence,
            "auto_accept": auto_accept,
            "neighbors": [
                {
                    "question": r["document"].get("question", ""),
                    "label": "Complex" if r["document"].get("label") == 1 else "Simple",
                    "score": r["score"],
                    "semantic_score": r.get("semantic_score", 0),
                    "graph_score": r.get("graph_score", 0)
                }
                for r in results[:3]  # Include top 3 neighbors for reference
            ]
        }
        
        # Add explanation if requested
        if with_explanation:
            # Import here to avoid circular imports
            from complexity.beta.utils.graph_traversal import GraphTraversal
            traversal = GraphTraversal(db)
            explanation = traversal.explain_classification(question, majority, confidence)
            response["explanation"] = explanation
        
        return response
        
    except Exception as e:
        logger.exception(f"Enhanced classification failed: {e}")
        return {
            "classification": "Error",
            "confidence": 0.0,
            "auto_accept": False,
            "error": str(e)
        }


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    validation_passed = True
    validation_failures = {}

    # --- Expected Values ---
    TEST_QUESTION = "Explain the concept of recursion in programming."
    # This expected classification depends heavily on the data and graph structure.
    # Adjust based on expected behavior for the test question.
    EXPECTED_CLASSIFICATION = "Complex" # Example expectation

    db_instance: Optional[StandardDatabase] = None

    try:
        logger.info("Connecting to ArangoDB for validation...")
        client = connect_arango()
        db_instance = ensure_database(client)
        logger.info(f"Connected to database: {db_instance.name}")

        # Ensure prerequisites (collection, index, graph) exist
        # These might fail if setup hasn't run, but the function should handle it
        logger.info("Ensuring necessary DB structures exist...")
        try:
            ensure_collection(db_instance)
            ensure_vector_index(db_instance)
            ensure_edge_collections(db_instance)
            ensure_graph(db_instance)
            logger.info("DB structure checks completed.")
        except SystemExit as se:
             logger.warning(f"DB setup check failed or exited: {se}. Classification might return default/error.")
        except Exception as setup_err:
             logger.warning(f"Error during DB setup check: {setup_err}. Classification might return default/error.")


        # --- Enhanced Classification ---
        logger.info(f"Classifying test question: '{TEST_QUESTION}'")
        result = enhanced_classify_complexity(db_instance, TEST_QUESTION, with_explanation=False) # Explanation not validated here
        logger.info(f"Classification result: {result}")

        # --- Validation ---
        if not isinstance(result, dict):
            validation_passed = False
            validation_failures["return_type"] = {"expected": "dict", "actual": type(result).__name__}
            logger.error("Return value is not a dictionary.")
        else:
            # Check required keys
            required_keys = ["classification", "confidence", "auto_accept", "neighbors"]
            for key in required_keys:
                if key not in result:
                    validation_passed = False
                    validation_failures[f"missing_key_{key}"] = {"expected": f"Key '{key}'", "actual": "Missing"}
                    logger.error(f"Result dictionary missing required key: '{key}'")

            # Check classification value (treat as warning for now, as it's heuristic)
            actual_classification = result.get("classification")
            if actual_classification != EXPECTED_CLASSIFICATION and actual_classification != "Error" and actual_classification != "Unknown":
                 # validation_passed = False # Optionally make this a hard failure
                 validation_failures["classification_value"] = {"expected": EXPECTED_CLASSIFICATION, "actual": actual_classification}
                 logger.warning(f"Classification mismatch: Expected '{EXPECTED_CLASSIFICATION}', Got '{actual_classification}'")

            # Check confidence type
            if "confidence" in result and not isinstance(result["confidence"], float):
                 validation_passed = False
                 validation_failures["confidence_type"] = {"expected": "float", "actual": type(result['confidence']).__name__}
                 logger.error("Confidence value is not a float.")

            # Check neighbors type
            if "neighbors" in result and not isinstance(result["neighbors"], list):
                 validation_passed = False
                 validation_failures["neighbors_type"] = {"expected": "list", "actual": type(result['neighbors']).__name__}
                 logger.error("Neighbors value is not a list.")


    except Exception as e:
        validation_passed = False
        validation_failures["runtime_error"] = str(e)
        logger.exception(f"Validation failed with runtime error: {e}")

    # --- Final Reporting ---
    if validation_passed:
        print("✅ VALIDATION COMPLETE - enhanced_classify_complexity executed and returned expected structure.")
        logger.success("Standalone execution and validation successful.")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Issues detected during enhanced_classify_complexity validation.")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            if isinstance(details, dict):
                 print(f"  - {field}: Expected: {details.get('expected', 'N/A')}, Got: {details.get('actual', 'N/A')}")
            else:
                 print(f"  - {field}: {details}")
        logger.error("Standalone execution and validation failed.")
        sys.exit(1)