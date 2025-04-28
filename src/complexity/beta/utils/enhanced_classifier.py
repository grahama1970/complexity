import os
import sys
from typing import Dict, Any, Tuple, Optional, List
from loguru import logger
from arango.database import StandardDatabase
from complexity.beta.utils.config import CONFIG
from complexity.beta.utils.classifier import get_EmbedderModel as get_embedder

def enhanced_classify_complexity(
    db: StandardDatabase, 
    question: str, 
    k: int = None,
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