# src/complexity/beta/utils/graph_traversal.py
"""
Module Description:
Provides the GraphTraversal class for performing graph-based operations in ArangoDB,
such as finding prerequisites, related topics, and generating explanations for
question complexity classifications based on graph structure.

Links:
- python-arango Driver: https://python-arango.readthedocs.io/en/latest/
- ArangoDB AQL Traversal: https://www.arangodb.com/docs/stable/aql/graphs-traversals.html
- Loguru: https://loguru.readthedocs.io/en/stable/

Sample Input/Output:

- GraphTraversal(db).find_prerequisites(question_id="complexity/123", max_depth=2):
  - Input: Document ID string, optional max depth.
  - Output: List[Dict[str, Any]] containing prerequisite documents and metadata.
    Example: [{'question': 'Define variable', 'label': 0, 'distance': 1, ...}, ...]

- GraphTraversal(db).explain_classification(question="What is recursion?", label=1, confidence=0.9):
  - Input: Question string, predicted label, confidence score.
  - Output: Dict[str, Any] containing explanation details.
    Example: {'explanation': 'Requires understanding...', 'prerequisites': [...], ...}

- Running main validation:
  python -m complexity.beta.utils.graph_traversal
  (Connects to DB, runs sample traversals/explanations, validates outputs, exits 0/1)
"""
import os
import sys
from typing import List, Dict, Any, Optional
from loguru import logger
from arango.database import StandardDatabase

from complexity.beta.utils.config import CONFIG
# Corrected import path for get_EmbedderModel
from complexity.beta.utils.arango_setup import get_EmbedderModel as get_embedder
# Import necessary setup functions for standalone execution
from complexity.beta.utils.arango_setup import connect_arango, ensure_database, ensure_collection, ensure_edge_collections, ensure_graph

class GraphTraversal:
    """Class for graph traversal operations."""
    
    def __init__(self, db: StandardDatabase):
        """Initialize with database connection."""
        self.db = db
    
    def find_prerequisites(self, question_id: str, max_depth: int = 2) -> List[Dict[str, Any]]: # Added Dict type args
        """
        Find prerequisites for a given question.
        
        Args:
            question_id: Document ID of the question
            max_depth: Maximum traversal depth
            
        Returns:
            List of prerequisite documents with distance
        """
        aql = """
        FOR vertex, edge, path IN 1..@max_depth OUTBOUND @start_id prerequisites
            RETURN {
                question: vertex.question,
                label: vertex.label,
                distance: LENGTH(path.edges),
                id: vertex._id,
                rationale: edge.rationale
            }
        """
        
        cursor = self.db.aql.execute(aql, bind_vars={
            "start_id": question_id,
            "max_depth": max_depth
        })
        
        return list(cursor)
    
    def find_related_topics(self, question_id: str, max_depth: int = 2) -> List[Dict[str, Any]]: # Added Dict type args
        """
        Find related topics for a given question.
        
        Args:
            question_id: Document ID of the question
            max_depth: Maximum traversal depth
            
        Returns:
            List of related documents with distance and relationship type
        """
        aql = """
        FOR vertex, edge, path IN 1..@max_depth ANY @start_id related_topics
            RETURN {
                question: vertex.question,
                label: vertex.label,
                distance: LENGTH(path.edges),
                relationship_type: edge.type,
                strength: edge.strength,
                rationale: edge.rationale,
                id: vertex._id
            }
        """
        
        cursor = self.db.aql.execute(aql, bind_vars={
            "start_id": question_id,
            "max_depth": max_depth
        })
        
        return list(cursor)
    
    def get_question_id(self, question: str) -> Optional[str]:
        """
        Get document ID for a question text.
        
        Args:
            question: Question text
            
        Returns:
            Document ID if found, None otherwise
        """
        collection_name = CONFIG["search"]["collection_name"]
        cursor = self.db.aql.execute(
            f"FOR doc IN {collection_name} FILTER doc.question == @question RETURN doc._id",
            bind_vars={"question": question}
        )
        results = list(cursor)
        return results[0] if results else None
    
    def get_most_similar_id(self, question: str) -> Optional[str]:
        """
        Get document ID for most similar question if exact match not found.
        
        Args:
            question: Question text
            
        Returns:
            Document ID of most similar question, None if no good matches
        """
        embedder = get_embedder()
        query_embedding = embedder.embed_batch([question])[0]
        
        collection_name = CONFIG["search"]["collection_name"]
        embedding_field = CONFIG["embedding"]["field"]
        
        aql = f"""
        FOR doc IN {collection_name}
            LET similarity = COSINE_SIMILARITY(doc.{embedding_field}, @embedding)
            FILTER similarity > 0.8
            SORT similarity DESC
            LIMIT 1
            RETURN doc._id
        """
        
        cursor = self.db.aql.execute(aql, bind_vars={"embedding": query_embedding})
        results = list(cursor)
        return results[0] if results else None
    
    def explain_classification(self, question: str, label: int, confidence: float) -> Dict[str, Any]: # Added Dict type args
        """
        Provide an explanation for classification using graph relationships.
        
        Args:
            question: Question text
            label: Classification label (0=Simple, 1=Complex)
            confidence: Classification confidence
            
        Returns:
            Dictionary with explanation details
        """
        # First try to find exact question
        question_id = self.get_question_id(question)
        
        # If not found, try to find similar question
        if not question_id:
            question_id = self.get_most_similar_id(question)
            if not question_id:
                return {
                    "explanation": "Classification based on semantic similarity only.",
                    "factor": "semantic_only",
                    "confidence": confidence
                }
        
        # Get prerequisites if complex, or check if there are prerequisites
        if label == 1:  # Complex
            prerequisites = self.find_prerequisites(question_id)
            if prerequisites:
                # Extract rationales from prerequisites
                rationales = [p.get("rationale", "Prerequisite relationship") for p in prerequisites[:3]]
                
                return {
                    "explanation": f"This question is classified as Complex because it requires understanding {len(prerequisites)} prerequisite concepts.",
                    "prerequisites": prerequisites[:3],  # List a few key prerequisites
                    "rationales": rationales,
                    "factor": "has_prerequisites",
                    "confidence": confidence
                }
        
        # Get related topics
        related = self.find_related_topics(question_id)
        related_by_label = {"simple": 0, "complex": 0}
        
        for r in related:
            if r.get("label") == 0:
                related_by_label["simple"] += 1
            else:
                related_by_label["complex"] += 1
        
        if label == 1 and related_by_label["complex"] > related_by_label["simple"]:
            complex_topics = [r for r in related[:3] if r.get("label") == 1]
            rationales = [r.get("rationale", "Related topic") for r in complex_topics]
            
            return {
                "explanation": f"This question is classified as Complex because it is related to {related_by_label['complex']} complex topics.",
                "related_topics": complex_topics,
                "rationales": rationales,
                "factor": "related_to_complex",
                "confidence": confidence
            }
        elif label == 0 and related_by_label["simple"] > related_by_label["complex"]:
            simple_topics = [r for r in related[:3] if r.get("label") == 0]
            rationales = [r.get("rationale", "Related topic") for r in simple_topics]
            
            return {
                "explanation": f"This question is classified as Simple because it is related to {related_by_label['simple']} simple topics.",
                "related_topics": simple_topics,
                "rationales": rationales,
                "factor": "related_to_simple",
                "confidence": confidence
            }
        
        return {
            "explanation": f"This question is classified based primarily on semantic features, with some influence from {len(related)} related questions.",
            "factor": "primarily_semantic",
            "confidence": confidence
        }


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    validation_passed = True
    validation_failures = {}

    # --- Expected Values ---
    # Use a question likely to exist if data has been loaded
    TEST_QUESTION = "What is the capital of France?"
    EXPECTED_LABEL = 0 # Assuming simple
    EXPECTED_CONFIDENCE = 0.0 # Placeholder, actual confidence varies

    db_instance: Optional[StandardDatabase] = None

    try:
        logger.info("Connecting to ArangoDB for validation...")
        client = connect_arango()
        db_instance = ensure_database(client)
        logger.info(f"Connected to database: {db_instance.name}")

        # Ensure prerequisites (collection, graph) exist
        logger.info("Ensuring necessary DB structures exist...")
        try:
            ensure_collection(db_instance)
            ensure_edge_collections(db_instance)
            ensure_graph(db_instance)
            logger.info("DB structure checks completed.")
        except Exception as setup_err:
             logger.warning(f"Error during DB setup check: {setup_err}. Traversal might fail or return empty.")

        traversal = GraphTraversal(db_instance)

        # --- Test get_question_id / get_most_similar_id ---
        logger.info(f"Attempting to find ID for: '{TEST_QUESTION}'")
        question_id = traversal.get_question_id(TEST_QUESTION)
        if not question_id:
            logger.warning(f"Exact match not found, trying similar match for: '{TEST_QUESTION}'")
            question_id = traversal.get_most_similar_id(TEST_QUESTION)

        if not question_id:
            # Cannot proceed with graph traversals if no ID found
            logger.warning(f"Could not find document ID for '{TEST_QUESTION}' or similar. Skipping graph traversal tests.")
            # Don't fail validation here, as it depends on data being present
        else:
            logger.info(f"Found document ID: {question_id}")

            # --- Test find_prerequisites ---
            logger.info(f"Finding prerequisites for {question_id}...")
            prereqs = traversal.find_prerequisites(question_id)
            if not isinstance(prereqs, list):
                validation_passed = False
                validation_failures["prereqs_type"] = {"expected": "list", "actual": type(prereqs).__name__}
                logger.error("find_prerequisites did not return a list.")
            else:
                logger.info(f"Found {len(prereqs)} prerequisites.")

            # --- Test find_related_topics ---
            logger.info(f"Finding related topics for {question_id}...")
            related = traversal.find_related_topics(question_id)
            if not isinstance(related, list):
                validation_passed = False
                validation_failures["related_type"] = {"expected": "list", "actual": type(related).__name__}
                logger.error("find_related_topics did not return a list.")
            else:
                logger.info(f"Found {len(related)} related topics.")

        # --- Test explain_classification ---
        logger.info(f"Generating explanation for: '{TEST_QUESTION}'")
        explanation = traversal.explain_classification(TEST_QUESTION, EXPECTED_LABEL, EXPECTED_CONFIDENCE)
        if not isinstance(explanation, dict):
            validation_passed = False
            validation_failures["explanation_type"] = {"expected": "dict", "actual": type(explanation).__name__}
            logger.error("explain_classification did not return a dict.")
        elif "explanation" not in explanation or "factor" not in explanation:
             validation_passed = False
             validation_failures["explanation_keys"] = {"expected": "keys 'explanation', 'factor'", "actual": "Missing keys"}
             logger.error("Explanation dictionary missing required keys.")
        else:
            logger.info(f"Explanation generated: Factor='{explanation.get('factor')}'")


    except Exception as e:
        validation_passed = False
        validation_failures["runtime_error"] = str(e)
        logger.exception(f"Validation failed with runtime error: {e}")

    # --- Final Reporting ---
    if validation_passed:
        print("✅ VALIDATION COMPLETE - GraphTraversal methods executed and returned expected types.")
        logger.success("Standalone execution and validation successful.")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Issues detected during GraphTraversal validation.")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            if isinstance(details, dict):
                 print(f"  - {field}: Expected: {details.get('expected', 'N/A')}, Got: {details.get('actual', 'N/A')}")
            else:
                 print(f"  - {field}: {details}")
        logger.error("Standalone execution and validation failed.")
        sys.exit(1)