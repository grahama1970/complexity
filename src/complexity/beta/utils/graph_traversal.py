import os
import sys
from typing import List, Dict, Any, Optional
from loguru import logger
from arango.database import StandardDatabase
from complexity.beta.utils.config import CONFIG
from complexity.beta.utils.classifier import get_EmbedderModel as get_embedder

class GraphTraversal:
    """Class for graph traversal operations."""
    
    def __init__(self, db: StandardDatabase):
        """Initialize with database connection."""
        self.db = db
    
    def find_prerequisites(self, question_id: str, max_depth: int = 2) -> List[Dict]:
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
    
    def find_related_topics(self, question_id: str, max_depth: int = 2) -> List[Dict]:
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
    
    def explain_classification(self, question: str, label: int, confidence: float) -> Dict:
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