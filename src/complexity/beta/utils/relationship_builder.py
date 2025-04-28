import os
import sys
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import numpy as np
from arango.database import StandardDatabase
from complexity.beta.utils.config import CONFIG
from complexity.beta.utils.classifier import get_embedder

class RelationshipBuilder:
    """Class for building and managing relationships between questions."""
    
    def __init__(self, db: StandardDatabase):
        """Initialize with database connection."""
        self.db = db
        self.embedder = get_embedder()
    
    def build_semantic_relationships(self, threshold: float = None) -> int:
        """
        Build relationships based on semantic similarity between documents.
        
        Args:
            threshold: Minimum similarity to create a relationship (defaults to config value)
            
        Returns:
            Number of relationships created
        """
        threshold = threshold or 0.85  # Default threshold
        collection_name = CONFIG["search"]["collection_name"]
        embedding_field = CONFIG["embedding"]["field"]
        
        # AQL to find document pairs with high similarity
        aql = f"""
        FOR doc1 IN {collection_name}
            FOR doc2 IN {collection_name}
                FILTER doc1._id < doc2._id  // Avoid duplicates and self-relationships
                LET similarity = COSINE_SIMILARITY(doc1.{embedding_field}, doc2.{embedding_field})
                FILTER similarity >= @threshold
                SORT similarity DESC
                RETURN {{
                    from: doc1._id,
                    to: doc2._id,
                    similarity: similarity,
                    from_question: doc1.question,
                    to_question: doc2.question
                }}
        """
        
        cursor = self.db.aql.execute(aql, bind_vars={"threshold": threshold})
        candidates = list(cursor)
        logger.info(f"Found {len(candidates)} candidate relationships with similarity >= {threshold}")
        
        # Check if related_topics collection exists
        existing_collections = [c['name'] for c in self.db.collections()]
        if "related_topics" not in existing_collections:
            self.db.create_collection("related_topics", edge=True)
            logger.info("Created edge collection 'related_topics'")
        
        # Use litellm to generate a rationale for semantic relationships
        from litellm import completion
        
        # Create related_topics edges for high similarity pairs
        related_topics = self.db.collection("related_topics")
        created = 0
        
        for candidate in candidates:
            # Check if relationship already exists
            existing = list(self.db.aql.execute(
                """
                FOR edge IN related_topics FILTER edge._from == @from AND edge._to == @to RETURN edge
                """,
                bind_vars={"from": candidate["from"], "to": candidate["to"]}
            ))
            
            if not existing:
                # Generate rationale for the relationship
                rationale = "Semantically similar based on vector similarity"
                
                # For high-similarity pairs, generate a more specific rationale
                if candidate["similarity"] > 0.9:
                    try:
                        prompt = f"""
                        Analyze these two related questions:
                        
                        Question 1: {candidate['from_question']}
                        Question 2: {candidate['to_question']}
                        
                        In one brief sentence (15 words or less), explain why these questions are related or similar.
                        """
                        
                        response = completion(
                            model=CONFIG["llm"]["model"],
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=50
                        )
                        
                        # Extract text from the response
                        rationale_text = response.choices[0].message.content.strip()
                        if rationale_text:
                            rationale = rationale_text
                    except Exception as e:
                        logger.warning(f"Failed to generate rationale: {e}")
                        # Fall back to default rationale
                
                related_topics.insert({
                    "_from": candidate["from"],
                    "_to": candidate["to"],
                    "type": "semantic_similarity",
                    "strength": candidate["similarity"],
                    "auto_generated": True,
                    "rationale": rationale
                })
                created += 1
        
        logger.info(f"Created {created} new 'related_topics' relationships")
        return created
    
    def analyze_prerequisites_with_llm(self, limit: int = 100) -> int:
        """
        Use LLM to analyze potential prerequisite relationships.
        
        Args:
            limit: Maximum number of pairs to analyze
            
        Returns:
            Number of relationships created
        """
        from litellm import completion
        
        # Get document pairs with existing related_topics edges
        aql = """
        FOR edge IN related_topics
            LET from_doc = DOCUMENT(edge._from)
            LET to_doc = DOCUMENT(edge._to)
            FILTER edge.type == "semantic_similarity" AND edge.strength >= @threshold
            SORT edge.strength DESC
            LIMIT @limit
            RETURN {
                from_id: edge._from,
                to_id: edge._to,
                from_question: from_doc.question,
                to_question: to_doc.question,
                similarity: edge.strength
            }
        """
        
        cursor = self.db.aql.execute(aql, bind_vars={
            "threshold": 0.8,
            "limit": limit
        })
        candidates = list(cursor)
        
        # Check if prerequisites collection exists
        existing_collections = [c['name'] for c in self.db.collections()]
        if "prerequisites" not in existing_collections:
            self.db.create_collection("prerequisites", edge=True)
            logger.info("Created edge collection 'prerequisites'")
        
        # Use litellm for LLM API calls
        prerequisites = self.db.collection("prerequisites")
        created = 0
        
        for candidate in candidates:
            # Check if prerequisite relationship already exists in either direction
            existing = list(self.db.aql.execute(
                """
                FOR edge IN prerequisites 
                FILTER (edge._from == @from AND edge._to == @to) OR
                       (edge._from == @to AND edge._to == @from)
                RETURN edge
                """,
                bind_vars={"from": candidate["from_id"], "to": candidate["to_id"]}
            ))
            
            if existing:
                continue
            
            # Prompt LLM to analyze prerequisite relationship
            prompt = f"""
            Analyze these two questions:
            
            Question 1: {candidate['from_question']}
            Question 2: {candidate['to_question']}
            
            Is one question a prerequisite for understanding the other? A prerequisite means you need to understand one concept before you can understand the other.
            
            Answer with only one of:
            1. Question 1 is a prerequisite for Question 2
            2. Question 2 is a prerequisite for Question 1
            3. Neither is a prerequisite for the other
            
            Then, on a new line, briefly explain your reasoning in 1-2 sentences.
            """
            
            # Use litellm for completion
            try:
                response = completion(
                    model=CONFIG["llm"]["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,  # Low temperature for consistent responses
                    max_tokens=150    # Limit response length
                )
                
                # Extract text from the response
                response_text = response.choices[0].message.content.strip()
                
                # Extract explanation/rationale from response (after the first line)
                rationale = ""
                if '\n' in response_text:
                    rationale = response_text.split('\n', 1)[1].strip()
                
                # Extract relationship type from response
                if "Question 1 is a prerequisite for Question 2" in response_text:
                    # Create prerequisite edge from 1 to 2
                    prerequisites.insert({
                        "_from": candidate["from_id"],
                        "_to": candidate["to_id"],
                        "type": "prerequisite",
                        "explanation": rationale or "Understanding the first concept is necessary before grasping the second concept.",
                        "rationale": rationale or "Understanding the first concept is necessary before grasping the second concept.",
                        "llm_generated": True,
                        "confidence": 0.8  # Default confidence for LLM-determined relationships
                    })
                    created += 1
                    logger.info(f"Created prerequisite: Q1 → Q2: {candidate['from_question'][:30]} → {candidate['to_question'][:30]}")
                elif "Question 2 is a prerequisite for Question 1" in response_text:
                    # Create prerequisite edge from 2 to 1
                    prerequisites.insert({
                        "_from": candidate["to_id"],
                        "_to": candidate["from_id"],
                        "type": "prerequisite",
                        "explanation": rationale or "Understanding the second concept is necessary before grasping the first concept.",
                        "rationale": rationale or "Understanding the second concept is necessary before grasping the first concept.",
                        "llm_generated": True,
                        "confidence": 0.8
                    })
                    created += 1
                    logger.info(f"Created prerequisite: Q2 → Q1: {candidate['to_question'][:30]} → {candidate['from_question'][:30]}")
                else:
                    logger.debug(f"No prerequisite relationship detected between questions")
                    
            except Exception as e:
                logger.exception(f"LLM API call failed: {e}")
                continue
        
        logger.info(f"Created {created} new 'prerequisites' relationships using LLM analysis")
        return created
    
    def track_user_sequence_relationships(self, question_sequence: List[str]) -> int:
        """
        Track sequences of questions asked by users to infer relationships.
        
        Args:
            question_sequence: List of questions in the order they were asked
            
        Returns:
            Number of relationships affected (created or strengthened)
        """
        if len(question_sequence) < 2:
            return 0
        
        collection_name = CONFIG["search"]["collection_name"]
        
        # Get document IDs for questions
        doc_ids = []
        for question in question_sequence:
            # Find document by question text
            cursor = self.db.aql.execute(
                f"FOR doc IN {collection_name} FILTER doc.question == @question RETURN doc._id",
                bind_vars={"question": question}
            )
            results = list(cursor)
            if results:
                doc_ids.append(results[0])
            else:
                # Question not in database, can't track relationship
                logger.warning(f"Question not found in database: {question}")
                continue
        
        if len(doc_ids) < 2:
            return 0
        
        # Check if related_topics collection exists
        existing_collections = [c['name'] for c in self.db.collections()]
        if "related_topics" not in existing_collections:
            self.db.create_collection("related_topics", edge=True)
            logger.info("Created edge collection 'related_topics'")
        
        # Process sequence relationships
        affected = 0
        for i in range(len(doc_ids) - 1):
            from_id = doc_ids[i]
            to_id = doc_ids[i + 1]
            
            # Check if relationship exists
            cursor = self.db.aql.execute(
                """
                FOR edge IN related_topics
                FILTER edge._from == @from AND edge._to == @to
                RETURN edge
                """,
                bind_vars={"from": from_id, "to": to_id}
            )
            existing = list(cursor)
            
            if existing:
                # Update existing relationship
                edge = existing[0]
                new_strength = edge.get("strength", 0.5) + 0.1
                self.db.collection("related_topics").update(
                    edge["_id"],
                    {
                        "strength": min(new_strength, 1.0), 
                        "user_sequence_count": edge.get("user_sequence_count", 0) + 1,
                        "rationale": edge.get("rationale", "Frequently asked together by users")
                    }
                )
                affected += 1
            else:
                # Create new relationship with rationale
                self.db.collection("related_topics").insert({
                    "_from": from_id,
                    "_to": to_id,
                    "type": "user_sequence",
                    "strength": 0.6,  # Initial strength for user sequence relationships
                    "user_sequence_count": 1,
                    "rationale": "These questions were asked sequentially by a user, suggesting a topical relationship"
                })
                affected += 1
        
        return affected
    
    def generate_all_relationships(self):
        """Generate all types of relationships."""
        # 1. Build semantic relationships
        semantic_count = self.build_semantic_relationships()
        
        # 2. Analyze prerequisites with LLM
        prerequisite_count = self.analyze_prerequisites_with_llm()
        
        logger.info(f"Relationship generation complete: {semantic_count} semantic, {prerequisite_count} prerequisites")
        return semantic_count + prerequisite_count