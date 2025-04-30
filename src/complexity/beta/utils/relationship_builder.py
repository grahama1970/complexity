# src/complexity/beta/utils/relationship_builder.py
"""
Module Description:
Provides the RelationshipBuilder class for creating and managing relationships
(semantic similarity, prerequisites) between question documents stored in ArangoDB.
Uses semantic similarity calculations and LLM analysis to infer relationships.

Links:
- python-arango Driver: https://python-arango.readthedocs.io/en/latest/
- ArangoDB Manual: https://www.arangodb.com/docs/stable/
- Loguru: https://loguru.readthedocs.io/en/stable/
- litellm: https://litellm.ai/

Sample Input/Output:

- RelationshipBuilder(db).build_semantic_relationships(threshold=0.85):
  - Input: Optional similarity threshold.
  - Output: Integer count of new 'related_topics' edges created.

- RelationshipBuilder(db).analyze_prerequisites_with_llm(limit=100):
  - Input: Optional limit on pairs to analyze.
  - Output: Integer count of new 'prerequisites' edges created.

- Running main validation:
  python -m complexity.beta.utils.relationship_builder
  (Connects to DB, runs relationship generation, validates execution, exits 0/1)
"""
import os
import sys
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import numpy as np
from arango.database import StandardDatabase

from complexity.beta.utils.config import CONFIG
# Corrected import path for get_EmbedderModel (renamed locally to get_embedder)
from complexity.beta.utils.arango_setup import get_EmbedderModel as get_embedder
# Import necessary setup functions for standalone execution
from complexity.beta.utils.arango_setup import connect_arango, ensure_database, ensure_collection, ensure_edge_collections

# Try to import PyTorch and our semantic search utilities
try:
    import torch
    from complexity.beta.utils.pytorch_semantic_search import load_documents_from_arango, pytorch_enhanced_search
    has_pytorch = True
    logger.info("PyTorch is available, using GPU-accelerated semantic search")
except ImportError:
    has_pytorch = False
    logger.info("PyTorch not available, will use AQL for semantic search")


class RelationshipBuilder:
    """Class for building and managing relationships between questions."""
    
    def __init__(self, db: StandardDatabase):
        """Initialize with database connection."""
        self.db = db
        self.embedder = get_embedder()
    
    def build_semantic_relationships(self, threshold: Optional[float] = None, filter_conditions: Optional[str] = None) -> int:
        """
        Build relationships based on semantic similarity between documents.
        Automatically selects the optimal approach based on query complexity:
        1. Direct ArangoDB: Simple queries (milliseconds)
        2. Two-stage ArangoDB: Filtered queries (milliseconds) 
        3. PyTorch: Complex relationship building (seconds)
        
        Args:
            threshold: Minimum similarity to create a relationship (defaults to config value)
            filter_conditions: Optional AQL filter conditions
            
        Returns:
            Number of relationships created
        """
        threshold = threshold or 0.85  # Default threshold
        
        # Determine if query requires relationship building or complex nesting
        requires_nesting = self._query_requires_nesting(filter_conditions)
        
        # Select appropriate approach based on query complexity
        if requires_nesting and has_pytorch:
            logger.info("Query requires relationship building - using PyTorch approach")
            return self.build_semantic_relationships_pytorch(threshold, filter_conditions=filter_conditions)
        elif filter_conditions and not requires_nesting:
            logger.info("Query requires filtering but not nesting - using two-stage ArangoDB approach")
            return self._build_relationships_two_stage_arangodb(threshold, filter_conditions)
        else:
            logger.info("Simple query - using direct ArangoDB approach")
            return self._build_relationships_direct_arangodb(threshold)
    
    def _query_requires_nesting(self, filter_conditions: Optional[str]) -> bool:
        """
        Determine if a query requires the PyTorch approach for nesting or relationship building.
        
        Args:
            filter_conditions: The filter conditions to analyze
            
        Returns:
            True if the query requires nesting capabilities, False otherwise
        """
        # If no filter conditions, no nesting required
        if not filter_conditions:
            return False
            
        # Check for specific patterns that require nesting
        nesting_indicators = [
            "FOR", "LET", "COLLECT", "WITH",  # AQL keywords that suggest nesting
            "IN",  # Array containment often requires nesting
            "LENGTH", "CONTAINS", "LIKE",  # Complex operations
            "?", ":",  # Ternary operators
        ]
        
        # Check if any nesting indicators are present
        for indicator in nesting_indicators:
            if indicator in filter_conditions:
                return True
                
        return False
    
    def _build_relationships_direct_arangodb(self, threshold: float) -> int:
        """
        Build relationships using direct ArangoDB approach for simple queries.
        Used when no filtering is required.
        
        Args:
            threshold: Minimum similarity threshold
            
        Returns:
            Number of relationships created
        """
        collection_name = CONFIG["search"]["collection_name"]
        embedding_field = CONFIG["embedding"]["field"]
        
        # Direct ArangoDB query for vector similarity
        aql = f"""
        FOR doc1 IN {collection_name}
            LET neighbors = (
                FOR doc2 IN {collection_name}
                    FILTER doc1._id != doc2._id  // Avoid self-relationships
                    LET similarity = APPROX_NEAR_COSINE(doc1.{embedding_field}, doc2.{embedding_field})
                    FILTER similarity >= @threshold
                    SORT similarity DESC
                    LIMIT 10  // Limit neighbors per document
                    RETURN {{
                        id: doc2._id,
                        similarity: similarity,
                        question: doc2.question
                    }}
            )
            FILTER LENGTH(neighbors) > 0  // Only return docs with at least one neighbor
            RETURN {{
                from: doc1._id,
                from_question: doc1.question,
                neighbors: neighbors
            }}
        """
        
        try:
            cursor = self.db.aql.execute(aql, bind_vars={"threshold": threshold})
            candidates = list(cursor)
            logger.info(f"Found {len(candidates)} candidate relationships with similarity >= {threshold}")
            
            # Check if related_topics collection exists
            existing_collections = [c['name'] for c in self.db.collections()]
            if "related_topics" not in existing_collections:
                self.db.create_collection("related_topics", edge=True)
                logger.info("Created edge collection 'related_topics'")
            
            # Create related_topics edges
            related_topics = self.db.collection("related_topics")
            created = 0
            
            # Use litellm to generate a rationale for semantic relationships
            from litellm import completion
            
            for candidate in candidates:
                from_id = candidate["from"]
                from_question = candidate["from_question"]
                
                for neighbor in candidate["neighbors"]:
                    to_id = neighbor["id"]
                    similarity = neighbor["similarity"]
                    to_question = neighbor.get("question", "")
                    
                    # Check if relationship already exists
                    existing = list(self.db.aql.execute(
                        """
                        FOR edge IN related_topics 
                        FILTER edge._from == @from AND edge._to == @to 
                        RETURN edge
                        """,
                        bind_vars={"from": from_id, "to": to_id}
                    ))
                    
                    if not existing:
                        # Generate rationale for the relationship
                        rationale = "Semantically similar based on vector similarity (direct ArangoDB)"
                        
                        # For high-similarity pairs, generate a more specific rationale
                        if similarity > 0.9:
                            try:
                                prompt = f"""
                                Analyze these two related questions:
                                
                                Question 1: {from_question}
                                Question 2: {to_question}
                                
                                In one brief sentence (15 words or less), explain why these questions are related or similar.
                                """
                                
                                response = completion(
                                    model=CONFIG["llm"]["model"],
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.3,
                                    max_tokens=50
                                )
                                
                                # Extract text from the response
                                choice = response.choices[0] if response.choices else None
                                message = choice.message if choice else None
                                content = message.content if message else None
                                rationale_text = content.strip() if content else None
                                if rationale_text:
                                    rationale = rationale_text
                            except Exception as e:
                                logger.warning(f"Failed to generate rationale: {e}")
                                # Fall back to default rationale
                        
                        # Create new relationship
                        related_topics.insert({
                            "_from": from_id,
                            "_to": to_id,
                            "type": "semantic_similarity",
                            "strength": similarity,
                            "auto_generated": True,
                            "rationale": rationale
                        })
                        created += 1
            
            logger.info(f"Created {created} new 'related_topics' relationships using direct ArangoDB search")
            return created
            
        except Exception as e:
            logger.error(f"Direct ArangoDB search failed: {str(e)}")
            return 0
    
    def _build_relationships_two_stage_arangodb(self, threshold: float, filter_conditions: str) -> int:
        """
        Build relationships using the two-stage ArangoDB approach for filtered queries.
        Accounts for ArangoDB limitation: APPROX_NEAR_COSINE cannot be combined with 
        complex filtering in a single query (Issue #21690).
        
        Args:
            threshold: Minimum similarity threshold
            filter_conditions: AQL filter conditions
            
        Returns:
            Number of relationships created
        """
        collection_name = CONFIG["search"]["collection_name"]
        embedding_field = CONFIG["embedding"]["field"]
        
        # Stage 1: Get IDs of documents matching filter conditions
        filter_query = f"""
        FOR doc IN {collection_name}
            FILTER {filter_conditions}
            RETURN doc._id
        """
        
        try:
            # Get filtered document IDs
            cursor = self.db.aql.execute(filter_query)
            filtered_ids = list(cursor)
            
            if not filtered_ids:
                logger.warning("No documents match the filter criteria")
                return 0
                
            logger.info(f"Found {len(filtered_ids)} documents matching filter conditions")
            
            # Stage 2: Run vector search on ALL documents, but only return 
            # similarities for pairs where BOTH documents are in filtered_ids
            # This addresses the limitation in ArangoDB (Issue #21690)
            vector_query = f"""
            FOR doc1 IN {collection_name}
                FILTER doc1._id IN @filtered_ids
                LET neighbors = (
                    FOR doc2 IN {collection_name}
                        FILTER doc1._id != doc2._id  // Avoid self-relationships
                        FILTER doc2._id IN @filtered_ids  // Only consider filtered docs
                        LET similarity = APPROX_NEAR_COSINE(doc1.{embedding_field}, doc2.{embedding_field})
                        FILTER similarity >= @threshold
                        SORT similarity DESC
                        LIMIT 10  // Limit neighbors per document
                        RETURN {{
                            id: doc2._id,
                            similarity: similarity,
                            question: doc2.question
                        }}
                )
                FILTER LENGTH(neighbors) > 0  // Only return docs with at least one neighbor
                RETURN {{
                    from: doc1._id,
                    from_question: doc1.question,
                    neighbors: neighbors
                }}
            """
            
            cursor = self.db.aql.execute(
                vector_query, 
                bind_vars={
                    "filtered_ids": filtered_ids,
                    "threshold": threshold
                }
            )
            
            candidates = list(cursor)
            logger.info(f"Found {len(candidates)} candidate relationships with similarity >= {threshold}")
            
            # Check if related_topics collection exists
            existing_collections = [c['name'] for c in self.db.collections()]
            if "related_topics" not in existing_collections:
                self.db.create_collection("related_topics", edge=True)
                logger.info("Created edge collection 'related_topics'")
            
            # Process candidates and create relationships
            related_topics = self.db.collection("related_topics")
            created = 0
            
            # Use litellm to generate a rationale for semantic relationships
            from litellm import completion
            
            for candidate in candidates:
                from_id = candidate["from"]
                from_question = candidate["from_question"]
                
                for neighbor in candidate["neighbors"]:
                    to_id = neighbor["id"]
                    similarity = neighbor["similarity"]
                    to_question = neighbor.get("question", "")
                    
                    # Check if relationship already exists
                    existing = list(self.db.aql.execute(
                        """
                        FOR edge IN related_topics 
                        FILTER edge._from == @from AND edge._to == @to 
                        RETURN edge
                        """,
                        bind_vars={"from": from_id, "to": to_id}
                    ))
                    
                    if not existing:
                        # Generate rationale for the relationship
                        rationale = "Semantically similar based on vector similarity (two-stage ArangoDB)"
                        
                        # For high-similarity pairs, generate a more specific rationale
                        if similarity > 0.9:
                            try:
                                prompt = f"""
                                Analyze these two related questions:
                                
                                Question 1: {from_question}
                                Question 2: {to_question}
                                
                                In one brief sentence (15 words or less), explain why these questions are related or similar.
                                """
                                
                                response = completion(
                                    model=CONFIG["llm"]["model"],
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.3,
                                    max_tokens=50
                                )
                                
                                # Extract text from the response
                                choice = response.choices[0] if response.choices else None
                                message = choice.message if choice else None
                                content = message.content if message else None
                                rationale_text = content.strip() if content else None
                                if rationale_text:
                                    rationale = rationale_text
                            except Exception as e:
                                logger.warning(f"Failed to generate rationale: {e}")
                                # Fall back to default rationale
                        
                        # Create new relationship
                        related_topics.insert({
                            "_from": from_id,
                            "_to": to_id,
                            "type": "semantic_similarity",
                            "strength": similarity,
                            "auto_generated": True,
                            "rationale": rationale
                        })
                        created += 1
            
            logger.info(f"Created {created} new 'related_topics' relationships using two-stage ArangoDB search")
            return created
            
        except Exception as e:
            logger.error(f"Two-stage ArangoDB search failed: {str(e)}")
            if has_pytorch:
                logger.info("Falling back to PyTorch implementation")
                return self.build_semantic_relationships_pytorch(threshold, filter_conditions=filter_conditions)
            return 0
    
    def build_semantic_relationships_pytorch(self, threshold: Optional[float] = None, 
                                      batch_size: int = 128, 
                                      filter_conditions: Optional[str] = None) -> int:
        """
        Build relationships based on semantic similarity using optimized PyTorch-based search.
        Handles complex filtering scenarios that ArangoDB's APPROX_NEAR_COSINE may not support.
        
        Args:
            threshold: float, minimum similarity threshold (defaults to config value)
            batch_size: int, batch size for processing
            filter_conditions: str, optional AQL filter conditions (e.g., "doc.category == 'math'")
            
        Returns:
            int: Number of relationships created
        """
        threshold = threshold or 0.85  # Default threshold
        collection_name = CONFIG["search"]["collection_name"]
        embedding_field = CONFIG["embedding"]["field"]
        
        logger.info(f"Starting PyTorch-based similarity search with threshold {threshold}")
        if filter_conditions:
            logger.info(f"Applying filter conditions: {filter_conditions}")
        
        # Load documents from ArangoDB with filtering
        embeddings, ids, questions, dimension = load_documents_from_arango(
            self.db, collection_name, embedding_field, 
            batch_size=1000, filter_conditions=filter_conditions
        )
        
        if embeddings is None:
            logger.error("Failed to load documents. Aborting similarity search.")
            return 0
        
        # Check if GPU is available
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            logger.info("GPU detected, using GPU acceleration")
        else:
            logger.info("No GPU detected, using CPU")
        
        # Perform similarity search using our optimized function
        results, elapsed_time = pytorch_enhanced_search(
            embeddings=embeddings,
            ids=ids,
            questions=questions,
            threshold=threshold,
            batch_size=batch_size,
            fp16=has_gpu,  # Use FP16 only with GPU
            cuda_streams=has_gpu,  # Use CUDA streams only with GPU
            use_ann=(len(embeddings) > 5000),  # Only use ANN for larger datasets
            nlist=min(4096, len(embeddings) // 39),  # FAISS rule of thumb
            nprobe=min(256, len(embeddings) // 39 // 4)  # FAISS rule of thumb
        )
        
        # Check if related_topics collection exists
        existing_collections = [c['name'] for c in self.db.collections()]
        if "related_topics" not in existing_collections:
            self.db.create_collection("related_topics", edge=True)
            logger.info("Created edge collection 'related_topics'")
        
        # Create related_topics edges for high similarity pairs
        related_topics = self.db.collection("related_topics")
        created = 0
        
        # Use litellm to generate a rationale for semantic relationships
        from litellm import completion
        
        for result in results:
            from_id = result["from"]
            from_question = result["from_question"]
            neighbors = result["neighbors"]
            
            for neighbor in neighbors:
                to_id = neighbor["id"]
                similarity = neighbor["similarity"]
                
                # Check if relationship already exists
                existing = list(self.db.aql.execute(
                    """
                    FOR edge IN related_topics 
                    FILTER edge._from == @from AND edge._to == @to 
                    RETURN edge
                    """,
                    bind_vars={"from": from_id, "to": to_id}
                ))
                
                if not existing:
                    # Generate rationale for the relationship
                    rationale = "Semantically similar based on vector similarity (PyTorch)"
                    
                    # For high-similarity pairs, generate a more specific rationale
                    if similarity > 0.9:
                        try:
                            prompt = f"""
                            Analyze these two related questions:
                            
                            Question 1: {from_question}
                            Question 2: {neighbor['question']}
                            
                            In one brief sentence (15 words or less), explain why these questions are related or similar.
                            """
                            
                            response = completion(
                                model=CONFIG["llm"]["model"],
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                                max_tokens=50
                            )
                            
                            # Extract text from the response
                            choice = response.choices[0] if response.choices else None
                            message = choice.message if choice else None
                            content = message.content if message else None
                            rationale_text = content.strip() if content else None
                            if rationale_text:
                                rationale = rationale_text
                        except Exception as e:
                            logger.warning(f"Failed to generate rationale: {e}")
                            # Fall back to default rationale
                    
                    # Create new relationship
                    related_topics.insert({
                        "_from": from_id,
                        "_to": to_id,
                        "type": "semantic_similarity",
                        "strength": similarity,
                        "auto_generated": True,
                        "rationale": rationale
                    })
                    created += 1
        
        logger.info(f"Created {created} new 'related_topics' relationships using PyTorch-based search")
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
                # Safer access to potentially nested response structure
                choice = response.choices[0] if response.choices else None
                message = choice.message if choice else None
                response_text = message.content.strip() if message and message.content else ""
                
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
                # Corrected update call: pass the document dict itself, then the update data
                update_data = {
                    "strength": min(new_strength, 1.0),
                    "user_sequence_count": edge.get("user_sequence_count", 0) + 1,
                    "rationale": edge.get("rationale", "Frequently asked together by users")
                }
                self.db.collection("related_topics").update(
                    edge, # Pass the full edge document dictionary
                    update_data,
                    check_rev=False # Set check_rev to False as we don't have _rev
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
    
    def generate_all_relationships(self, filter_conditions: Optional[str] = None):
        """
        Generate all types of relationships.
        
        Args:
            filter_conditions: Optional AQL filter conditions for document selection
            
        Returns:
            int: Total number of relationships created
        """
        # 1. Build semantic relationships with optional filtering
        semantic_count = self.build_semantic_relationships(filter_conditions=filter_conditions)
        
        # 2. Analyze prerequisites with LLM
        prerequisite_count = self.analyze_prerequisites_with_llm()
        
        logger.info(f"Relationship generation complete: {semantic_count} semantic, {prerequisite_count} prerequisites")
        return semantic_count + prerequisite_count


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    validation_passed = True
    validation_failures = {}
    db_instance: Optional[StandardDatabase] = None
    total_relationships = -1 # Default to -1 to indicate it didn't run/complete

    try:
        logger.info("Connecting to ArangoDB for validation...")
        client = connect_arango()
        db_instance = ensure_database(client)
        logger.info(f"Connected to database: {db_instance.name}")

        # Ensure prerequisites (collections) exist
        logger.info("Ensuring necessary DB collections exist...")
        try:
            ensure_collection(db_instance) # Ensures 'complexity' collection
            ensure_edge_collections(db_instance) # Ensures 'prerequisites', 'related_topics'
            logger.info("DB collection checks completed.")
        except Exception as setup_err:
             logger.warning(f"Error during DB setup check: {setup_err}. Relationship building might fail.")
             # Allow continuing but expect potential failures

        # --- Initialize and Run Relationship Builder ---
        logger.info("Initializing RelationshipBuilder...")
        builder = RelationshipBuilder(db_instance)

        logger.info("Running generate_all_relationships...")
        # Note: This can be time-consuming and may make LLM calls if thresholds are met
        total_relationships = builder.generate_all_relationships()
        logger.info(f"generate_all_relationships completed, total count: {total_relationships}")

        # --- Validation ---
        # Basic validation: Check if the function ran without error and returned a non-negative integer.
        if total_relationships < 0:
            validation_passed = False
            validation_failures["return_value"] = {"expected": "non-negative integer", "actual": total_relationships}
            logger.error(f"generate_all_relationships returned unexpected value: {total_relationships}")
        else:
            logger.info("generate_all_relationships returned a valid count.")


    except Exception as e:
        validation_passed = False
        validation_failures["runtime_error"] = str(e)
        logger.exception(f"Validation failed with runtime error: {e}")

    # --- Final Reporting ---
    if validation_passed:
        print(f"✅ VALIDATION COMPLETE - RelationshipBuilder executed successfully (Count: {total_relationships}).")
        logger.success("Standalone execution and validation successful.")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Issues detected during RelationshipBuilder execution or validation.")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            if isinstance(details, dict):
                 print(f"  - {field}: Expected: {details.get('expected', 'N/A')}, Got: {details.get('actual', 'N/A')}")
            else:
                 print(f"  - {field}: {details}")
        logger.error("Standalone execution and validation failed.")
        sys.exit(1)