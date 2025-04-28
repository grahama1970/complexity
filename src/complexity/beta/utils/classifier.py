from loguru import logger
import numpy as np
from complexity.beta.utils.config import CONFIG
from complexity.rag.rag_classifier import ModernBertEmbedder, DOC_PREFIX
from complexity.beta.utils.arango_setup import ensure_vector_index

def get_embedder():
    """Return the ModernBertEmbedder instance."""
    try:
        return ModernBertEmbedder(model_name=CONFIG["embedding"]["model_name"],)
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        raise

def classify_complexity(db, question: str, k: int):
    """Classify a question using weighted majority voting based on k-NN search."""
    try:
        embedder = get_embedder()
        query_embedding = embedder.embed_batch([question], prefix=DOC_PREFIX)[0]
        
        # Hallucinated
        # AQL query to retrieve k nearest neighbors with similarity scores and embeddings
        # aql_query = f"""
        #     FOR doc IN @@view
        #         SEARCH ANALYZER(VSS_COSINE_SIMILARITY(doc.{CONFIG["embedding"]["field"]}, @query_embedding), "vss")
        #         SORT VSS_COSINE_SIMILARITY(doc.{CONFIG["embedding"]["field"]}, @query_embedding) DESC
        #         LIMIT @k
        #         RETURN {{
        #             label: doc.label,
        #             similarity: VSS_COSINE_SIMILARITY(doc.{CONFIG["embedding"]["field"]}, @query_embedding),
        #             embedding: doc.{CONFIG["embedding"]["field"]}
        #         }}
        # """
        aql_query = f"""
            FOR doc IN {CONFIG["search"]["collection_name"]}
            LET score = APPROX_NEAR_COSINE(doc.embedding, @query_embedding)
            SORT score DESC
            LIMIT @k
            RETURN {{
                label: doc.label,
                similarity: score,
                embedding: doc.{CONFIG["embedding"]["field"]}
            }}
        """
        cursor = db.aql.execute(
            aql_query,
            bind_vars={
                "query_embedding": query_embedding, #.tolist(),  # Convert to list for AQL
                "k": k
            }
        )
        neighbors = list(cursor)
        
        if not neighbors:
            logger.warning(f"No neighbors found for question: {question}")
            return 0, 0.0, []
        
        # Weighted voting
        weighted_votes = {0: 0.0, 1: 0.0}
        for neighbor in neighbors:
            weighted_votes[neighbor["label"]] += neighbor["similarity"]
        
        pred_label = 1 if weighted_votes[1] > weighted_votes[0] else 0
        confidence = weighted_votes[pred_label] / (weighted_votes[0] + weighted_votes[1]) if (weighted_votes[0] + weighted_votes[1]) > 0 else 0.0
        
        return pred_label, confidence, neighbors
    
    except Exception as e:
        logger.error(f"Classification failed for question '{question}': {e}")
        raise