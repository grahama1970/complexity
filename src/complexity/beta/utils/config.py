import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "arango": {
        "host": os.getenv("ARANGO_HOST", "http://localhost:8529"),
        "user": os.getenv("ARANGO_USER", "root"),
        "password": os.getenv("ARANGO_PASSWORD", "openSesame"),
        "db_name": os.getenv("ARANGO_DB_NAME", "memory_bank"),
    },
    "dataset": {
        "name": "wesley7137/question_complexity_classification",
        "split": "train",
    },
    "embedding": {
        "model_name": "BAAI/bge-large-en-v1.5",  # Changed to BGE model
        "dimensions": 1024,  # BGE model dimensions (different from nomic's 768)
        "field": "embedding",
        "batch_size": 32,
    },
    "search": {
        "collection_name": "complexity",
        "view_name": "complexity_view",
        "text_analyzer": "text_en",
        "vector_index_nlists": 18,
        "insert_batch_size": 1000,
    },
    "classification": {
        "default_k": 25,
        "confidence_threshold": 0.7,
    },
    "graph": {
        "edge_collections": ["prerequisites", "related_topics"],
        "max_traversal_depth": 2,
        "relationship_confidence_threshold": 0.7,
        "semantic_weight": 0.7,  # Weight for semantic similarity in combined score
        "graph_weight": 0.3,     # Weight for graph relationships in combined score
        "auto_relationship_threshold": 0.85  # Min similarity to automatically create relationships
    },
    
    "llm": {
        "api_type": "openai",  # or "anthropic", "local", etc.
        "model": "gpt-4o-mini",  # or "claude-3-opus", etc.
        "api_key_env": os.getenv('OPENAI_API_KEY'),  # Environment variable name for API key
        "temperature": 0.2,  # Default temperature for LLM calls
        "max_tokens": 150,  # Default max tokens for LLM calls
        "litellm_cache": True  # Enable caching to reduce API costs
    }
}

# Validate environment
def validate_config():
    """Validate that required environment variables are set."""
    if not all(CONFIG["arango"].values()):
        missing = [k for k, v in CONFIG["arango"].items() if not v]
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    logger.info("Configuration validated successfully")

if __name__ == "__main__":
    validate_config()