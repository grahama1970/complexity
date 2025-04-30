# src/complexity/beta/utils/config.py
"""
Module Description:
Defines the central configuration dictionary (CONFIG) for the complexity project.
Loads settings from environment variables using python-dotenv for database connections,
dataset details, embedding models, search parameters, classification thresholds,
graph settings, and LLM configurations. Includes a validation function to ensure
required environment variables are set.

Links:
- python-dotenv: https://github.com/theskumar/python-dotenv
- os module: https://docs.python.org/3/library/os.html

Sample Input/Output:

- Accessing config values:
  from complexity.beta.utils.config import CONFIG
  db_host = CONFIG["arango"]["host"]
  model_name = CONFIG["embedding"]["model_name"]

- Running validation:
  python -m complexity.beta.utils.config
  (Prints validation status and exits with 0 or 1)
"""
import os
import sys # Import sys for exit codes
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
def validate_config() -> bool:
    """
    Validate that required environment variables are set.
    Returns True if valid, False otherwise. Logs errors.
    """
    validation_passed = True
    missing = []

    # Check ArangoDB config
    if not all(CONFIG["arango"].values()):
        missing = [f"ARANGO_{k.upper()}" for k, v in CONFIG["arango"].items() if not v]
        logger.error(f"Missing ArangoDB environment variables: {', '.join(missing)}")
        validation_passed = False

    # Add checks for other critical env vars if necessary, e.g., LLM API keys
    # Example: Check if LLM API key is set if type is openai
    if CONFIG["llm"]["api_type"] == "openai" and not CONFIG["llm"]["api_key_env"]:
         missing_llm_key = "OPENAI_API_KEY" # Assuming this is the env var name
         logger.error(f"Missing LLM environment variable: {missing_llm_key} (required for api_type='openai')")
         missing.append(missing_llm_key)
         validation_passed = False

    if validation_passed:
        logger.info("Configuration environment variables validated successfully.")
    else:
        logger.error(f"Validation failed. Missing variables: {', '.join(missing)}")

    return validation_passed


if __name__ == "__main__":
    logger.remove() # Remove default handler
    logger.add(sys.stderr, level="INFO") # Add back INFO level for __main__

    logger.info("Running configuration validation...")
    is_valid = validate_config()

    if is_valid:
        print("✅ VALIDATION COMPLETE - Required environment variables are set.")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Missing required environment variables. See logs for details.")
        sys.exit(1)