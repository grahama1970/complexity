# src/complexity/arangodb/config.py
import os

# ArangoDB Connection Settings
ARANGO_HOST = os.environ.get("ARANGO_HOST", "http://localhost:8529")
ARANGO_USER = os.environ.get("ARANGO_USER", "root")
ARANGO_PASSWORD = os.environ.get("ARANGO_PASSWORD", "")
ARANGO_DB_NAME = os.environ.get("ARANGO_DB_NAME", "memory_bank")

# Collection & View Names
COLLECTION_NAME = "complexity"  # Updated to match your actual collection name
EDGE_COLLECTION_NAME = "relationships"
MESSAGES_COLLECTION_NAME = "messages"
VIEW_NAME = "complexity_view"  # Changed to match collection name
GRAPH_NAME = "knowledge_graph"

# Embedding Configuration
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Your actual embedding model
EMBEDDING_DIMENSIONS = 1024  # Updated to match your model's dimensions
VECTOR_INDEX_NLISTS = int(os.environ.get("VECTOR_INDEX_NLISTS", 2))

# Search Configuration
EMBEDDING_FIELD = "embedding"  # Field containing the embeddings
SEARCH_FIELDS = ["question", "answer", "tags", "title", "context"]
ALL_DATA_FIELDS_PREVIEW = ["_key", "question", "answer", "tags", "title", "context"]

# Analyzer Configuration
TEXT_ANALYZER = "text_en"
TAG_ANALYZER = "text_en"