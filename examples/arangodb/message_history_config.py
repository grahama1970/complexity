# src/pdf_extractor/arangodb/message_history_config.py

# Collection Names
MESSAGE_COLLECTION_NAME = "messages"

# Message Types
MESSAGE_TYPE_USER = "USER"
MESSAGE_TYPE_AGENT = "AGENT"
MESSAGE_TYPE_SYSTEM = "SYSTEM"

# Edge Collection Name
MESSAGE_EDGE_COLLECTION_NAME = "message_links"

# Graph Name
MESSAGE_GRAPH_NAME = "message_graph"

# Relationship Types
RELATIONSHIP_TYPE_NEXT = "NEXT"
RELATIONSHIP_TYPE_REFERS_TO = "REFERS_TO"
