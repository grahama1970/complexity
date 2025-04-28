# src/pdf_extractor/arangodb/examples/tag_search_relationship_demo.py
import sys
import uuid
from loguru import logger
from arango.database import StandardDatabase
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database, ensure_edge_collection, ensure_graph
from pdf_extractor.arangodb.config import COLLECTION_NAME, RELATIONSHIP_TYPE_PREREQUISITE
from pdf_extractor.arangodb.relationship_api import add_relationship, get_relationships

# Set up logging
logger.remove()
logger.add(sys.stderr, level="INFO")

def mock_hybrid_search(db: StandardDatabase, query_text: str, tag_filters=None, top_n=5):
    """Mock hybrid search with tag filtering."""
    vertex_collection = db.collection(COLLECTION_NAME)
    
    # Build AQL query with tag filtering if provided
    tag_filter_clause = ""
    if tag_filters and isinstance(tag_filters, list) and len(tag_filters) > 0:
        tag_list = ', '.join([f'"{tag}"' for tag in tag_filters])
        tag_filter_clause = f"FILTER POSITION(doc.tags, {tag_list})"
    
    # Simple AQL query to search for documents with tag filtering
    aql = f"""
    FOR doc IN {COLLECTION_NAME}
    {tag_filter_clause}
    SORT RAND()
    LIMIT {top_n}
    RETURN doc
    """
    
    cursor = db.aql.execute(aql)
    results = []
    
    # Construct search results
    for i, doc in enumerate(cursor):
        results.append({
            "doc": doc,
            "score": 0.9 - (i * 0.1),
            "rrf_score": 0.9 - (i * 0.1)
        })
    
    return {
        "results": results,
        "
