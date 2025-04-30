# src/pdf_extractor/arangodb/examples/agent_workflow.py
import sys
import uuid
from loguru import logger
from complexity.arangodb.arango_setup_unknown import connect_arango, ensure_database, ensure_edge_collection, ensure_graph
from complexity.arangodb.agent_decision import (
    evaluate_relationship_need, create_strategic_relationship, mock_hybrid_search
)
from complexity.arangodb.advanced_query_solution import solve_query
from complexity.arangodb.config import COLLECTION_NAME, EDGE_COLLECTION_NAME

logger.remove()
logger.add(sys.stderr, level="INFO")

def create_test_documents(db):
    """Create test documents."""
    test_id = uuid.uuid4().hex[:8]
    doc1_key = f"test_doc1_{test_id}"
    doc2_key = f"test_doc2_{test_id}"
    
    vertex_collection = db.collection(COLLECTION_NAME)
    vertex_collection.insert({
        "_key": doc1_key,
        "problem": "Slow database queries",
        "content": "Basic database optimization",
        "tags": ["database", "performance"]
    })
    vertex_collection.insert({
        "_key": doc2_key,
        "problem": "API response time",
        "content": "Advanced database techniques",
        "tags": ["database", "performance"]
    })
    return doc1_key, doc2_key

def test_relationship_impact(db, query, doc1_key, doc2_key):
    """Test if relationship improves query results."""
    # Auto inputs for testing
    auto_inputs = {
        "rationale": "This relationship is essential because learning basic database optimization principles is a prerequisite to understanding and implementing advanced database techniques. The former builds foundational knowledge while the latter applies it to complex scenarios.",
        "confidence_score": 1  # 1 = best/most confident
    }
    
    before = solve_query(db, query)
    relationship = create_strategic_relationship(db, doc1_key, doc2_key, query, auto_inputs)
    if not relationship:
        return False, "Failed to create relationship"
    after = solve_query(db, query)
    improved = len(after["results"]) > len(before["results"])
    return improved, f"Results: {len(before['results'])} before, {len(after['results'])} after"

def run_workflow_example():
    """Run agent workflow."""
    client = connect_arango()
    db = ensure_database(client)
    ensure_edge_collection(db)
    ensure_graph(db)
    
    doc1_key, doc2_key = create_test_documents(db)
    
    query = "Optimize database and API performance"
    logger.info(f"Query: {query}")
    
    # Use mock_hybrid_search from agent_decision in this example
    search_results = mock_hybrid_search(db, query, top_n=5)
    logger.info(f"Search found {len(search_results.get('results', []))} results")
    
    need = evaluate_relationship_need(db, query)
    logger.info(f"Need score: {need['need_score']}/10")
    
    improved = False
    impact_message = "No relationships created"
    
    if need["need_score"] >= 3:  # Lower threshold for example
        improved, impact_message = test_relationship_impact(db, query, doc1_key, doc2_key)
        if not improved:
            logger.error(f"Relationship did not improve results: {impact_message}")
    
    final_result = solve_query(db, query)
    logger.info(f"Solved with attempt {final_result.get('attempt', 0)}")
    
    return {
        "documents": [doc1_key, doc2_key], 
        "relationships": [], 
        "impact": impact_message,
        "final_results": len(final_result.get("results", []))
    }

def cleanup_resources(db, resources):
    """Clean up resources."""
    vertex_collection = db.collection(COLLECTION_NAME)
    edge_collection = db.collection(EDGE_COLLECTION_NAME)
    
    for doc_key in resources.get("documents", []):
        if doc_key:
            # Find and delete all relationships for this document
            try:
                for direction in ["OUTBOUND", "INBOUND"]:
                    aql = f"""
                    FOR v, e IN 1..1 {direction} @start_vertex GRAPH @graph_name
                    RETURN e
                    """
                    bind_vars = {
                        "start_vertex": f"{COLLECTION_NAME}/{doc_key}", 
                        "graph_name": "document_graph"
                    }
                    edges = list(db.aql.execute(aql, bind_vars=bind_vars))
                    for edge in edges:
                        edge_collection.delete(edge["_key"])
            except Exception as e:
                logger.error(f"Error cleaning up relationships: {e}")
            
            # Delete the document
            try:
                vertex_collection.delete(doc_key)
            except Exception as e:
                logger.error(f"Error deleting document {doc_key}: {e}")

if __name__ == "__main__":
    client = connect_arango()
    db = ensure_database(client)
    
    try:
        resources = run_workflow_example()
        
        # Validate results
        if resources.get("final_results", 0) >= 1:
            print("✅ Workflow example validation passed")
        else:
            print("❌ Workflow example failed: Insufficient results")
            sys.exit(1)
            
        # Clean up
        cleanup_resources(db, resources)
    except Exception as e:
        print(f"❌ Workflow example failed: {e}")
        sys.exit(1)
