"""
Enhanced relationship operations for supporting CLI graph commands.

This module provides compatibility functions to bridge the parameter mismatch
between the CLI graph commands and underlying relationship operations.
"""

from typing import Dict, Any, Optional
from loguru import logger

from arango.database import StandardDatabase
from complexity.arangodb.db_operations import create_relationship, delete_relationship_by_key

def create_edge_from_cli(
    db: StandardDatabase,
    from_key: str,
    to_key: str,
    collection: str,
    edge_collection: str,
    edge_type: str,
    rationale: str,
    attributes: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Create a relationship edge using CLI-friendly parameters.
    
    This function bridges the gap between the CLI graph command parameters
    and the underlying relationship creation function.
    
    Args:
        db: ArangoDB database handle
        from_key: Key of the source document
        to_key: Key of the target document
        collection: Name of the document collection
        edge_collection: Name of the edge collection
        edge_type: Type of relationship
        rationale: Reason for the relationship
        attributes: Additional properties for the edge
        
    Returns:
        Optional[Dict[str, Any]]: The created edge document if successful, None otherwise
    """
    try:
        # Create the edge document directly using ArangoDB collection API
        edge = {
            "_from": f"{collection}/{from_key}",
            "_to": f"{collection}/{to_key}",
            "type": edge_type,
            "rationale": rationale,
            **(attributes or {})
        }
        
        # Insert the edge directly into the specified edge collection
        edge_coll = db.collection(edge_collection)
        result = edge_coll.insert(edge, return_new=True)
        
        logger.info(f"Created edge in {edge_collection}: {result.get('_key', result)}")
        return result["new"] if "new" in result else result
        
    except Exception as e:
        logger.error(f"Failed to create edge: {e}")
        return None

def delete_edge_from_cli(
    db: StandardDatabase,
    edge_key: str,
    edge_collection: str
) -> bool:
    """
    Delete a relationship edge using CLI-friendly parameters.
    
    Args:
        db: ArangoDB database handle
        edge_key: Key of the edge to delete
        edge_collection: Name of the edge collection
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Delete the edge directly using ArangoDB collection API
        edge_coll = db.collection(edge_collection)
        result = edge_coll.delete(edge_key, ignore_missing=True)
        
        logger.info(f"Deleted edge from {edge_collection}: {edge_key}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete edge: {e}")
        return False