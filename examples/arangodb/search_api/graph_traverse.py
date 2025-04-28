# src/pdf_extractor/arangodb/search_api/graph_traverse.py
import sys
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger
from arango.database import StandardDatabase
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME, GRAPH_NAME
)
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

def graph_traverse(
    db: StandardDatabase,
    start_vertex_key: str,
    min_depth: int = 1,
    max_depth: int = 1,
    direction: str = "ANY",
    relationship_types: Optional[List[str]] = None,
    vertex_filter: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Traverse the graph from a start vertex and return the connecting vertices.

    Args:
        db: ArangoDB database
        start_vertex_key: Key of the starting vertex
        min_depth: Minimum traversal depth
        max_depth: Maximum traversal depth
        direction: Direction of traversal (OUTBOUND, INBOUND, ANY)
        relationship_types: Optional list of relationship types to filter
        vertex_filter: Optional filter for vertices
        limit: Maximum number of results to return

    Returns:
        Dictionary with results and metadata
    """
    logger.info(f"Traversing graph from {start_vertex_key} (depth: {min_depth}..{max_depth}, direction: {direction})")

    # Validate parameters
    # Convert direction to uppercase for case-insensitive comparison
    direction_upper = direction.upper() if isinstance(direction, str) else direction
    if direction_upper not in ["OUTBOUND", "INBOUND", "ANY"]:
        # Log the actual invalid value received
        logger.error(f"Invalid direction received: '{direction}' (type: {type(direction)})")
        return {"results": [], "count": 0, "error": f"Invalid direction: '{direction}'. Must be OUTBOUND, INBOUND, or ANY."}

    if min_depth < 0 or max_depth < min_depth:
        logger.error(f"Invalid depth range: {min_depth}..{max_depth}")
        return {"results": [], "count": 0, "error": f"Invalid depth range: {min_depth}..{max_depth}"}

    try:
        # Construct AQL query
        start_vertex = f"{COLLECTION_NAME}/{start_vertex_key}"

        # Build edge filter if relationship types are provided
        edge_filter = ""
        if relationship_types:
            type_list = ", ".join([f"'{t}'" for t in relationship_types])
            edge_filter = f"FILTER e.type IN [{type_list}]"

        # Build vertex filter if provided
        vert_filter = ""
        if vertex_filter:
            conditions = []
            for field, value in vertex_filter.items():
                if isinstance(value, str):
                    conditions.append(f"v.{field} == '{value}'")
                else:
                    conditions.append(f"v.{field} == {value}")

            if conditions:
                vert_filter = f"FILTER {' AND '.join(conditions)}"

        aql = f"""
        FOR v, e, p IN {min_depth}..{max_depth} {direction} @start_vertex GRAPH @graph_name
        {edge_filter}
        {vert_filter}
        LIMIT @limit
        RETURN {{
            "vertex": v,
            "edge": e,
            "path": p
        }}
        """

        bind_vars = {
            "start_vertex": start_vertex,
            "graph_name": GRAPH_NAME,
            "limit": limit
        }

        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)

        logger.info(f"Traversal found {len(results)} results")
        return {
            "results": results,
            "count": len(results),
            "params": {
                "start_vertex_key": start_vertex_key, # Use key here for params dict
                "min_depth": min_depth,
                "max_depth": max_depth,
                "direction": direction,
                "relationship_types": relationship_types
            }
        }
    except Exception as e:
        logger.error(f"Traversal error: {e}")
        return {"results": [], "count": 0, "error": str(e)}

# --- Start of Corrected validate_traversal ---
def validate_traversal(traversal_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate traversal results against expected fixture data.

    Args:
        traversal_results: Results from graph_traverse function
        fixture_path: Path to fixture JSON file with expected results

    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}

    try:
        # Load fixture data
        with open(fixture_path, "r") as f:
            expected_data = json.load(f)

        # Check for top-level errors first
        if "error" in traversal_results:
            validation_failures["search_error"] = {
                "expected": "No error",
                "actual": traversal_results["error"]
            }
            return False, validation_failures

        # Structural validation
        if "results" not in traversal_results:
            validation_failures["missing_results"] = {"expected": "Results field present", "actual": "Results field missing"}
        if "count" not in traversal_results:
            validation_failures["missing_count"] = {"expected": "Count field present", "actual": "Count field missing"}
        if "params" not in traversal_results:
            validation_failures["missing_params"] = {"expected": "Params field present", "actual": "Params field missing"}

        # Return early if basic structure is missing
        if validation_failures:
             return False, validation_failures

        # Content validation - Count
        if "expected_count" in expected_data:
            # Use exact count comparison now
            if traversal_results.get("count") != expected_data["expected_count"]:
                validation_failures["count_value"] = {
                    "expected": expected_data['expected_count'],
                    "actual": traversal_results.get("count", 0)
                }

        # Params validation
        if "expected_params" in expected_data:
            expected_params = expected_data["expected_params"]
            actual_params = traversal_results.get("params", {})

            for param_name, expected_value in expected_params.items():
                if param_name not in actual_params:
                    validation_failures[f"missing_param_{param_name}"] = {
                        "expected": f"{param_name} present",
                        "actual": f"{param_name} missing"
                    }
                elif actual_params[param_name] != expected_value:
                    validation_failures[f"param_{param_name}"] = {
                        "expected": expected_value,
                        "actual": actual_params[param_name]
                    }

        # Result structure and vertex key validation
        actual_results = traversal_results.get("results", [])
        if "expected_result_vertex_keys" in expected_data:
             expected_keys = set(expected_data["expected_result_vertex_keys"])
             actual_keys = set()
             for i, result in enumerate(actual_results):
                 # Check basic structure of each result
                 required_fields = ["vertex", "edge", "path"]
                 for field in required_fields:
                     if field not in result:
                         validation_failures[f"result_{i}_missing_{field}"] = {
                             "expected": f"{field} present", "actual": f"{field} missing"
                         }
                 # Collect vertex key if structure is ok
                 if "vertex" in result and "_key" in result["vertex"]:
                     actual_keys.add(result["vertex"]["_key"])
                 else:
                      validation_failures[f"result_{i}_missing_vertex_key"] = {
                          "expected": "Vertex with _key", "actual": "Missing or incomplete"
                      }

             # Compare sets of keys
             if actual_keys != expected_keys:
                 validation_failures["vertex_keys_mismatch"] = {
                     "expected": sorted(list(expected_keys)),
                     "actual": sorted(list(actual_keys)),
                     "missing_in_actual": sorted(list(expected_keys - actual_keys)),
                     "extra_in_actual": sorted(list(actual_keys - expected_keys))
                 }

        return len(validation_failures) == 0, validation_failures

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, {"validation_error": {"expected": "Successful validation", "actual": str(e)}}
# --- End of Corrected validate_traversal ---

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Path to test fixture
    fixture_path = "src/test_fixtures/traversal_expected.json"

    try:
        # Initialize ArangoDB connection
        client = connect_arango()
        db = ensure_database(client)

        # Define a fixed start key based on setup_arangodb.py
        start_node_key = "doc1_f5f1489c"
        logger.info(f"Using start node key: {start_node_key}")

        # Ensure the start node exists (optional, but good practice)
        if not db.collection(COLLECTION_NAME).has(start_node_key):
             logger.error(f"Start node '{start_node_key}' not found. Run setup_arangodb.py.")
             print(f"❌ SETUP ERROR: Start node '{start_node_key}' not found. Run setup_arangodb.py.")
             sys.exit(1)

        # Run traversal with fixed start key and parameters matching potential fixture
        test_params = {
            "start_vertex_key": start_node_key,
            "min_depth": 1,
            "max_depth": 2,
            "direction": "ANY" # Default direction used in validate_traversal fixture expectation
        }
        results = graph_traverse(db, **test_params)

        # Validate results
        validation_passed, validation_failures = validate_traversal(results, fixture_path)

        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All graph traversal results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Graph traversal results don't match expected values")
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
