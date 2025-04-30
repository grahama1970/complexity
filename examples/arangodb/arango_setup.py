# src/pdf_extractor/arangodb/arango_setup.py
# https://gist.github.com/grahama1970/7aa0cef104e3150ee6788ac6794556c5

import sys
import json
import os
import time
import requests
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
from arango.client import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import (
    ArangoClientError,
    ArangoServerError,
    DatabaseCreateError,
    CollectionCreateError,
    GraphCreateError,
    ViewCreateError,
    ViewDeleteError,
    IndexCreateError,
    CollectionDeleteError,
)
from complexity.arangodb.config import (
    ARANGO_HOST,
    ARANGO_USER,
    ARANGO_PASSWORD,
    ARANGO_DB_NAME,
    COLLECTION_NAME,
    EDGE_COLLECTION_NAME,
    MESSAGES_COLLECTION_NAME,
    GRAPH_NAME,
    VIEW_NAME,
    EMBEDDING_DIMENSIONS,
    TEXT_ANALYZER,
    TAG_ANALYZER,
    VECTOR_INDEX_NLISTS,
)
from complexity.arangodb._archive.message_history_config import ( # Add correct import block
    MESSAGE_EDGE_COLLECTION_NAME,
    MESSAGE_GRAPH_NAME
)

# Import embedding utils – handle potential import errors during setup
try:
    from complexity.arangodb.embedding_utils import get_embedding
except ImportError as e:
    logger.warning(f"Could not import embedding utilities: {e}. Embedding generation will fail.")
    get_embedding = None

# Define embedding field constant
EMBEDDING_FIELD = "embedding"


def connect_arango() -> ArangoClient:
    """Establishes a connection to the ArangoDB server."""
    logger.info(f"Connecting to ArangoDB at {ARANGO_HOST}...")
    if not ARANGO_PASSWORD:
        logger.error("ARANGO_PASSWORD environment variable is not set.")
        sys.exit(1)

    try:
        client = ArangoClient(hosts=ARANGO_HOST, request_timeout=30)
        sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASSWORD)
        version_info = sys_db.version()
        if isinstance(version_info, str):
            version = version_info
        elif isinstance(version_info, dict):
            version = version_info.get("version", "unknown")
        else:
            version = "unknown"
        logger.info(f"Successfully connected to ArangoDB version {version}.")
        return client

    except (ArangoClientError, ArangoServerError) as e:
        logger.exception(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)


def ensure_database(client: ArangoClient) -> StandardDatabase:
    """Ensures the specified database exists."""
    try:
        sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASSWORD)
        current_databases = sys_db.databases()

        if ARANGO_DB_NAME not in current_databases:
            logger.info(f"Database '{ARANGO_DB_NAME}' not found. Creating...")
            sys_db.create_database(ARANGO_DB_NAME)
            logger.info(f"Database '{ARANGO_DB_NAME}' created successfully.")
        else:
            logger.info(f"Database '{ARANGO_DB_NAME}' already exists.")

        return client.db(ARANGO_DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)

    except (DatabaseCreateError, ArangoServerError, ArangoClientError) as e:
        logger.exception(f"Failed to ensure database: {e}")
        sys.exit(1)


def ensure_collection(db: StandardDatabase, collection_name: str) -> None:
    """Ensures the specified DOCUMENT collection exists."""
    try:
        if not db.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' not found. Creating...")
            db.create_collection(collection_name) # Removed waitForSync
            logger.info(f"Collection '{collection_name}' created successfully.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

    except (CollectionCreateError, ArangoServerError) as e:
        logger.exception(f"Failed to ensure collection '{collection_name}': {e}")
        sys.exit(1)


def ensure_edge_collection(
    db: StandardDatabase,
    collection_name: str = EDGE_COLLECTION_NAME
) -> None:
    """Ensures the specified EDGE collection exists."""
    try:
        if db.has_collection(collection_name):
            props = db.collection(collection_name).properties()
            if props.get("type") != 3:  # 3 = edge
                logger.info(f"Collection '{collection_name}' is not edge; recreating...")
                db.delete_collection(collection_name)
                db.create_collection(collection_name, edge=True) # Removed waitForSync
                logger.info(f"Edge collection '{collection_name}' recreated.")
            else:
                logger.info(f"Edge collection '{collection_name}' already exists.")
        else:
            logger.info(f"Edge collection '{collection_name}' not found. Creating...")
            db.create_collection(collection_name, edge=True) # Removed waitForSync
            logger.info(f"Edge collection '{collection_name}' created.")

    except (CollectionCreateError, CollectionDeleteError, ArangoServerError) as e:
        logger.exception(f"Failed to ensure edge collection '{collection_name}': {e}")
        sys.exit(1)


def ensure_graph(
    db: StandardDatabase,
    graph_name: str = GRAPH_NAME,
    edge_collection: str = EDGE_COLLECTION_NAME,
    vertex_collection: str = COLLECTION_NAME
) -> None:
    """Ensures the graph defining relationships exists."""
    try:
        if not (db.has_collection(vertex_collection) and db.has_collection(edge_collection)):
            logger.error("Cannot ensure graph: Required collections missing.")
            sys.exit(1)

        if not db.has_graph(graph_name):
            logger.info(f"Graph '{graph_name}' not found. Creating...")
            edge_def = {
                "edge_collection": edge_collection,
                "from_vertex_collections": [vertex_collection],
                "to_vertex_collections": [vertex_collection],
            }
            db.create_graph(graph_name, edge_definitions=[edge_def])
            logger.info(f"Graph '{graph_name}' created.")
        else:
            logger.info(f"Graph '{graph_name}' already exists.")

    except (GraphCreateError, ArangoServerError) as e:
        logger.exception(f"Failed to ensure graph: {e}")
        sys.exit(1)



# --- Message History Graph Setup ---

def ensure_message_edge_collection(db: StandardDatabase, collection_name: str) -> None: # Removed default argument
    """Ensures the message history edge collection exists."""
    ensure_edge_collection(db, collection_name) # Pass the name

def ensure_message_graph(db: StandardDatabase, graph_name: str, edge_collection: str, vertex_collection: str) -> None: # Removed default arguments
    """Ensures the message history graph exists."""
    ensure_graph( # Pass the names
        db,
        graph_name=graph_name,
        edge_collection=edge_collection,
        vertex_collection=vertex_collection
    )


def ensure_arangosearch_view(
    db: StandardDatabase,
    view_name: str = VIEW_NAME
) -> None:
    """Ensures an ArangoSearch view exists with specified configuration."""
    try:
        # 1. Ensure text analyzer
        analyzers = {a["name"] for a in db.analyzers()}
        if "text_en" not in analyzers:
            logger.info("Analyzer 'text_en' not found. Creating...")
            db.create_analyzer(
                "text_en",
                {"type": "text", "properties": {"locale": "en", "stemming": True, "case": "lower"}}
            )
            logger.info("Analyzer 'text_en' created.")

        # 2. Define view properties (no 'type' or 'name' keys)
        view_props = {
            "links": {
                COLLECTION_NAME: {
                    "fields": {
                        "problem":  {"analyzers": [TEXT_ANALYZER]},
                        "solution": {"analyzers": [TEXT_ANALYZER]},
                        "context":  {"analyzers": [TEXT_ANALYZER]},
                        "tags":     {"analyzers": [TAG_ANALYZER]},
                        EMBEDDING_FIELD: {}
                    },
                    "includeAllFields": False
                }
            },
            "primarySort":              [{"field": "_key", "direction": "asc"}],
            "commitIntervalMsec":       1000,
            "consolidationIntervalMsec":1000
        }

        existing = {v["name"] for v in db.views()}
        if view_name in existing:
            curr_links = db.view(view_name).get("links", {})
            if curr_links != view_props["links"]:
                logger.info(f"View '{view_name}' links mismatch; recreating...")
                db.delete_view(view_name)
                db.create_view(
                    name=view_name,
                    view_type="arangosearch",
                    properties=view_props
                )
                logger.info(f"Search view '{view_name}' recreated.")
            else:
                logger.info(f"Updating search view '{view_name}'...")
                db.update_view(
                    name=view_name,
                    properties=view_props
                )
                logger.info(f"Search view '{view_name}' updated.")
        else:
            logger.info(f"Creating search view '{view_name}'...")
            db.create_view(
                name=view_name,
                view_type="arangosearch",
                properties=view_props
            )
            logger.info(f"Search view '{view_name}' created.")

    except (ViewCreateError, ViewDeleteError, ArangoServerError) as e:
        logger.exception(f"Failed to ensure ArangoSearch view '{view_name}': {e}")
        sys.exit(1)


def ensure_vector_index(db: StandardDatabase, collection_name: str = COLLECTION_NAME) -> None:
    """Ensures a vector index exists on the embedding field."""
    try:
        col = db.collection(collection_name)
        indexes = col.indexes()  # list all existing indexes

        # If an old 'vector_index' exists, drop it first
        for idx in indexes:
            if idx.get("name") == "vector_index":
                idx_id = idx.get("id") or idx.get("name")
                logger.info(f"Dropping existing 'vector_index' (id={idx_id})...")
                col.delete_index(idx_id)
                break

        # Now create the new experimental vector index
        cfg = {
            "type":   "vector",
            "fields": [EMBEDDING_FIELD],
            "params": {
                "metric":    "cosine",
                "dimension": EMBEDDING_DIMENSIONS,
                "nLists":    VECTOR_INDEX_NLISTS
            },
            "name":   "vector_index"
        }
        logger.info(f"Creating vector index on '{collection_name}.{EMBEDDING_FIELD}'...")
        col.add_index(cfg)
        logger.info(f"Vector index 'vector_index' created with nLists={VECTOR_INDEX_NLISTS}.")

    except IndexCreateError as e:
        text = str(e).lower()
        # If ArangoDB still complains about a duplicate name, skip
        if "duplicate value" in text:
            logger.warning("Duplicate-name on 'vector_index'; assuming it exists and skipping.")
        else:
            logger.exception(f"Failed to ensure vector index: {e}")
            sys.exit(1)
    except ArangoServerError as e:
        logger.exception(f"Server error ensuring vector index: {e}")
        sys.exit(1)





def check_view_configuration(
    db: StandardDatabase,
    view_name: str = VIEW_NAME
) -> Dict[str, Any]:
    """Checks the configuration of the ArangoSearch view."""
    try:
        if not any(v["name"] == view_name for v in db.views()):
            return {"exists": False, "details": "View not found"}

        props = db.view(view_name)
        links = props.get("links", {}).get(COLLECTION_NAME, {})
        expected = {
            "problem":  [TEXT_ANALYZER],
            "solution": [TEXT_ANALYZER],
            "context":  [TEXT_ANALYZER],
            "tags":     [TAG_ANALYZER],
            EMBEDDING_FIELD: []
        }

        # Verify that each expected field is present with the correct analyzers
        fields_ok = all(
            field in links.get("fields", {}) and
            links["fields"][field].get("analyzers", []) == expected[field]
            for field in expected
        )
        include_ok = links.get("includeAllFields") is False

        result = {
            "exists": True,
            "type": props.get("type"),
            "links_match_expected": fields_ok and include_ok,
            "commitIntervalMsec": props.get("commitIntervalMsec"),
            "consolidationIntervalMsec": props.get("consolidationIntervalMsec")
        }
        return result

    except (ArangoServerError, ArangoClientError) as e:
        logger.exception(f"Failed to check view configuration: {e}")
        return {"exists": False, "details": str(e)}



import requests

def check_vector_index_configuration(
    db: StandardDatabase,
    collection_name: str = COLLECTION_NAME
) -> Dict[str, Any]:
    """Checks the vector index via HTTP API with withHidden=true."""
    url = f"{ARANGO_HOST}/_db/{ARANGO_DB_NAME}/_api/index"
    auth = (ARANGO_USER, ARANGO_PASSWORD)
    params = {"collection": collection_name, "withHidden": "true"}
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, auth=auth)
            resp.raise_for_status()
            for idx in resp.json().get("indexes", []):
                if (
                    idx.get("type") == "vector" and
                    idx.get("params", {}).get("dimension") == EMBEDDING_DIMENSIONS and
                    idx.get("params", {}).get("metric") == "cosine" and
                    idx.get("params", {}).get("nLists") == VECTOR_INDEX_NLISTS
                ):
                    result = {
                        "exists": True,
                        "type":   idx["type"],
                        "fields": idx["fields"],
                        "params": idx["params"],
                        "name":   idx.get("name")
                    }
                    logger.info(f"Vector index found: {result}")
                    return result

            logger.warning(f"Attempt {attempt}: vector index not found.")
            time.sleep(1)

        except requests.RequestException as e:
            logger.exception(f"HTTP error on attempt {attempt}: {e}")
            time.sleep(1)

    logger.error("Vector index 'vector_index' not found after retries.")
    return {"exists": False, "details": "Vector index not found"}




def create_test_documents(db: StandardDatabase, collection_name: str) -> None:
    """Create some test documents for validation, matching log keys."""
    if get_embedding is None:
        logger.error("Embedding function not available. Cannot create test documents.")
        sys.exit(1)

    try:
        col = db.collection(collection_name)
        logger.info(f"Truncating collection '{collection_name}'...")
        col.truncate()
        logger.info(f"Collection '{collection_name}' truncated.")

        docs = [
            {"_key": "doc1_f5f1489c", "problem": "Python error when processing JSON data",
             "solution": "Use try/except blocks to handle JSON parsing exceptions",
             "context": "Error handling in data processing", "tags": ["python", "json", "error-handling"]},
            {"_key": "doc2_f5f1489c", "problem": "Python script runs out of memory with large datasets",
             "solution": "Use chunking to process large data incrementally",
             "context": "Performance optimization", "tags": ["python", "memory", "optimization"]},
            {"_key": "doc3_f5f1489c", "problem": "Need to search documents efficiently",
             "solution": "Use ArangoDB's vector search with embeddings",
             "context": "Document retrieval", "tags": ["arangodb", "vector-search", "embeddings"]},
        ]

        logger.info("Generating and adding embeddings to test documents...")
        for doc in docs:
            text = f"{doc['problem']} {doc['solution']} {doc['context']}"
            vec = get_embedding(text)
            if not vec or len(vec) != EMBEDDING_DIMENSIONS:
                logger.error(f"Invalid embedding for {doc['_key']}: {vec}")
                sys.exit(1)
            doc[EMBEDDING_FIELD] = vec
            col.insert(doc, overwrite=True)
            logger.info(f"Inserted/updated document '{doc['_key']}'.")

        count = col.count()
        logger.info(f"Collection '{collection_name}' now contains {count} documents.")
        if count != len(docs):
            logger.error(f"Expected {len(docs)} docs, found {count}.")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Failed to create test documents: {e}")
        sys.exit(1)


def create_test_relationships(
    db: StandardDatabase,
    edge_collection: str,
    vertex_collection: str
) -> None:
    """Create some test relationships for validation."""
    try:
        edge_col = db.collection(edge_collection)
        vert_col = db.collection(vertex_collection)

        if edge_col.count() > 0:
            logger.info(f"Edge collection '{edge_collection}' already populated.")
            return

        keys = ["doc1_f5f1489c", "doc2_f5f1489c", "doc3_f5f1489c"]
        if sum(1 for k in keys if vert_col.has(k)) < 2:
            logger.error("Not enough vertices to create relationships.")
            sys.exit(1)

        rels = [
            {"_from": f"{vertex_collection}/doc1_f5f1489c", "_to": f"{vertex_collection}/doc3_f5f1489c",
             "type": "related_python_issue", "weight": 0.7},
            {"_from": f"{vertex_collection}/doc2_f5f1489c", "_to": f"{vertex_collection}/doc3_f5f1489c",
             "type": "related_performance_issue", "weight": 0.5},
        ]

        for r in rels:
            edge_col.insert(r)
            logger.info(f"Created relationship {r['_from']} → {r['_to']}")

    except Exception as e:
        logger.exception(f"Failed to create test relationships: {e}")
        sys.exit(1)


def validate_setup(db: StandardDatabase) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """Validate database setup end-to-end."""
    failures: Dict[str, Dict[str, Any]] = {}

    try:
        # Collections
        existing = [c["name"] for c in db.collections() if not c["name"].startswith("_")]
        for name in (COLLECTION_NAME, EDGE_COLLECTION_NAME, MESSAGES_COLLECTION_NAME):
            if name not in existing:
                failures[f"missing_collection_{name}"] = {
                    "expected": f"{name} exists", "actual": f"{name} not found"
                }

        # Graph
        if not db.has_graph(GRAPH_NAME):
            failures[f"missing_graph_{GRAPH_NAME}"] = {
                "expected": f"{GRAPH_NAME} exists", "actual": "Graph not found"
            }

        # View
        if not any(v["name"] == VIEW_NAME for v in db.views()):
            failures[f"missing_view_{VIEW_NAME}"] = {
                "expected": f"{VIEW_NAME} exists", "actual": "View not found"
            }
        else:
            cfg = check_view_configuration(db, VIEW_NAME)
            if not cfg.get("links_match_expected", False):
                failures[f"view_config_{VIEW_NAME}"] = {
                    "expected": "View links match",
                    "actual": "View links mismatch"
                }

        # Vector index
        vcfg = check_vector_index_configuration(db, COLLECTION_NAME)
        if not vcfg.get("exists", False):
            failures[f"missing_vector_index_{EMBEDDING_FIELD}"] = {
                "expected": f"Vector index on {EMBEDDING_FIELD}",
                "actual": vcfg.get("details", "Not found")
            }

        # Doc & edge counts
        if db.collection(COLLECTION_NAME).count() != 3:
            failures["document_count"] = {"expected": 3, "actual": db.collection(COLLECTION_NAME).count()}
        if db.collection(EDGE_COLLECTION_NAME).count() != 2:
            failures["edge_count"] = {"expected": 2, "actual": db.collection(EDGE_COLLECTION_NAME).count()}

        return len(failures) == 0, failures

    except Exception as e:
        logger.exception(f"Validation error: {e}")
        return False, {"validation_error": {"expected": "no exception", "actual": str(e)}}


if __name__ == "__main__":
    logger.add(sys.stderr,
            format="{time:HH:mm:ss} | {level:<5} | {message}",
            level="INFO",
            colorize=True)

    try:
        client = connect_arango()
        db = ensure_database(client)
        ensure_collection(db, COLLECTION_NAME)
        ensure_edge_collection(db)
        ensure_collection(db, MESSAGES_COLLECTION_NAME)
        create_test_documents(db, COLLECTION_NAME)
        create_test_relationships(db, EDGE_COLLECTION_NAME, COLLECTION_NAME)
        ensure_graph(db)
        # Call functions with the globally imported constants (Corrected typo: MESSAGE_ not MESSAGES_)
        ensure_message_edge_collection(db, MESSAGE_EDGE_COLLECTION_NAME)
        ensure_message_graph(db, MESSAGE_GRAPH_NAME, MESSAGE_EDGE_COLLECTION_NAME, MESSAGES_COLLECTION_NAME)

        ensure_arangosearch_view(db) # Keep original indentation
        ensure_vector_index(db)

        logger.info("Validating ArangoDB setup...")
        passed, errors = validate_setup(db)
        if passed:
            logger.info("✅ ArangoDB setup completed successfully.")
            sys.exit(0)
        else:
            logger.error("❌ Validation failed:")
            for key, detail in errors.items():
                logger.error(f"  - {key}: expected={detail['expected']}, actual={detail['actual']}")
            sys.exit(1)

    except Exception as e: # Correct indentation relative to 'try'
        logger.exception(f"❌ ERROR during setup: {e}") # Correct indentation
        sys.exit(1)
