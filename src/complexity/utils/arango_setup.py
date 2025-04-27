"""
ArangoDB setup and configuration for the complexity analyzer.

This module handles the setup and configuration of ArangoDB collections, views, and indexes
for the complexity analysis system. It manages vector search capabilities for embeddings
and text search for questions.

Documentation:
- ArangoDB Python Driver: https://docs.python-arango.com
- ArangoDB ArangoSearch: https://www.arangodb.com/docs/stable/arangosearch.html
"""

import sys
import json
import os
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, cast
from loguru import logger
import requests
from dotenv import load_dotenv
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

# Load environment variables from .env file
load_dotenv()

# Configuration - Using environment variables from .env file
ARANGO_HOST = os.getenv("ARANGO_HOST")
ARANGO_USER = os.getenv("ARANGO_USER")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD")
ARANGO_DB_NAME = os.getenv("ARANGO_DB_NAME")

# Validate required environment variables are loaded and not None
if not all([ARANGO_HOST, ARANGO_USER, ARANGO_PASSWORD, ARANGO_DB_NAME]):
    missing = [k for k, v in {"ARANGO_HOST": ARANGO_HOST, "ARANGO_USER": ARANGO_USER, "ARANGO_PASSWORD": ARANGO_PASSWORD, "ARANGO_DB_NAME": ARANGO_DB_NAME}.items() if not v]
    logger.error(f"Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

# Variables ARANGO_HOST, ARANGO_USER, ARANGO_PASSWORD, ARANGO_DB_NAME are confirmed non-None here.
# We can cast them if needed elsewhere, but the initial check ensures they exist.

COLLECTION_NAME = "complexity"
VIEW_NAME = "complexity_view"

# Embedding configuration for modernbert-embed-base
EMBEDDING_DIMENSIONS = 768
EMBEDDING_FIELD = "embedding"

# Search configuration
TEXT_ANALYZER = "text_en"
TAG_ANALYZER = "text_en" # Reusing text_en for tags as well
VECTOR_INDEX_NLISTS = 3  # Must be <= number of training docs (currently 3)

def generate_test_embedding(dim: int = EMBEDDING_DIMENSIONS) -> List[float]:
    """Generate a test embedding vector of specified dimension, normalized for cosine similarity."""
    vec = np.random.randn(dim)
    normalized_vec = vec / np.linalg.norm(vec)  # Normalize to unit length for cosine similarity
    # Convert to list of Python floats (important for ArangoDB)
    return [float(x) for x in normalized_vec]

def connect_arango() -> ArangoClient:
    """Establishes a connection to the ArangoDB server."""
    logger.info(f"Connecting to ArangoDB at {ARANGO_HOST}...")
    # Password check is handled globally after load_dotenv

    try:
        # Ensure ARANGO_HOST is treated as string for ArangoClient
        client = ArangoClient(hosts=str(ARANGO_HOST), request_timeout=30)
        # Use validated, non-None variables
        sys_db = client.db("_system", username=str(ARANGO_USER), password=str(ARANGO_PASSWORD))
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
        # Ensure ARANGO_DB_NAME is treated as string
        db_name = str(ARANGO_DB_NAME)
        sys_db = client.db("_system", username=str(ARANGO_USER), password=str(ARANGO_PASSWORD))
        current_databases = sys_db.databases()

        if db_name not in current_databases:
            logger.info(f"Database '{db_name}' not found. Creating...")
            sys_db.create_database(db_name)
            logger.info(f"Database '{db_name}' created successfully.")
        else:
            logger.info(f"Database '{db_name}' already exists.")

        return client.db(db_name, username=str(ARANGO_USER), password=str(ARANGO_PASSWORD))

    except (DatabaseCreateError, ArangoServerError, ArangoClientError) as e:
        logger.exception(f"Failed to ensure database: {e}")
        sys.exit(1)

def ensure_collection(db: StandardDatabase, collection_name: str) -> None:
    """Ensures the specified DOCUMENT collection exists."""
    try:
        if not db.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' not found. Creating...")
            db.create_collection(collection_name)
            logger.info(f"Collection '{collection_name}' created successfully.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

    except (CollectionCreateError, ArangoServerError) as e:
        logger.exception(f"Failed to ensure collection '{collection_name}': {e}")
        sys.exit(1)

def ensure_arangosearch_view(db: StandardDatabase, view_name: str = VIEW_NAME) -> None:
    """Ensures an ArangoSearch view exists with specified configuration."""
    try:
        # 1. Ensure text analyzer
        analyzers = {a["name"] for a in db.analyzers()}
        if TEXT_ANALYZER not in analyzers:
            logger.info(f"Analyzer '{TEXT_ANALYZER}' not found. Creating...")
            db.create_analyzer(
                TEXT_ANALYZER,
                {"type": "text", "properties": {"locale": "en", "stemming": True, "case": "lower"}}
            )
            logger.info(f"Analyzer '{TEXT_ANALYZER}' created.")

        # 2. Define view properties
        view_props = {
            "links": {
                COLLECTION_NAME: {
                    "fields": {
                        "question": {"analyzers": [TEXT_ANALYZER]},
                        EMBEDDING_FIELD: {} # Index embedding field for vector search
                    },
                    "includeAllFields": False # Only include specified fields
                }
            },
            "primarySort": [{"field": "_key", "direction": "asc"}],
            "commitIntervalMsec": 1000,
            "consolidationIntervalMsec": 1000
        }

        existing = {v["name"] for v in db.views()}
        if view_name in existing:
            # Check if properties need update (simple link check for now)
            current_props = db.view(view_name)
            if current_props.get("links") != view_props["links"]:
                 logger.info(f"View '{view_name}' properties mismatch; recreating...")
                 db.delete_view(view_name)
                 db.create_view(name=view_name, view_type="arangosearch", properties=view_props)
                 logger.info(f"Search view '{view_name}' recreated.")
            else:
                 logger.info(f"Search view '{view_name}' already exists with correct properties.")
                 # Optionally update other properties if needed: db.update_view(...)
        else:
            logger.info(f"Creating search view '{view_name}'...")
            db.create_view(name=view_name, view_type="arangosearch", properties=view_props)
            logger.info(f"Search view '{view_name}' created.")

    except (ViewCreateError, ViewDeleteError, ArangoServerError) as e:
        logger.exception(f"Failed to ensure ArangoSearch view '{view_name}': {e}")
        sys.exit(1)

def ensure_vector_index(db: StandardDatabase, collection_name: str = COLLECTION_NAME) -> None:
    """Ensures a vector index exists on the embedding field."""
    try:
        col = db.collection(collection_name)
        indexes = col.indexes()

        # Check if a suitable vector index already exists
        vector_index_exists = False
        for idx in indexes:
            if idx.get("name") == "vector_index":
                # Basic check for type and field - could be more robust
                if idx.get("type") == "vector" and EMBEDDING_FIELD in idx.get("fields", []):
                     # Check parameters match
                     params = idx.get("params", {})
                     if (params.get("dimension") == EMBEDDING_DIMENSIONS and
                         params.get("metric") == "cosine"):
                         logger.info(f"Vector index 'vector_index' already exists with correct parameters.")
                         vector_index_exists = True
                         break
                     else:
                         logger.warning(f"Existing 'vector_index' has incorrect parameters. Dropping...")
                         col.delete_index(idx.get("id") or idx.get("name"))
                         break
                else:
                     logger.warning(f"Existing index named 'vector_index' is not a valid vector index on '{EMBEDDING_FIELD}'. Dropping...")
                     col.delete_index(idx.get("id") or idx.get("name"))
                     break

        if not vector_index_exists:
            # Create new vector index
            cfg = {
                "type": "vector",
                "fields": [EMBEDDING_FIELD],
                "params": {
                    "metric": "cosine",
                    "dimension": EMBEDDING_DIMENSIONS,
                    "nLists": VECTOR_INDEX_NLISTS # nLists is required
                },
                "name": "vector_index"
            }
            logger.info(f"Creating vector index on '{collection_name}.{EMBEDDING_FIELD}'...")
            col.add_index(cfg)
            logger.info(f"Vector index 'vector_index' created with nLists={VECTOR_INDEX_NLISTS}.")

    except IndexCreateError as e:
        if "duplicate name" in str(e).lower(): # Adjusted error check string
            logger.warning("Duplicate name error on 'vector_index'; assuming it exists and skipping creation.")
        else:
            logger.exception(f"Failed to ensure vector index: {e}")
            sys.exit(1)
    except ArangoServerError as e:
        logger.exception(f"Server error ensuring vector index: {e}")
        sys.exit(1)

def check_view_configuration(db: StandardDatabase, view_name: str = VIEW_NAME) -> Dict[str, Any]:
    """Checks the configuration of the ArangoSearch view."""
    try:
        if not any(v["name"] == view_name for v in db.views()):
            return {"exists": False, "details": "View not found"}

        props = db.view(view_name)
        links = props.get("links", {}).get(COLLECTION_NAME, {})
        expected_fields = {
            "question": [TEXT_ANALYZER],
            EMBEDDING_FIELD: [] # Empty list means just index the field for vector search
        }

        # Verify fields and analyzers
        actual_fields = links.get("fields", {})
        fields_ok = True
        for field, analyzers in expected_fields.items():
            if field not in actual_fields:
                fields_ok = False
                break
            # For embedding field, we don't expect analyzers key
            if field == EMBEDDING_FIELD:
                if "analyzers" in actual_fields[field]: # Should not have analyzers
                     fields_ok = False
                     break
            elif actual_fields[field].get("analyzers", []) != analyzers:
                 fields_ok = False
                 break

        include_ok = links.get("includeAllFields") is False

        return {
            "exists": True,
            "type": props.get("type"),
            "links_match_expected": fields_ok and include_ok,
            "commitIntervalMsec": props.get("commitIntervalMsec"),
            "consolidationIntervalMsec": props.get("consolidationIntervalMsec")
        }

    except (ArangoServerError, ArangoClientError) as e:
        logger.exception(f"Failed to check view configuration: {e}")
        return {"exists": False, "details": str(e)}

def check_vector_index_configuration(
    db: StandardDatabase,
    collection_name: str = COLLECTION_NAME
) -> Dict[str, Any]:
    """Checks the vector index configuration via HTTP API."""
    # Assert that variables are strings to satisfy type checker for the tuple
    assert isinstance(ARANGO_USER, str), "ARANGO_USER must be a string"
    assert isinstance(ARANGO_PASSWORD, str), "ARANGO_PASSWORD must be a string"
    assert isinstance(ARANGO_HOST, str), "ARANGO_HOST must be a string"
    assert isinstance(ARANGO_DB_NAME, str), "ARANGO_DB_NAME must be a string"

    url = f"{ARANGO_HOST}/_db/{ARANGO_DB_NAME}/_api/index"
    # Use validated, non-None variables for auth tuple
    auth: Tuple[str, str] = (ARANGO_USER, ARANGO_PASSWORD)
    params = {"collection": collection_name, "withHidden": "true"}

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, auth=auth)
            resp.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            indexes = resp.json().get("indexes", [])
            for idx in indexes:
                if idx.get("name") == "vector_index":
                    if (idx.get("type") == "vector" and
                        EMBEDDING_FIELD in idx.get("fields", []) and
                        idx.get("params", {}).get("dimension") == EMBEDDING_DIMENSIONS and
                        idx.get("params", {}).get("metric") == "cosine"):
                        # nLists might not be present if not explicitly set or if collection was empty
                        logger.info(f"Vector index 'vector_index' found with correct parameters.")
                        return {
                            "exists": True,
                            "type": idx["type"],
                            "fields": idx["fields"],
                            "params": idx["params"],
                            "name": idx.get("name")
                        }
                    else:
                        # Found an index named 'vector_index' but params don't match
                        logger.error(f"Found index 'vector_index' but parameters are incorrect: {idx}")
                        return {"exists": False, "details": "Incorrect parameters"}

            # If loop finishes without finding the index
            logger.warning(f"Attempt {attempt + 1}: Vector index 'vector_index' not found.")
            time.sleep(1) # Wait before retrying

        except requests.exceptions.RequestException as e:
            logger.exception(f"HTTP error checking index on attempt {attempt + 1}: {e}")
            time.sleep(1) # Wait before retrying
        except json.JSONDecodeError as e:
             logger.exception(f"Failed to decode JSON response on attempt {attempt + 1}: {e}")
             time.sleep(1)

    logger.error("Vector index 'vector_index' not found or configuration incorrect after retries.")
    return {"exists": False, "details": "Vector index not found or configuration incorrect"}


def create_test_documents(db: StandardDatabase, collection_name: str) -> None:
    """Create test documents with embeddings for index validation."""
    try:
        col = db.collection(collection_name)
        logger.info(f"Preparing test documents for collection '{collection_name}'...")

        # Test documents with known content
        docs = [
            {"_key": "test1", "question": "What is vector search?",
             "context": "Exploring vector search capabilities",
             EMBEDDING_FIELD: generate_test_embedding()},
            {"_key": "test2", "question": "How do embeddings work?",
             "context": "Understanding embedding models",
             EMBEDDING_FIELD: generate_test_embedding()},
            {"_key": "test3", "question": "Why use cosine similarity?",
             "context": "Vector similarity metrics comparison",
             EMBEDDING_FIELD: generate_test_embedding()}
        ]

        # Insert documents
        results = col.insert_many(docs, overwrite=True)
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
             logger.error(f"Errors inserting test documents: {errors}")
             raise ValueError("Failed to insert all test documents")

        inserted_count = len(results) - len(errors)
        logger.info(f"Successfully inserted/updated {inserted_count} test documents.")

        count = col.count()
        if count != len(docs):
            # This might happen if overwrite=False and docs existed
            logger.warning(f"Expected {len(docs)} documents, but collection count is {count}. Overwriting might affect this.")
            # Decide if this is critical - for setup, maybe not if count >= len(docs)
            if count < len(docs):
                 raise ValueError(f"Collection count {count} is less than expected {len(docs)}")

    except Exception as e:
        logger.exception(f"Failed to create test documents: {e}")
        sys.exit(1)

def validate_setup(db: StandardDatabase) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """Validate the database setup matches requirements."""
    expected = {
        "database": ARANGO_DB_NAME,
        "collection": COLLECTION_NAME,
        "view": VIEW_NAME,
        "embedding_dimension": EMBEDDING_DIMENSIONS,
        "text_analyzer": TEXT_ANALYZER,
        "vector_metric": "cosine"
    }

    failures: Dict[str, Dict[str, Any]] = {}

    try:
        # Database Validation (Implicitly checked by getting db object)
        logger.info(f"Checking database '{ARANGO_DB_NAME}'...")
        # No direct check needed if db object is valid

        # Collection Validation
        logger.info(f"Checking collection '{COLLECTION_NAME}'...")
        if not db.has_collection(COLLECTION_NAME):
            failures[f"missing_collection_{COLLECTION_NAME}"] = {
                "expected": f"{COLLECTION_NAME} exists",
                "actual": f"{COLLECTION_NAME} not found"
            }
        else:
             # Test Documents Validation (only if collection exists)
             logger.info(f"Checking test document count in '{COLLECTION_NAME}'...")
             doc_count = db.collection(COLLECTION_NAME).count()
             # Allow for more docs if run multiple times, but must have at least 3
             if doc_count < 3:
                 failures["test_documents"] = {
                     "expected": ">= 3",
                     "actual": doc_count
                 }

        # View Validation
        logger.info(f"Checking view '{VIEW_NAME}' configuration...")
        view_cfg = check_view_configuration(db, VIEW_NAME)
        if not view_cfg.get("exists", False):
            failures[f"missing_view_{VIEW_NAME}"] = {
                "expected": f"{VIEW_NAME} exists",
                "actual": view_cfg.get("details", "View not found")
            }
        elif not view_cfg.get("links_match_expected", False):
            failures[f"view_config_{VIEW_NAME}"] = {
                "expected": "View links match expected configuration",
                "actual": "View links mismatch or includeAllFields is not False"
            }

        # Vector Index Validation
        logger.info(f"Checking vector index 'vector_index' on '{COLLECTION_NAME}'...")
        vcfg = check_vector_index_configuration(db, COLLECTION_NAME)
        if not vcfg.get("exists", False):
            failures[f"vector_index_{EMBEDDING_FIELD}"] = {
                "expected": f"Vector index on {EMBEDDING_FIELD} with dim={EMBEDDING_DIMENSIONS}, metric=cosine",
                "actual": vcfg.get("details", "Not found or incorrect")
            }
        # No need for else block, check_vector_index_configuration already logs details

        return len(failures) == 0, failures

    except Exception as e:
        logger.exception(f"Validation error: {e}")
        return False, {"validation_error": {"expected": "no exception", "actual": str(e)}}

if __name__ == "__main__":
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level:<5} | {message}",
        level="INFO",
        colorize=True
    )

    try:
        # Step 1: Setup basic components
        client = connect_arango()
        db = ensure_database(client)
        ensure_collection(db, COLLECTION_NAME)

        # Step 2: Create test documents (required for vector index)
        create_test_documents(db, COLLECTION_NAME)

        # Step 3: Setup search view and vector index
        ensure_arangosearch_view(db)
        ensure_vector_index(db) # Try creating index after docs exist

        # Step 4: Validate setup
        logger.info("Validating ArangoDB setup...")
        passed, errors = validate_setup(db)

        if passed:
            logger.info("✅ VALIDATION COMPLETE - ArangoDB setup matches all expected values")
            sys.exit(0)
        else:
            logger.error("❌ VALIDATION FAILED - Setup does not match expected values")
            for key, detail in errors.items():
                logger.error(f"  - {key}: Expected: {detail['expected']}, Got: {detail['actual']}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"❌ ERROR during setup: {e}")
        sys.exit(1)
