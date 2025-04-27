from typing import Dict, Any
from loguru import logger
from arango import ArangoClient
from arango.exceptions import ArangoError, ServerConnectionError
from typing import List

def initialize_database(config: Dict[str, Any]):
    """
    Sets up and connects to the ArangoDB client, ensuring the database is created if it doesn't exist.

    Args:
        config (dict): Either a standalone `arango_config` dictionary or a larger `config` dictionary
                    containing `arango_config` as a nested field.

    Returns:
        db: The connected ArangoDB database instance or None if an error occurs.
    """
    try:
        # Handle both standalone `arango_config` and nested `arango_config` cases
        if "arango_config" in config:
            arango_config = config["arango_config"]
        else:
            arango_config = config

        # Extract configuration values with defaults
        arango_config = config.get("arango_config", {})
        hosts = arango_config.get("hosts", ["http://localhost:8529"])
        db_name = arango_config.get("db_name", "verifaix")
        username = arango_config.get("username", "root")
        password = arango_config.get("password", "openSesame")

        # Initialize the ArangoDB client
        client = ArangoClient(hosts=hosts)

        # Connect to the database
        db = client.db(db_name, username=username, password=password)
        # logger.success(f"Connected to database '{db_name}'.")
        return db

    except ArangoError as e:
        logger.error(f"ArangoDB error: {e}")
        return None
    except ServerConnectionError as e:
        logger.error(f"Failed to connect to ArangoDB server: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def store_docs_in_arango(db, docs: List[Dict[str, Any]], config: Dict[str, Any]):
    """
    Store documents in the specified Arango collection. 
    Each doc is a dict with question, rating, label, embedding. We'll add _key.
    """
    coll_name = config["arango_collection"]
    if not db.has_collection(coll_name):
        db.create_collection(coll_name)
        logger.info(f"Created Arango collection '{coll_name}'")

    coll = db.collection(coll_name)

    BATCH_SIZE = 1000
    total = len(docs)
    logger.info(f"Storing {total} docs in collection '{coll_name}'")

    inserted = 0
    for start_idx in range(0, total, BATCH_SIZE):
        batch = docs[start_idx:start_idx + BATCH_SIZE]
        to_insert = []
        for i, d in enumerate(batch):
            doc_copy = d.copy()
            doc_copy["_key"] = f"doc_{start_idx + i}"
            to_insert.append(doc_copy)
        coll.insert_many(to_insert, overwrite=True)
        inserted += len(batch)

    logger.info(f"Inserted {inserted} documents total into ArangoDB.")



def create_arangosearch_view(db, config: Dict[str, Any]):
    """
    Create an ArangoSearch view that indexes:
      - question => text analyzer
      - embedding => identity analyzer (for COSINE_SIMILARITY)
    """
    view_name = config["arango_view"]
    coll_name = config["arango_collection"]

    if db.has_arangosearch_view(view_name):
        logger.info(f"View '{view_name}' exists. Dropping.")
        db.delete_arangosearch_view(view_name)

    logger.info(f"Creating view '{view_name}'...")

    properties = {
        "links": {
            coll_name: {
                "fields": {
                    "question": {"analyzers": ["text_en"]},
                    "embedding": {"analyzers": ["identity"]},
                }
            }
        }
    }
    db.create_arangosearch_view(view_name, properties=properties)
    logger.info(f"ArangoSearch view '{view_name}' created successfully.")
