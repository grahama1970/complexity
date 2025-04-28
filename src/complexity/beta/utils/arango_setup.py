import sys
from loguru import logger
from arango.client import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import (
    DatabaseCreateError,
    CollectionCreateError,
    ViewCreateError,
    ViewDeleteError,
    IndexCreateError,
)
from complexity.beta.utils.config import CONFIG

def connect_arango() -> ArangoClient:
    """Connect to ArangoDB."""
    logger.info(f"Connecting to ArangoDB at {CONFIG['arango']['host']}")
    try:
        client = ArangoClient(hosts=CONFIG["arango"]["host"])
        sys_db = client.db("_system", username=CONFIG["arango"]["user"], password=CONFIG["arango"]["password"])
        logger.info(f"Connected to ArangoDB version {sys_db.version()}")
        return client
    except Exception as e:
        logger.exception(f"Connection failed: {e}")
        raise

def ensure_database(client: ArangoClient) -> StandardDatabase:
    """Ensure database exists."""
    try:
        sys_db = client.db("_system", username=CONFIG["arango"]["user"], password=CONFIG["arango"]["password"])
        db_name = CONFIG["arango"]["db_name"]
        if db_name not in sys_db.databases():
            logger.info(f"Creating database '{db_name}'")
            sys_db.create_database(db_name)
        return client.db(db_name, username=CONFIG["arango"]["user"], password=CONFIG["arango"]["password"])
    except DatabaseCreateError as e:
        logger.exception(f"Database setup failed: {e}")
        raise

def ensure_collection(db: StandardDatabase) -> None:
    """Ensure collection exists."""
    try:
        name = CONFIG["search"]["collection_name"]
        if not db.has_collection(name):
            logger.info(f"Creating collection '{name}'")
            db.create_collection(name)
        logger.info(f"Collection '{name}' ready")
    except CollectionCreateError as e:
        logger.exception(f"Collection creation failed: {e}")
        raise

def ensure_arangosearch_view(db: StandardDatabase) -> None:
    """Ensure ArangoSearch view."""
    try:
        view_name = CONFIG["search"]["view_name"]
        analyzer = CONFIG["search"]["text_analyzer"]
        links = {
            CONFIG["search"]["collection_name"]: {
                "fields": {
                    "question": {"analyzers": [analyzer]},
                    CONFIG["embedding"]["field"]: {},
                },
                "includeAllFields": False,
            }
        }
        props = {"links": links}
        if analyzer not in {a["name"] for a in db.analyzers()}:
            logger.info(f"Creating analyzer '{analyzer}'")
            db.create_analyzer(
                analyzer,
                {"type": "text", "properties": {"locale": "en", "stemming": True, "case": "lower"}},
            )
        if view_name in {v["name"] for v in db.views()}:
            current = db.view(view_name)
            if current.get("links", {}) != links:
                logger.info(f"Recreating view '{view_name}'")
                db.delete_view(view_name)
                db.create_view(name=view_name, view_type="arangosearch", properties=props)
            else:
                logger.info(f"Search view '{view_name}' up-to-date")
        else:
            logger.info(f"Creating view '{view_name}'")
            db.create_view(name=view_name, view_type="arangosearch", properties=props)
        logger.info(f"Search view '{view_name}' ready")
    except (ViewCreateError, ViewDeleteError) as e:
        logger.exception(f"View setup failed: {e}")
        raise

def ensure_vector_index(db: StandardDatabase) -> None:
    """Ensure vector index after data insertion."""
    try:
        col = db.collection(CONFIG["search"]["collection_name"])
        if col.count() < 3:
            logger.error(f"Collection has {col.count()} documents; need at least 3 to create vector index")
            raise ValueError("Insufficient documents for vector index")
        
        # Drop existing vector index if present
        for idx in col.indexes():
            if idx.get("name") == "vector_index":
                logger.info(f"Dropping existing vector_index (id={idx.get('id')})")
                col.delete_index(idx.get("id"))
                break
        
        cfg = {
            "type": "vector",
            "fields": [CONFIG["embedding"]["field"]],
            "params": {
                "metric": "cosine",
                "dimension": CONFIG["embedding"]["dimensions"],
                "nLists": CONFIG["search"]["vector_index_nlists"]
            },
            "name": "vector_index"
        }
        col.add_index(cfg)
        logger.info("Vector index created successfully")
        
        # Validate index
        vector_index = [
            idx for idx in col.indexes()
            if idx.get("name") == "vector_index" and idx.get("type") == "vector"
        ]
        if not vector_index:
            logger.error("Vector index not found after creation")
            raise ValueError("Vector index creation failed")
    
    except IndexCreateError as e:
        logger.exception(f"Vector index creation failed: {e}")
        raise

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    try:
        client = connect_arango()
        db = ensure_database(client)
        ensure_collection(db)
        ensure_arangosearch_view(db)
        logger.info("ArangoDB setup completed successfully")
    except Exception as e:
        logger.exception(f"ArangoDB setup failed: {e}")
        sys.exit(1)