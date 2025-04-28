# src/complexity/utils/arango_setup.py
import sys
import os
import requests
import torch
from typing import Dict, List, Tuple
from loguru import logger
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from arango.client import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import (
    DatabaseCreateError,
    CollectionCreateError,
    ViewCreateError,
    ViewDeleteError,
    IndexCreateError,
    DocumentInsertError,
)
from dotenv import load_dotenv

# Import EmbedderModel
try:
    from complexity.rag.rag_classifer import EmbedderModel, EMBEDDING_MODEL_NAME, DOC_PREFIX
except ImportError:
    logger.error("Could not import EmbedderModel. Ensure PYTHONPATH includes 'src'.")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "arango": {
        "host": os.getenv("ARANGO_HOST", "http://localhost:8529"),
        "user": os.getenv("ARANGO_USER", "root"),
        "password": os.getenv("ARANGO_PASSWORD", "openSesame"),
        "db_name": os.getenv("ARANGO_DB_NAME", "memory_bank"),
    },
    "dataset": {
        "name": "wesley7137/question_complexity_classification",
        "split": "train",
    },
    "embedding": {
        "dimensions": 768,  # nomic-embed-text-v1
        "field": "embedding",
        "batch_size": 32,
    },
    "search": {
        "collection_name": "complexity",
        "view_name": "complexity_view",
        "text_analyzer": "text_en",
        "vector_index_nlists": 8,
        "insert_batch_size": 1000,
    },
    "classification": {
        "default_k": 5,
        "confidence_threshold": 0.7,
    },
}

# Validate environment
if not all(CONFIG["arango"].values()):
    missing = [k for k, v in CONFIG["arango"].items() if not v]
    logger.error(f"Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

# Cached EmbedderModel
_EmbedderModel_instance = None

def get_EmbedderModel():
    """Return singleton EmbedderModel."""
    global _EmbedderModel_instance
    if _EmbedderModel_instance is None:
        logger.info(f"Initializing EmbedderModel: {EMBEDDING_MODEL_NAME}")
        _EmbedderModel_instance = EmbedderModel(EMBEDDING_MODEL_NAME)
    return _EmbedderModel_instance

def connect_arango():
    """Connect to ArangoDB."""
    logger.info(f"Connecting to ArangoDB at {CONFIG['arango']['host']}")
    try:
        client = ArangoClient(hosts=CONFIG["arango"]["host"])
        sys_db = client.db("_system", username=CONFIG["arango"]["user"], password=CONFIG["arango"]["password"])
        logger.info(f"Connected to ArangoDB version {sys_db.version()}")
        return client
    except Exception as e:
        logger.exception(f"Connection failed: {e}")
        sys.exit(1)

def ensure_database(client):
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
        sys.exit(1)

def ensure_collection(db):
    """Ensure collection exists."""
    try:
        name = CONFIG["search"]["collection_name"]
        if not db.has_collection(name):
            logger.info(f"Creating collection '{name}'")
            db.create_collection(name)
        logger.info(f"Collection '{name}' ready")
    except CollectionCreateError as e:
        logger.exception(f"Collection creation failed: {e}")
        sys.exit(1)

def ensure_arangosearch_view(db):
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
        sys.exit(1)

def load_and_index_dataset(db: StandardDatabase) -> None:
    """Load dataset, embed texts with progress bars, and insert into collection."""
    logger.info("Loading dataset...")
    try:
        ds = load_dataset(
            CONFIG['dataset']['name'], split=CONFIG['dataset']['split'], trust_remote_code=True
        )
        if not isinstance(ds, Dataset):
            logger.error(f"Unexpected dataset type: {type(ds)}")
            sys.exit(1)

        # TEsting with a small subset (remove late)
        # ds = ds.select(range(20))
        
        texts: List[str] = []
        docs: List[Dict[str, Any]] = []
        for item in tqdm(ds, desc="Preparing docs"):
            q = item.get('question'); r = item.get('rating')
            if not q or r is None:
                continue
            try:
                label = 1 if float(r) >= 0.5 else 0
            except (ValueError, TypeError):
                continue
            texts.append(q)
            docs.append({'question': q, 'label': label, 'validated': True})

        # embed in batches with tqdm
        batch_size = 32
        embeddings: List[List[float]] = []
        emb_pbar = tqdm(total=len(texts), desc="Embedding docs")
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embs = get_EmbedderModel().embed_batch(batch, prefix=DOC_PREFIX)
            embeddings.extend(batch_embs)
            emb_pbar.update(len(batch_embs))
        emb_pbar.close()

        if len(embeddings) != len(docs):
            logger.error(f"Embedding mismatch: {len(embeddings)} vs {len(docs)}")
            sys.exit(1)

        for doc, emb in zip(docs, embeddings):
            doc[CONFIG['embedding']['field']] = emb

        # bulk insert with tqdm
        col = db.collection(CONFIG['search']['collection_name'])
        col.truncate()
        ins_pbar = tqdm(total=len(docs), desc="Inserting docs")
        for i in range(0, len(docs), 1000):
            batch = docs[i:i + 1000]
            col.insert_many(batch, overwrite=True)
            ins_pbar.update(len(batch))
        ins_pbar.close()
        logger.info(f"Inserted {len(docs)} docs in batches of 1000")

    except Exception as err:
        logger.exception(f"Loading/indexing failed: {err}")
        sys.exit(1)

def ensure_vector_index(db):
    """Ensure vector index after data insertion."""
    try:
        col = db.collection(CONFIG["search"]["collection_name"])
        if col.count() < 3:
            logger.error(f"Collection has {col.count()} documents; need at least 3 to create vector index")
            sys.exit(1)

        # Drop existing vector index if present
        for idx in col.indexes():
            if idx.get("name") == "vector_index":
                logger.info(f"Dropping existing vector_index (id={idx.get('id')})")
                col.delete_index(idx.get("id"))
                break

        try:
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
            
            result = col.add_index(cfg)
            logger.info("Index creation result:", result)
        except Exception as e:
            logger.exception("Exception:", e)

        # Validate index
        vector_index = [
            idx for idx in col.indexes()
            if idx.get("name") == "vector_index" and idx.get("type") == "vector"
        ]
        if not vector_index:
            logger.error("Vector index not found after creation")
            sys.exit(1)
        # idx = vector_index[0]
        # Try both top-level and 'params' for parameters
        # params = idx.get("params", idx)
        # expected = {
        #     "metric": cfg["params"]["metric"],
        #     "dimension": cfg["params"]["dimension"],
        #     "nLists": cfg["params"]["nLists"]
        # }
        # for k, v in expected.items():
        #     if params.get(k) != v:
        #         logger.error(f"Index parameter mismatch for '{k}': expected {v}, got {params.get(k)}")
        #         sys.exit(1)
        # logger.info(f"Vector index validated: params={params}")
    
    except IndexCreateError as e:
        logger.exception(f"Vector index creation failed: {e}")
        sys.exit(1)


def classify_complexity(db: StandardDatabase, question: str, k: int = None) -> Tuple[int, float, bool]:
    """Classify question complexity using semantic search."""
    k = k or CONFIG["classification"]["default_k"]
    try:
        emb = get_EmbedderModel().embed_batch([question])[0]
        """NOTES:
         - FILTER must come AFTER SORT in AQL
         - APPROX_NEAR_COSINE MUST be used on the Collection, NOT the View
        """
        aql = f"""
        FOR doc IN {CONFIG['search']['collection_name']}
            LET score = APPROX_NEAR_COSINE(doc.{CONFIG['embedding']['field']}, @emb)
            SORT score DESC
            LIMIT @k
            RETURN {{ label: doc.label, score: score }}
        """
        cursor = db.aql.execute(aql, bind_vars={"emb": emb, "k": k})
        results = list(cursor)
        if not results:
            logger.warning("No neighbors found")
            return 0, 0.0, False
        votes = {0: 0.0, 1: 0.0}
        total = 0.0
        for r in results:
            if r["score"] > 0:
                votes[r["label"]] += r["score"]
                total += r["score"]
        if total <= 0:
            return 0, 0.0, False
        majority = max(votes, key=votes.get)
        confidence = votes[majority] / total
        auto_accept = confidence >= CONFIG["classification"]["confidence_threshold"] and len(results) >= k
        logger.info(f"Classification: label={majority}, confidence={confidence:.2f}, auto_accept={auto_accept}")
        return majority, confidence, auto_accept
    except Exception as e:
        logger.exception(f"Classification failed: {e}")
        return 0, 0.0, False

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    try:
        # Set PyTorch memory config to avoid fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        client = connect_arango()
        db = ensure_database(client)
        # ensure_collection(db)
        # ensure_arangosearch_view(db)
        # load_and_index_dataset(db)  # Insert data FIRST
        # ensure_vector_index(db)     # Create index AFTER data
        result = classify_complexity(db, "What is the capital of France?")
        logger.info(f"Test classification: {result}")
        logger.info("Setup completed successfully")
    except Exception as e:
        logger.exception(f"Setup failed: {e}")
        sys.exit(1)