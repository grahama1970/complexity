# src/complexity/beta/utils/arango_setup.py
import sys
import os
import requests
import torch
from typing import Dict, List, Tuple, Any, Optional
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

# Import EmbedderModel and CONFIG from the appropriate modules
try:
    from complexity.beta.rag.rag_classifier import EmbedderModel, DOC_PREFIX
    from complexity.beta.utils.config import CONFIG  # Import CONFIG from config.py
except ImportError:
    logger.error("Could not import required modules. Ensure PYTHONPATH includes 'src'.")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Validate environment
if not all(CONFIG["arango"].values()):
    missing = [k for k, v in CONFIG["arango"].items() if not v]
    logger.error(f"Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

# Cached EmbedderModel
_EmbedderModel_instance = None

def get_EmbedderModel(force_new=False):
    """Return singleton EmbedderModel or create a new one if requested."""
    global _EmbedderModel_instance
    if _EmbedderModel_instance is None or force_new:
        logger.info(f"Initializing EmbedderModel: {CONFIG['embedding']['model_name']}")
        _EmbedderModel_instance = EmbedderModel(CONFIG["embedding"]["model_name"])
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

def ensure_edge_collections(db):
    """Ensure edge collections for relationships exist."""
    try:
        # Get list of all collections in the database
        existing_collections = [c['name'] for c in db.collections()]
        
        # Create each edge collection if it doesn't exist
        edge_collections = ["prerequisites", "related_topics"]  # Define edge collections directly
        
        for collection_name in edge_collections:
            if collection_name not in existing_collections:
                logger.info(f"Creating edge collection '{collection_name}'")
                db.create_collection(collection_name, edge=True)
            logger.info(f"Edge collection '{collection_name}' ready")
    except Exception as e:
        logger.exception(f"Edge collection setup failed: {e}")
        sys.exit(1)

def ensure_graph(db):
    """Ensure named graph exists for traversals."""
    try:
        graph_name = "complexity_graph"
        
        # Get list of existing graphs
        existing_graphs = [g['name'] for g in db.graphs()]
        
        # Define edge collections directly
        edge_collections = ["prerequisites", "related_topics"]
        collection_name = CONFIG["search"]["collection_name"]
        
        if graph_name not in existing_graphs:
            logger.info(f"Creating graph '{graph_name}'")
            edge_definitions = []
            for edge_collection in edge_collections:
                edge_definitions.append({
                    "edge_collection": edge_collection,
                    "from_vertex_collections": [collection_name],
                    "to_vertex_collections": [collection_name]
                })
            db.create_graph(graph_name, edge_definitions)
        logger.info(f"Graph '{graph_name}' ready")
    except Exception as e:
        logger.exception(f"Graph setup failed: {e}")
        sys.exit(1)

def load_and_index_dataset(db: StandardDatabase, EmbedderModel=None) -> None:
    """Load dataset, embed texts with progress bars, and insert into collection."""
    logger.info("Loading dataset...")
    try:
        ds = load_dataset(
            CONFIG['dataset']['name'], split=CONFIG['dataset']['split'], trust_remote_code=True
        )
        if not isinstance(ds, Dataset):
            logger.error(f"Unexpected dataset type: {type(ds)}")
            sys.exit(1)
        
        # Get the EmbedderModel - either use the provided one or get/create the default one
        emb = EmbedderModel if EmbedderModel else get_EmbedderModel()
        
        # Log which embedding model is being used - safely check for attribute
        embedding_model_name = getattr(emb, 'model_name', CONFIG['embedding']['model_name'])
        logger.info(f"Generating embeddings using {embedding_model_name}")
        
        # Prepare data
        texts: List[str] = []
        docs: List[Dict[str, Any]] = []
        for item in tqdm(ds, desc="Preparing docs"):
            q = item.get('question')
            r = item.get('rating')
            if not q or r is None:
                continue
            try:
                label = 1 if float(r) >= 0.5 else 0
            except (ValueError, TypeError):
                continue
            texts.append(q)
            docs.append({'question': q, 'label': label, 'validated': True})

        # Embed in batches with tqdm
        batch_size = CONFIG["embedding"]["batch_size"]
        embeddings: List[List[float]] = []
        emb_pbar = tqdm(total=len(texts), desc="Embedding docs")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embs = emb.embed_batch(batch, prefix=DOC_PREFIX)
            embeddings.extend(batch_embs)
            emb_pbar.update(len(batch_embs))
        emb_pbar.close()

        if len(embeddings) != len(docs):
            logger.error(f"Embedding mismatch: {len(embeddings)} vs {len(docs)}")
            sys.exit(1)

        for doc, emb in zip(docs, embeddings):
            doc[CONFIG['embedding']['field']] = emb

        # Explicit message about truncating the collection
        logger.info(f"Truncating collection {CONFIG['search']['collection_name']} to replace existing embeddings with new embeddings")
        
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

def ensure_vector_index(db:StandardDatabase):
    """Ensure vector index after data insertion."""
    try:
        col = db.collection(CONFIG["search"]["collection_name"])
        doc_count = col.count()
        
        if doc_count < 3:
            logger.error(f"Collection has {doc_count} documents; need at least 3 to create vector index")
            sys.exit(1)

        # Log current indexes for debugging
        current_indexes = list(col.indexes())
        logger.info(f"Current indexes: {len(current_indexes)}")
        
        # Drop existing vector index if present
        vector_index = None
        for idx in current_indexes:
            if idx.get("type") == "vector":
                vector_index = idx
                logger.info(f"Found existing vector index: {idx.get('name')} (id={idx.get('id')})")
                
                # Check if index has correct dimensions
                params = idx.get("params", {})
                current_dimension = params.get("dimension", 0)
                if (current_dimension != CONFIG["embedding"]["dimensions"] or 
                    idx.get("fields", []) != [CONFIG["embedding"]["field"]]):
                    logger.info(f"Dropping incompatible vector index: current dimension={current_dimension}, needed={CONFIG['embedding']['dimensions']}")
                    col.delete_index(idx.get("id"))
                    vector_index = None
                else:
                    logger.info(f"Existing vector index is compatible")
                break
        
        # Create new vector index if needed
        if not vector_index:
            # Create new vector index with correct dimensions
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
            
            logger.info(f"Creating vector index with dimension {CONFIG['embedding']['dimensions']}")
            try:
                result = col.add_index(cfg)
                logger.info(f"Index creation result: {result}")
            except Exception as e:
                logger.exception(f"Exception creating vector index: {e}")
                raise

        # Validate index
        updated_indexes = list(col.indexes())
        vector_indexes = [
            idx for idx in updated_indexes
            if idx.get("type") == "vector" and CONFIG["embedding"]["field"] in idx.get("fields", [])
        ]
        
        if not vector_indexes:
            logger.error("Vector index not found after creation attempt")
            sys.exit(1)
            
        # Check the dimensions
        vector_idx = vector_indexes[0]
        params = vector_idx.get("params", {})
        current_dim = params.get("dimension", 0)
        if current_dim != CONFIG["embedding"]["dimensions"]:
            logger.error(f"Vector index dimension mismatch: current={current_dim}, needed={CONFIG['embedding']['dimensions']}")
            sys.exit(1)
            
        logger.info(f"Vector index verified successfully: dimension={current_dim}, metric={params.get('metric', 'unknown')}")
    
    except Exception as e:
        logger.exception(f"Vector index creation failed: {e}")
        sys.exit(1)

def classify_complexity(db: StandardDatabase, question: str, k: int = None, return_neighbors: bool = False) -> Tuple[int, float, bool, Optional[List[Dict]]]:
    """
    Classify question complexity using semantic search.
    
    Args:
        db: Database connection
        question: Question to classify
        k: Number of neighbors to consider
        return_neighbors: Whether to return neighbor documents
        
    Returns:
        Tuple of (label, confidence, auto_accept, neighbors)
    """
    k = k or CONFIG["classification"]["default_k"]
    try:
        emb = get_EmbedderModel().embed_batch([question])[0]
        
        collection_name = CONFIG["search"]["collection_name"]
        embedding_field = CONFIG["embedding"]["field"]
        
        if return_neighbors:
            # Query that returns full document information
            aql = f"""
            FOR doc IN {collection_name}
                LET score = COSINE_SIMILARITY(doc.{embedding_field}, @emb)
                SORT score DESC
                LIMIT @k
                RETURN {{ 
                    label: doc.label, 
                    score: score, 
                    question: doc.question,
                    id: doc._id,
                    embedding: doc.{embedding_field}
                }}
            """
        else:
            # Simpler query that only returns label and score
            aql = f"""
            FOR doc IN {collection_name}
                LET score = COSINE_SIMILARITY(doc.{embedding_field}, @emb)
                SORT score DESC
                LIMIT @k
                RETURN {{ label: doc.label, score: score }}
            """
            
        cursor = db.aql.execute(aql, bind_vars={"emb": emb, "k": k})
        results = list(cursor)
        if not results:
            logger.warning("No neighbors found")
            return (0, 0.0, False, []) if return_neighbors else (0, 0.0, False)
        
        votes = {0: 0.0, 1: 0.0}
        total = 0.0
        
        # Apply exponential weighting to emphasize closer matches
        exponent = 2.0  # Configurable, higher values give more weight to close matches
        
        for r in results:
            if r["score"] > 0:
                weight = r["score"] ** exponent  # Exponential weighting
                votes[r["label"]] += weight
                total += weight
                
        if total <= 0:
            return (0, 0.0, False, []) if return_neighbors else (0, 0.0, False)
            
        majority = max(votes, key=votes.get)
        confidence = votes[majority] / total
        auto_accept = confidence >= CONFIG["classification"]["confidence_threshold"] and len(results) >= k
        
        logger.info(f"Classification: label={majority}, confidence={confidence:.2f}, auto_accept={auto_accept}")
        
        if return_neighbors:
            return majority, confidence, auto_accept, results
        return majority, confidence, auto_accept
        
    except Exception as e:
        logger.exception(f"Classification failed: {e}")
        return (0, 0.0, False, []) if return_neighbors else (0, 0.0, False)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    try:
        # Set PyTorch memory config to avoid fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        client = connect_arango()
        db = ensure_database(client)
        ensure_collection(db)
        ensure_arangosearch_view(db)
        
        # Create a new EmbedderModel instance with the configured model
        EmbedderModel = EmbedderModel(CONFIG["embedding"]["model_name"])
        
        # Load and index data with the new EmbedderModel
        load_and_index_dataset(db, EmbedderModel=EmbedderModel)
        ensure_vector_index(db)
        
        # Test classification
        result = classify_complexity(db, "What is the capital of France?")
        logger.info(f"Test classification: {result}")
        
        logger.info("Setup completed successfully")
    except Exception as e:
        logger.exception(f"Setup failed: {e}")
        sys.exit(1)