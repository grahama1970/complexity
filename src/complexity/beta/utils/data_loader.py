from loguru import logger
import sys
from datasets import load_dataset
from tqdm.auto import tqdm
from complexity.beta.utils.config import CONFIG
from complexity.rag.rag_classifier import EmbedderModel, DOC_PREFIX
from complexity.utils.file_utils import load_env_file

def load_and_index_dataset(db, model_name: str):
    """Load dataset, balance classes, and index into ArangoDB with embeddings."""
    try:
        # Load environment variables
        load_env_file()
        
        # Load dataset
        logger.info("Loading dataset for indexing")
        dataset = load_dataset(
            CONFIG["dataset"]["name"],
            split=CONFIG["dataset"]["split"],
            trust_remote_code=True
        )
        logger.info(f"Original dataset size: {len(dataset)}")
        
        # Filter invalid entries
        valid_data = []
        skipped = 0
        for item in dataset:
            rating = item.get("rating")
            question = item.get("question")
            
            # Skip records where rating is None
            if rating is None:
                logger.warning(f"Skipping document with None rating: {item}")
                skipped += 1
                continue
                
            # Check for other invalid conditions
            if question is None or not isinstance(question, str):
                logger.warning(f"Skipping document with invalid question: {item}")
                skipped += 1
                continue
            
            try:
                float(rating)  # Ensure rating is numeric
                valid_data.append(item)
            except (ValueError, TypeError):
                logger.warning(f"Skipping document with non-numeric rating: {item}")
                skipped += 1
                continue
        
        if not valid_data:
            logger.error("No valid documents found in dataset")
            raise ValueError("Dataset contains no valid documents")
        
        logger.info(f"Filtered dataset size: {len(valid_data)} ({skipped} documents skipped)")
        
        # Balance classes
        labels = [1 if float(item["rating"]) >= 0.5 else 0 for item in valid_data]
        simple_items = [item for item, label in zip(valid_data, labels) if label == 0]
        complex_items = [item for item, label in zip(valid_data, labels) if label == 1]
        target_count = min(len(simple_items), len(complex_items))
        balanced_data = simple_items[:target_count] + complex_items[:target_count]
        logger.info(f"Balanced dataset: {len(balanced_data)} documents (Simple: {target_count}, Complex: {target_count})")
        
        # Initialize EmbedderModel and collection
        EmbedderModel = EmbedderModel(model_name=model_name)
        col = db.collection(CONFIG["search"]["collection_name"])
        col.truncate()  # Clear existing data
        
        # Index balanced data with embeddings
        batch_size = CONFIG["embedding"]["batch_size"]
        logger.remove()
        logger.add(tqdm_sink, level="INFO")
        logger.add(tqdm_sink, level="DEBUG")
        logger.add(sys.stderr, level="WARNING")
        for i in tqdm(range(0, len(balanced_data), batch_size), desc="Embedding docs"):
            batch = balanced_data[i:i + batch_size]
            texts = [item["question"] for item in batch]
            labels = [1 if float(item["rating"]) >= 0.5 else 0 for item in batch]
            embeddings = EmbedderModel.embed_batch(texts, prefix=DOC_PREFIX)
            docs = [
                {
                    "question": q,
                    "label": l,
                    "validated": True,
                    CONFIG["embedding"]["field"]: e # e.tolist()
                }
                for q, l, e in zip(texts, labels, embeddings)
            ]
            col.insert_many(docs)
        
        # Restore logging
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.info(f"Indexed {len(balanced_data)} documents into {CONFIG['search']['collection_name']}")
        
    except Exception as e:
        logger.exception(f"Failed to load and index dataset: {e}")
        raise

def tqdm_sink(message):
    """Custom loguru sink to discard INFO and DEBUG logs during tqdm."""
    if message.record["level"].name in ["INFO", "DEBUG"]:
        return
    tqdm.write(str(message), end="")