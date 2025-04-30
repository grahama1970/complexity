# src/complexity/beta/utils/data_loader.py
"""
Module Description:
Provides functions for loading the question complexity dataset, filtering invalid entries,
balancing classes, generating embeddings, and indexing the processed data into ArangoDB.
Includes a custom loguru sink for integration with tqdm progress bars.

Links:
- Hugging Face Datasets: https://huggingface.co/docs/datasets/
- Loguru: https://loguru.readthedocs.io/en/stable/
- tqdm: https://tqdm.github.io/
- python-arango Driver: https://python-arango.readthedocs.io/en/latest/

Sample Input/Output:

- load_and_index_dataset(db: StandardDatabase, model_name: str):
  - Input: ArangoDB StandardDatabase instance, embedding model name string.
  - Output: None. (Populates the configured ArangoDB collection).
"""
from loguru import logger
import sys
from typing import Any # Import Any for loguru Record type hint
from datasets import load_dataset, Dataset # Import Dataset type
from tqdm.auto import tqdm
from arango.database import StandardDatabase # Import StandardDatabase type hint

from complexity.beta.utils.config import CONFIG
# Correct import path assuming rag_classifier is in beta.rag
from complexity.beta.rag.rag_classifier import EmbedderModel, DOC_PREFIX
# Import necessary setup functions for standalone execution
from complexity.beta.utils.arango_setup import connect_arango, ensure_database, ensure_collection
# Removed unused import: from complexity.utils.file_utils import load_env_file

def load_and_index_dataset(db: StandardDatabase, model_name: str) -> None:
    """Load dataset, balance classes, and index into ArangoDB with embeddings."""
    try:
        # Environment variables are loaded by config.py or arango_setup.py now
        # load_env_file() # Removed redundant call

        # Load dataset
        logger.info("Loading dataset for indexing")
        dataset = load_dataset(
            CONFIG["dataset"]["name"],
            split=CONFIG["dataset"]["split"],
            trust_remote_code=True
        )
        # Add type check for dataset length
        dataset_len = 0
        if isinstance(dataset, (Dataset, list)): # Check if it's a type with len()
             dataset_len = len(dataset)
        logger.info(f"Original dataset size: {dataset_len}")
        
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

def tqdm_sink(message: Any) -> None: # Added type hint for loguru Record (approximated with Any)
    """Custom loguru sink to discard INFO and DEBUG logs during tqdm."""
    if message.record["level"].name in ["INFO", "DEBUG"]:
        return
    tqdm.write(str(message), end="")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    validation_passed = True
    validation_failures = {}

    # --- Expected Values ---
    EXPECTED_COLLECTION_NAME = CONFIG["search"]["collection_name"]
    MODEL_NAME_FOR_TEST = CONFIG["embedding"]["model_name"]

    db_instance: Optional[StandardDatabase] = None

    try:
        logger.info("Connecting to ArangoDB for validation...")
        client = connect_arango()
        db_instance = ensure_database(client)
        logger.info(f"Connected to database: {db_instance.name}")

        # Ensure collection exists before loading
        ensure_collection(db_instance)
        if not db_instance.has_collection(EXPECTED_COLLECTION_NAME):
             logger.error(f"Prerequisite failed: Collection '{EXPECTED_COLLECTION_NAME}' does not exist.")
             raise RuntimeError(f"Collection '{EXPECTED_COLLECTION_NAME}' missing.")

        # --- Load and Index Data ---
        logger.info(f"Running load_and_index_dataset with model: {MODEL_NAME_FOR_TEST}")
        load_and_index_dataset(db_instance, MODEL_NAME_FOR_TEST)

        # --- Validation ---
        logger.info(f"Validating collection '{EXPECTED_COLLECTION_NAME}' content...")
        col = db_instance.collection(EXPECTED_COLLECTION_NAME)
        doc_count = col.count()
        assert isinstance(doc_count, int) # Assert type for safety

        if doc_count == 0:
            validation_passed = False
            validation_failures["document_count"] = {"expected": "> 0", "actual": 0}
            logger.error("Validation Error: Collection is empty after loading.")
        else:
            logger.info(f"Validation Info: Collection contains {doc_count} documents.")
            # Optional: Add check for embedding field presence in a sample doc
            sample_doc = col.random()
            if sample_doc and CONFIG["embedding"]["field"] not in sample_doc:
                 validation_passed = False
                 validation_failures["embedding_field"] = {"expected": f"Field '{CONFIG['embedding']['field']}'", "actual": "Not found in sample doc"}
                 logger.error(f"Validation Error: Embedding field '{CONFIG['embedding']['field']}' missing in sample document.")
            elif sample_doc:
                 logger.info(f"Validation Info: Embedding field '{CONFIG['embedding']['field']}' found in sample document.")


    except Exception as e:
        validation_passed = False
        validation_failures["runtime_error"] = str(e)
        logger.exception(f"Validation failed with runtime error: {e}")

    # --- Final Reporting ---
    if validation_passed:
        print("✅ VALIDATION COMPLETE - load_and_index_dataset executed and collection contains data.")
        logger.success("Standalone execution and validation successful.")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Issues detected during data loading or validation.")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            if isinstance(details, dict):
                 print(f"  - {field}: Expected: {details.get('expected', 'N/A')}, Got: {details.get('actual', 'N/A')}")
            else:
                 print(f"  - {field}: {details}")
        logger.error("Standalone execution and validation failed.")
        sys.exit(1)