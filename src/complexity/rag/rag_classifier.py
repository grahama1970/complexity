#!/usr/bin/env python3
"""
Module: rag_classifier.py
Description: Classifies questions as 'Simple' or 'Complex' using a RAG approach.
             It embeds questions using a sentence transformer model (ModernBert)
             and retrieves similar questions from an ArangoDB database to perform
             a majority vote classification based on pre-assigned labels.
             This script assumes the ArangoDB database, collection, and view
             have already been set up by `src/complexity/utils/arango_setup.py`.

Links:
- python-arango: https://docs.python-arango.com
- sentence-transformers: https://sbert.net
- ModernBert (nomic-ai/modernbert-embed-base): https://huggingface.co/nomic-ai/modernbert-embed-base

Sample Input (Environment Variables):
export ARANGO_HOST="http://localhost:8529"
export ARANGO_DB_NAME="complexity" # Or memory_bank, ensure consistency
export ARANGO_COLLECTION_NAME="complexity_docs" # Or complexity, ensure consistency
export ARANGO_VIEW_NAME="complexity_view"
export ARANGO_USER="root"
export ARANGO_PASSWORD="your_password" # Or leave empty if no password
export HF_TOKEN="your_hf_token" # Optional, for private datasets

Sample Output (Printed Table):
+-----------------------------------------------------+----------------+-------------+
| Question                                            | Classification | Time (ms)   |
+=====================================================+================+=============+
| What is the most common color of an apple?          | Simple         | 150.23      |
| Explain the process of nuclear fission in detail.   | Complex        | 180.45      |
| ...                                                 | ...            | ...         |
+-----------------------------------------------------+----------------+-------------+
"""

import os
import sys
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch.nn.functional as F
import torch

from transformers import AutoTokenizer, AutoModel
from arango import ArangoClient # Keep ArangoClient for type hinting
from arango.database import StandardDatabase # Import for type hinting
from loguru import logger
from tqdm.auto import tqdm
from tabulate import tabulate
from dotenv import load_dotenv
from pathlib import Path # Import Path

# Assuming file_utils and arango_utils are in paths accessible via PYTHONPATH
# Correct imports based on file_utils.py content
from complexity.utils.file_utils import load_text_file, get_project_root
from complexity.rag.database.arango_utils import initialize_database # Keep initialize_database

# Load environment variables from .env file if it exists
load_dotenv()

# Get project root dynamically
PROJECT_ROOT = get_project_root()
AQL_FILE_PATH = PROJECT_ROOT / "src/complexity/rag/database/aql/rag_classifier.aql"

# --- Configuration from Environment Variables ---
# Use defaults consistent with arango_setup.py and examples/arangodb/config.py
ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
# Use the DB name from .env (memory_bank) as the primary source, fallback to 'complexity'
ARANGO_DB_NAME = os.getenv("ARANGO_DB_NAME", "complexity")
# Use collection/view names consistent with arango_setup.py
ARANGO_COLLECTION_NAME = os.getenv("ARANGO_COLLECTION_NAME", "complexity") # Align default with arango_setup.py
ARANGO_VIEW_NAME = os.getenv("ARANGO_VIEW_NAME", "complexity_view") # Align default with arango_setup.py
# Get user/password, defaulting like examples/arangodb/config.py
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "") # Default to empty string if not set

# Log if defaults are being used for credentials (optional but helpful)
if ARANGO_USER == "root" and not os.getenv("ARANGO_USER"):
     logger.info("ARANGO_USER not set, using default 'root'.")
if ARANGO_PASSWORD == "" and not os.getenv("ARANGO_PASSWORD"):
     logger.info("ARANGO_PASSWORD not set, using default empty password.")


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1") # Changed model
DOC_PREFIX = os.getenv("DOC_PREFIX", "search_document: ")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "search_query: ")

TOP_K_NEIGHBORS = int(os.getenv("TOP_K_NEIGHBORS", "3")) # Changed default k from 5 to 3
# --- End Configuration ---


##############################################################################
# 1) MAIN FUNCTION
##############################################################################

def main():
    """
    Main pipeline steps:
      1. Initialize the in-memory ModernBertEmbedder.
      2. Initialize connection to the existing ArangoDB.
      3. Verify required ArangoDB collection and view exist.
      4. Classify new user questions with QUERY_PREFIX using retrieval.
      5. Tabulate and validate inference results.
    """
    logger.info(f"Attempting to connect to ArangoDB: host='{ARANGO_HOST}', db='{ARANGO_DB_NAME}', user='{ARANGO_USER}'")
    try:
        # 1) Initialize the in-memory ModernBertEmbedder (loaded once!)
        embedder = ModernBertEmbedder(EMBEDDING_MODEL_NAME)

        # 2) Initialize connection to ArangoDB
        # Construct config dict for initialize_database using loaded env vars
        arango_config = {
            "hosts": [ARANGO_HOST],
            "db_name": ARANGO_DB_NAME,
            "username": ARANGO_USER,
            "password": ARANGO_PASSWORD, # Pass the value directly (could be "" from getenv)
        }
        db: StandardDatabase | None = initialize_database(arango_config)
        if db is None:
             # initialize_database now logs the specific error
             logger.error("Failed to initialize database connection (see previous error). Exiting.")
             sys.exit(1)
         # Success message is handled within initialize_database

        # 3) Verify collection and view exist using iteration (compatible with python-arango v8+)
        logger.info(f"Verifying collection '{ARANGO_COLLECTION_NAME}' exists in db '{ARANGO_DB_NAME}'...")
        try:
            existing_collections = [c['name'] for c in db.collections()]
            if ARANGO_COLLECTION_NAME not in existing_collections:
                logger.error(f"ArangoDB collection '{ARANGO_COLLECTION_NAME}' not found in db '{ARANGO_DB_NAME}'. Please run arango_setup.py first.")
                sys.exit(1)
        except Exception as e:
             logger.exception(f"Error listing collections in database '{ARANGO_DB_NAME}': {e}")
             sys.exit(1)


        logger.info(f"Verifying view '{ARANGO_VIEW_NAME}' exists in db '{ARANGO_DB_NAME}'...")
        try:
            existing_views = [v['name'] for v in db.views()]
            if ARANGO_VIEW_NAME not in existing_views:
                 logger.error(f"ArangoDB view '{ARANGO_VIEW_NAME}' not found in db '{ARANGO_DB_NAME}'. Please run arango_setup.py first.")
                 sys.exit(1)
        except Exception as e:
             logger.exception(f"Error listing views in database '{ARANGO_DB_NAME}': {e}")
             sys.exit(1)

        logger.info(f"Confirmed collection '{ARANGO_COLLECTION_NAME}' and view '{ARANGO_VIEW_NAME}' exist.")


        # 4) Classify some test questions (using query: prefix)
        questions_to_classify = [
            "What is the most common color of an apple?", # Expected: Simple
            "Explain the process of nuclear fission in detail.", # Expected: Complex
            "What is the half-life of uranium-238?", # Expected: Complex (specific knowledge)
            "How does a nuclear reactor generate electricity?", # Expected: Complex
            "What is the capital of France?", # Expected: Simple
            "Give me a list of all the planets in the solar system.", # Expected: Simple
        ]
        # Define expected results for validation
        EXPECTED_RESULTS = {
            "What is the most common color of an apple?": "Simple",
            "Explain the process of nuclear fission in detail.": "Complex",
            "What is the half-life of uranium-238?": "Complex",
            "How does a nuclear reactor generate electricity?": "Complex",
            "What is the capital of France?": "Simple",
            "Give me a list of all the planets in the solar system.": "Simple",
        }


        # Prepare config dict for classification functions
        classification_config = {
            "arango_view": ARANGO_VIEW_NAME,
            "top_k_neighbors": TOP_K_NEIGHBORS,
            "query_prefix": QUERY_PREFIX,
            "aql_file_path": AQL_FILE_PATH # Use constant defined above
        }

        results_table = [["Question", "Classification", "Expected", "Time (ms)"]]
        validation_passed = True
        expected_columns = 4 # Added Expected column
        actual_results = {}

        logger.info(f"Classifying {len(questions_to_classify)} questions...")
        for question in tqdm(questions_to_classify, desc="Classifying Questions"):
            start_time = time.perf_counter()
            classification = classify_question(question, db, embedder, classification_config)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            expected_classification = EXPECTED_RESULTS.get(question, "N/A")
            results_table.append([question, classification, expected_classification, f"{elapsed_ms:.2f}"])
            actual_results[question] = classification

            # Basic Validation during loop
            if classification not in {'Simple', 'Complex', 'Unknown'}:
                logger.error(f"Validation Error: Unexpected classification value '{classification}' for question: '{question}'")
                validation_passed = False
            # Stricter check: fail if mismatch against known expected result
            if expected_classification != "N/A" and classification != expected_classification:
                 logger.error(f"Validation Mismatch: For '{question}', Expected '{expected_classification}', Got '{classification}'")
                 validation_passed = False


        # 5) Tabulate and Validate Results
        logger.info("Classification Results:\n" + tabulate(results_table, headers="firstrow", tablefmt="grid"))

        # --- Detailed Validation ---
        if not results_table or len(results_table) != len(questions_to_classify) + 1:
             logger.error(f"Validation Error: Expected {len(questions_to_classify)} result rows, but got {len(results_table) - 1}")
             validation_passed = False
        if len(results_table[0]) != expected_columns:
             logger.error(f"Validation Error: Expected {expected_columns} columns in header, but got {len(results_table[0])}")
             validation_passed = False
        for i, row in enumerate(results_table[1:], 1):
             if len(row) != expected_columns:
                 logger.error(f"Validation Error: Row {i} - Expected {expected_columns} columns, but got {len(row)}: {row}")
                 validation_passed = False
                 break # Only report first malformed row

        # Check if all questions were processed
        if len(actual_results) != len(questions_to_classify):
            logger.error(f"Validation Error: Processed {len(actual_results)} questions, expected {len(questions_to_classify)}")
            validation_passed = False

        # Final check on classification values (already done in loop, but good to have final check)
        all_classifications_valid = all(c in {'Simple', 'Complex', 'Unknown'} for c in actual_results.values())
        if not all_classifications_valid:
             logger.error("Validation Error: One or more classification results had invalid values.")
             validation_passed = False

        # Check against expected results (already done in loop, this is redundant but safe)
        mismatches = []
        for q, expected in EXPECTED_RESULTS.items():
            actual = actual_results.get(q)
            if actual != expected:
                mismatches.append(f"'{q}' (Expected: {expected}, Got: {actual})")
        if mismatches:
             # This was already logged as ERROR in the loop, changing to warning here to avoid duplicate exit trigger
             logger.warning(f"{len(mismatches)} classification mismatches found (see details above).")
             # validation_passed = False # Already set in loop


        if validation_passed:
            print("\n✅ VALIDATION COMPLETE - Results table structure and classification values match expected.")
            sys.exit(0)
        else:
            print("\n❌ VALIDATION FAILED - See errors above.")
            sys.exit(1)


    except Exception as e:
        logger.exception(f"Fatal error in main pipeline: {e}") # Use logger.exception for traceback
        sys.exit(1)


##############################################################################
# 2) MODERNBERT EMBEDDER (LOAD ONCE)
##############################################################################

class ModernBertEmbedder:
    """
    A simple class that loads the ModernBert model once.
    Provides methods for embedding single texts or batches,
    so we don't reload the model for each call. Handles prefixing internally.
    Uses the specified model name (e.g., "nomic-ai/modernbert-embed-base").
    """
    def __init__(self, model_name: str):
        logger.info(f"Initializing ModernBertEmbedder with model: {model_name}")
        try:
            # Ensure correct model name is used and trust remote code
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model.eval()

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.model = self.model.to(self.device)
                logger.info("Using GPU for embedding.")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for embedding.")
        except Exception as e:
            logger.exception(f"Failed to load model '{model_name}'. Please check model name and connectivity.")
            raise # Re-raise after logging

    def _embed_core(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Core embedding logic, takes tokenized inputs."""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            sum_hidden = (last_hidden * attention_mask).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            # Clamp sum_mask to avoid division by zero
            mean_emb = sum_hidden / torch.clamp(sum_mask, min=1e-9)
            # L2 Normalize
            emb = F.normalize(mean_emb, p=2, dim=1)
        return emb

    def embed_text(self, text: str, prefix: str = "") -> List[float]:
        """
        Embed a single text synchronously with the specified prefix.
        Returns a Python list of floats (L2-normalized).
        Prefix should be "search_document: " or "search_query: ".
        """
        full_text = f"{prefix}{text}"
        # Use max_length suitable for the model (e.g., 512 for BERT variants)
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        emb = self._embed_core(inputs)
        emb_list = emb[0].cpu().tolist()
        return emb_list

    def embed_batch(self, texts: List[str], prefix: str = "") -> List[List[float]]:
        """
        Embed multiple texts at once with the specified prefix.
        Returns a list of lists of floats.
        Prefix should be "search_document: " or "search_query: ".
        """
        full_texts = [f"{prefix}{text}" for text in texts]
        # Use max_length suitable for the model
        inputs = self.tokenizer(full_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        emb = self._embed_core(inputs)
        emb_list = emb.cpu().tolist()
        return emb_list


##############################################################################
# 3) LOAD & FILTER DATASET (REMOVED - Handled by arango_setup.py)
##############################################################################

# def load_and_filter_dataset(...) -> ... :
#     """ ... """
#     # This function is no longer needed here as data loading and embedding
#     # are assumed to be handled by the arango_setup.py script.


##############################################################################
# 4) CONCURRENT EMBEDDING FOR DATASET (REMOVED - Handled by arango_setup.py)
##############################################################################

# def embed_dataset_concurrent(...) -> ... :
#     """ ... """
#     # This function is no longer needed here as data loading and embedding
#     # are assumed to be handled by the arango_setup.py script.


##############################################################################
# 5) CLASSIFICATION VIA RETRIEVAL
##############################################################################

def classify_question(
    question_text: str,
    db: StandardDatabase, # Type hint for Arango DB connection
    embedder: ModernBertEmbedder,
    config: Dict[str, Any]
) -> str:
    """
    Classify the question as 'Simple' or 'Complex' by:
      1) Embedding the question with the query prefix.
      2) Retrieving top_k neighbors using a combined BM25 and vector similarity search via AQL.
      3) Performing a majority vote on the 'label' of the retrieved neighbors.
    Returns 'Simple', 'Complex', or 'Unknown' if retrieval fails or there's a tie.
    """
    try:
        # Retrieve top documents using the provided config
        top_docs = retrieve_top_docs(question_text, db, embedder, config)
        if not top_docs:
            logger.warning(f"No similar documents found for question: '{question_text}'")
            return "Unknown" # Return Unknown if no docs are retrieved

        # Extract labels from retrieved documents, default to 0 if 'label' is missing
        # Assumes the documents stored by arango_setup.py have a 'label' field (0 or 1)
        labels = [doc["doc"].get("label", 0) for doc in top_docs]
        num_complex = sum(1 for l in labels if l == 1)
        num_simple = len(labels) - num_complex

        # Majority vote logic
        if num_complex > num_simple:
            return "Complex"
        elif num_simple > num_complex:
            return "Simple"
        else:
            # Handle ties - currently defaults to Complex, could be configured
            logger.debug(f"Tie ({num_simple} Simple vs {num_complex} Complex) for: '{question_text}'. Defaulting to Complex.")
            return "Complex" # Consistent tie-breaking
    except Exception as e:
        logger.exception(f"Error during classification for question '{question_text}': {e}")
        return "Unknown" # Return Unknown in case of error during classification


def retrieve_top_docs(
    question_text: str,
    db: StandardDatabase, # Type hint
    embedder: ModernBertEmbedder,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Run the AQL query to retrieve top_k documents based on combined BM25 and cosine similarity.
    Requires 'arango_view', 'top_k_neighbors', 'query_prefix', 'aql_file_path' in config.
    Formats the view name into the AQL query string.
    """
    # Extract necessary config values
    view_name = config["arango_view"]
    top_k = config["top_k_neighbors"]
    query_prefix = config["query_prefix"]
    aql_file_path = config["aql_file_path"]

    try:
        # 1) Embed the question with the query prefix
        query_emb = embedder.embed_text(question_text, prefix=query_prefix)

        # 2) Load AQL query template from file
        aql_template = load_text_file(aql_file_path)

        # 3) Format the AQL query string with the actual view name
        formatted_aql_query = aql_template.format(view_name=view_name)

        # 4) Prepare bind variables (view name is now part of the query string)
        bind_vars = {
            "emb": query_emb,
            # "text": question_text, # Removed as AQL query no longer uses @text
            "top_k": top_k
        }

        # 5) Execute the formatted AQL query
        logger.debug(f"Executing formatted AQL query on view '{view_name}' with k={top_k}")
        cursor = db.aql.execute(formatted_aql_query, bind_vars=bind_vars)

        # The simplified AQL query returns documents directly. Consume the cursor into a list.
        try:
            actual_docs = list(cursor) # Convert the cursor iterator to a list of documents
            logger.debug(f"Retrieved {len(actual_docs)} documents for question: '{question_text}'")
            return actual_docs
        except StopIteration: # This exception is unlikely with list() but kept for safety
             logger.debug(f"Retrieved 0 documents (empty cursor) for question: '{question_text}'")
             return []
        except Exception as e:
             # Catch potential errors during cursor iteration
             logger.exception(f"Error processing cursor results for question '{question_text}': {e}")
             return []

    except FileNotFoundError:
        logger.error(f"AQL query file not found at: {aql_file_path}")
        return []
    except Exception as e:
        # Log the exception with traceback for better debugging
        logger.exception(f"Error retrieving documents for question '{question_text}': {e}")
        return []


##############################################################################
# LAUNCH
##############################################################################

if __name__ == "__main__":
    # Configure Loguru for clear console output
    logger.remove() # Remove default handler to avoid duplicate messages
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level:<5} | {message}", # Simple format
        level="INFO", # Default level
        colorize=True
    )
    # Optional: Add file logger for debugging
    # logger.add("logs/rag_classifier_{time}.log", level="DEBUG", rotation="1 day")

    logger.info("Starting RAG Classifier...")
    # Credential checks moved to top after loading .env

    main()
