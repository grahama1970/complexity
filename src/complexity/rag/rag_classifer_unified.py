#!/usr/bin/env python3

import os
import time
from typing import List, Dict, Any
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from arango import ArangoClient
from loguru import logger
from tqdm.auto import tqdm
from tabulate import tabulate
from complexity.utils.file_utils import project_root, load_env_file
PROJECT_ROOT = project_root()
load_env_file()


##############################################################################
# 1) MAIN FUNCTION WITH CONFIG
##############################################################################

def main():
    """
    Main pipeline steps:
      1. Load & filter dataset (no train/val/test splitting).
      2. Initialize a single in-memory EmbedderModel.
      3. Concurrently embed dataset questions with "doc:" prefix.
      4. Store docs in ArangoDB + create ArangoSearch view.
      5. Classify new user questions with "query:" prefix (no repeated model loads).
      6. Tabulate inference results.
    """
    PROJECT_ROOT = project_root()
    config = load_env_file(PROJECT_ROOT / ".env")
    config = {
        # ===================== Dataset Info =====================
        "hf_dataset_name": "wesley7137/question_complexity_classification",
        "hf_token": os.getenv("HF_TOKEN", None),

        # ===================== ArangoDB Info =====================
        "arango_host": "http://localhost:8529",
        "arango_db": "verifaix",
        "arango_username": "root",
        "arango_password": "openSesame",
        "arango_collection": "questions_complexity",
        "arango_view": "questions_complexity_view",

        # ===================== ModernBert Model =====================
        # From https://huggingface.co/nomic-ai/modernbert-embed-base
        "embedding_model_name": "nomic-ai/modernbert-embed-base",

        # We'll prepend "doc: " for dataset, "query: " for inference
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",

        # ===================== Concurrency + Batching =====================
        "batch_size": 32,    # how many records per thread batch
        "max_workers": 4,    # number of threads for dataset embedding

        # ===================== Retrieval/Classification =====================
        "top_k_neighbors": 5,  # how many neighbors we retrieve for classification
    }

    try:
        # 1) Load & filter the dataset
        dataset = load_and_filter_dataset(config)

        # 2) Initialize the in-memory EmbedderModel (loaded once!)
        EmbedderModel = EmbedderModel(config["embedding_model_name"])

        # 3) Embed dataset concurrently (using doc: prefix), store label=0/1
        docs = embed_dataset_concurrent(dataset, EmbedderModel, config)

        # 4) Store docs in ArangoDB + create an ArangoSearch view
        db = connect_arango_db(config)
        store_docs_in_arango(db, docs, config)
        create_arangosearch_view(db, config)

        # 5) Classify some test questions (using query: prefix)
        questions = [
            "What is the most common color of an apple?",
            "Explain the process of nuclear fission in detail.",
            "What is the half-life of uranium-238?",
            "How does a nuclear reactor generate electricity?",
            "What is the capital of France?",
            "Give me a list of all the planets in the solar system.",
        ]

        results_table = [["Question", "Classification", "Time (ms)"]]
        for question in questions:
            start_time = time.perf_counter()
            classification = classify_question(question, db, EmbedderModel, config)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            results_table.append([question, classification, f"{elapsed_ms:.2f}"])

        logger.info("\n" + tabulate(results_table, headers="firstrow", tablefmt="grid"))

    except Exception as e:
        logger.error(f"Fatal error in main pipeline: {e}")


##############################################################################
# 2) MODERNBERT EmbedderModel (LOAD ONCE)
##############################################################################

class EmbedderModel:
    """
    A simple class that loads the ModernBert model once. 
    Provides methods for embedding single texts or batches, 
    so we don't reload the model for each call.
    """
    def __init__(self, model_name: str):
        logger.info(f"Initializing EmbedderModel with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            logger.info("Using GPU for embedding.")
        else:
            logger.info("Using CPU for embedding.")

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text synchronously. 
        Returns a Python list of floats (L2-normalized).
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            sum_hidden = (last_hidden * attention_mask).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            mean_emb = sum_hidden / torch.clamp(sum_mask, min=1e-9)
            emb = F.normalize(mean_emb, p=2, dim=1)
            emb_list = emb[0].cpu().tolist()
        return emb_list

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts at once. 
        Returns a list of lists of floats.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            sum_hidden = (last_hidden * attention_mask).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            mean_emb = sum_hidden / torch.clamp(sum_mask, min=1e-9)
            emb = F.normalize(mean_emb, p=2, dim=1)
            emb_list = emb.cpu().tolist()
        return emb_list


##############################################################################
# 3) LOAD & FILTER DATASET
##############################################################################

def load_and_filter_dataset(config: Dict[str, Any]):
    """
    Loads the dataset from Hugging Face, filters invalid rating examples, 
    and returns a list of dicts: {"question": str, "rating": float}.
    """
    ds_name = config["hf_dataset_name"]
    hf_token = config["hf_token"]

    logger.info(f"Loading dataset '{ds_name}' from Hugging Face...")
    raw = load_dataset(ds_name, token=hf_token)
    if "train" not in raw:
        raise ValueError("Dataset must have a 'train' split.")

    full_ds = raw["train"]
    logger.info(f"Initial dataset size: {len(full_ds)}")

    def is_valid(example):
        r = example.get("rating")
        return (r is not None and isinstance(r, (float, int)) and 0 <= r <= 1)

    filtered = full_ds.filter(is_valid)
    logger.info(f"Filtered dataset: {len(filtered)}/{len(full_ds)} valid records")

    # Convert to a plain list of dicts
    result = []
    for ex in filtered:
        result.append({
            "question": ex["question"],
            "rating": ex["rating"]
        })
    return result


##############################################################################
# 4) CONCURRENT EMBEDDING FOR DATASET
##############################################################################

def embed_dataset_concurrent(dataset: List[Dict[str, Any]], EmbedderModel: EmbedderModel, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Embeds each dataset question with "doc:" prefix, using a ThreadPoolExecutor 
    + TQDM. Batches are processed in parallel. 
    Output docs: {"question", "rating", "label", "embedding"} 
    where label=1 if rating>0.5 else 0.
    """
    doc_prefix = config["doc_prefix"]
    batch_size = config["batch_size"]
    max_workers = config["max_workers"]
    total_len = len(dataset)
    logger.info(f"Embedding {total_len} records with concurrency; batch_size={batch_size}, max_workers={max_workers}")

    # We'll produce a final list of docs
    results = [None] * total_len

    def process_chunk(start_idx: int):
        """
        Embed a chunk of dataset questions in a single batch.
        """
        end_idx = min(start_idx + batch_size, total_len)
        chunk = dataset[start_idx:end_idx]
        texts = [f"{doc_prefix}{item['question']}" for item in chunk]
        emb_batch = EmbedderModel.embed_batch(texts)

        chunk_out = []
        for i, emb in enumerate(emb_batch):
            idx = start_idx + i
            rating = dataset[idx]["rating"]
            label = 1 if rating > 0.5 else 0
            chunk_out.append((idx, {
                "question": dataset[idx]["question"],
                "rating": rating,
                "label": label,
                "embedding": emb
            }))
        return chunk_out

    # Launch thread for each chunk
    futures = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    for start_idx in range(0, total_len, batch_size):
        futures.append(executor.submit(process_chunk, start_idx))

    # Collect results with TQDM
    with tqdm(total=len(futures), desc="Embedding Dataset") as pbar:
        for fut in as_completed(futures):
            chunk_result = fut.result()
            for idx, doc in chunk_result:
                results[idx] = doc
            pbar.update(1)

    executor.shutdown(wait=True)

    # Filter out any Nones
    final_docs = [r for r in results if r is not None]
    logger.info(f"Concurrent embedding complete: {len(final_docs)} docs embedded.")
    return final_docs


##############################################################################
# 5) ARANGODB CONNECTION & STORAGE
##############################################################################

def connect_arango_db(config: Dict[str, Any]):
    """
    Connect to ArangoDB and return a database handle.
    """
    client = ArangoClient(hosts=config["arango_host"])
    db = client.db(
        config["arango_db"],
        username=config["arango_username"],
        password=config["arango_password"]
    )
    return db

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


##############################################################################
# 6) CREATE ARANGOSEARCH VIEW
##############################################################################

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


##############################################################################
# 7) CLASSIFICATION VIA RETRIEVAL
##############################################################################

def classify_question(question_text: str, db, EmbedderModel: EmbedderModel, config: Dict[str, Any]) -> str:
    """
    Classify the question as 'Simple' or 'Complex' by:
      1) Embedding question with "query: "
      2) Retrieving top neighbors (BM25 + embedding)
      3) Majority vote on label => return "Simple" or "Complex"
    """
    top_docs = retrieve_top_docs(question_text, db, EmbedderModel, config)
    if not top_docs:
        return "Unknown"

    labels = [doc["doc"].get("label", 0) for doc in top_docs]
    num_complex = sum(1 for l in labels if l == 1)
    num_simple = len(labels) - num_complex

    if num_complex > num_simple:
        return "Complex"
    elif num_simple > num_complex:
        return "Simple"
    else:
        # Tie => pick "Complex" or "Simple"
        return "Complex"


def retrieve_top_docs(question_text: str, db, EmbedderModel: EmbedderModel, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run an AQL query merging BM25 + COSINE_SIMILARITY from embedding.
    """
    view_name = config["arango_view"]
    top_k = config["top_k_neighbors"]
    query_prefix = config["query_prefix"]

    # 1) Embed the question once
    query_emb = EmbedderModel.embed_text(f"{query_prefix}{question_text}")

    # 2) AQL
    aql = f"""
    LET embedding_results = (
      FOR doc IN {view_name}
        LET similarity = COSINE_SIMILARITY(doc.embedding, @emb)
        FILTER similarity >= 0
        SORT similarity DESC
        LIMIT @top_k
        RETURN {{
          doc: doc,
          similarity_score: similarity,
          bm25_score: 0
        }}
    )

    LET bm25_results = (
      FOR doc IN {view_name}
        SEARCH ANALYZER(
          doc.question IN TOKENS(@text, "text_en"),
          "text_en"
        )
        LET bm25_score = BM25(doc)
        SORT bm25_score DESC
        LIMIT @top_k
        RETURN {{
          doc: doc,
          similarity_score: 0,
          bm25_score: bm25_score
        }}
    )

    LET merged_results = (
      FOR result IN UNION_DISTINCT(embedding_results, bm25_results)
        COLLECT doc = result.doc INTO group = result
        LET similarity_score = MAX(group[*].similarity_score)
        LET bm25_score       = MAX(group[*].bm25_score)
        RETURN {{
          "doc": doc,
          "similarity_score": similarity_score,
          "bm25_score": bm25_score
        }}
    )

    LET final_results = (
      FOR mr IN merged_results
        SORT mr.similarity_score DESC, mr.bm25_score DESC
        LIMIT @top_k
        RETURN mr
    )

    RETURN final_results
    """

    bind_vars = {
        "emb": query_emb,
        "text": question_text,
        "top_k": top_k
    }

    cursor = db.aql.execute(aql, bind_vars=bind_vars)
    results_list = list(cursor)
    if results_list and isinstance(results_list[0], list):
        return results_list[0]
    else:
        return []


##############################################################################
# LAUNCH
##############################################################################

if __name__ == "__main__":
    main()
