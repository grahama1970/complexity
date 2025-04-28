import sys
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from arango.database import StandardDatabase
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from tabulate import tabulate
from complexity.beta.utils.config import CONFIG
from complexity.beta.rag.rag_classifier import EmbedderModel, DOC_PREFIX
from complexity.beta.utils.classifier import classify_complexity, get_EmbedderModel
from complexity.beta.utils.arango_setup import (
    connect_arango, 
    ensure_database, 
    ensure_collection, 
    ensure_arangosearch_view, 
    ensure_vector_index,
    load_and_index_dataset
)
from complexity.utils.file_utils import get_project_root, load_env_file
from pathlib import Path
import os
import time

# Load environment variables
PROJECT_ROOT = get_project_root()
load_env_file()
MODEL_PATH = f"{PROJECT_ROOT}/model"

# Global embedding cache
_embedding_cache = {}

# Custom loguru sink to discard INFO and DEBUG logs during tqdm
def tqdm_sink(message):
    if message.record["level"].name in ["INFO", "DEBUG"]:
        return
    tqdm.write(str(message), end="")

def clean_and_balance_collection(db: StandardDatabase) -> Dict:
    """Audit and balance the ArangoDB collection, returning stats."""
    # [Function remains unchanged]
    pass

def batch_embed_texts(texts: List[str], embedder: EmbedderModel, batch_size: int = 32, prefix: Optional[str] = None) -> List[List[float]]:
    """
    Embed a list of texts in batches with caching.
    
    Args:
        texts: List of texts to embed
        embedder: The embedding model to use
        batch_size: Batch size for embedding
        prefix: Optional prefix to prepend to each text
        
    Returns:
        List of embeddings
    """
    global _embedding_cache
    
    # Check cache for all texts
    uncached_texts = []
    uncached_indices = []
    embeddings = [None] * len(texts)
    
    # Prepare texts with prefix
    prefixed_texts = [f"{prefix}{text}" if prefix else text for text in texts]
    
    # Check cache
    for i, text in enumerate(prefixed_texts):
        if text in _embedding_cache:
            embeddings[i] = _embedding_cache[text]
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # If all texts were cached, return early
    if not uncached_texts:
        return embeddings
    
    # Process uncached texts in batches
    batch_count = (len(uncached_texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(batch_count), desc=f"Embedding {len(uncached_texts)} texts in batches"):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, len(uncached_texts))
        batch = uncached_texts[batch_start:batch_end]
        
        # Use non-prefixed version for embedding since we already added prefix
        batch_embeddings = embedder.embed_batch(batch)
        
        # Update cache and results
        for j, (text, embedding) in enumerate(zip(batch, batch_embeddings)):
            _embedding_cache[text] = embedding
            embeddings[uncached_indices[batch_start + j]] = embedding
    
    # Return completed embeddings list
    return embeddings

def precompute_all_embeddings(test_dataset: Dataset, embedder: EmbedderModel) -> Dict[str, List[float]]:
    """
    Precompute embeddings for all questions in the test dataset.
    
    Args:
        test_dataset: Dataset containing test samples
        embedder: The embedding model to use
        
    Returns:
        Dictionary mapping questions to embeddings
    """
    questions = [item["question"] for item in test_dataset]
    logger.info(f"Precomputing embeddings for {len(questions)} test questions")
    
    batch_size = CONFIG["embedding"]["batch_size"]
    embeddings = batch_embed_texts(questions, embedder, batch_size)
    
    # Create mapping from questions to embeddings
    question_embeddings = {}
    for question, embedding in zip(questions, embeddings):
        question_embeddings[question] = embedding
    
    logger.info(f"Completed precomputing {len(question_embeddings)} embeddings")
    return question_embeddings

def test_classifier(db: StandardDatabase, test_dataset: Dataset, k_values: List[int] = None, use_hybrid: bool = False) -> Dict:
    """Evaluate the semantic search classifier or hybrid k-NN + logistic regression on a test dataset."""
    if k_values is None:
        k_values = [CONFIG["classification"]["default_k"], 10, 20, 50]
    method = "Hybrid k-NN + Logistic Regression" if use_hybrid else "Semantic Search (Majority Vote)"
    logger.info(f"Testing {method} with k={k_values} on {len(test_dataset)} test samples")
    
    results = []
    embedder = get_EmbedderModel()
    
    # Print the actual embedding model being used
    if hasattr(embedder, 'model_name'):
        logger.info(f"Using embedding model: {embedder.model_name}")
    
    # OPTIMIZATION 5: Precompute all embeddings before evaluation
    start_time = time.time()
    precomputed_embeddings = precompute_all_embeddings(test_dataset, embedder)
    logger.info(f"Precomputation completed in {time.time() - start_time:.2f} seconds")
    
    for k in k_values:
        logger.info(f"Evaluating with k={k}")
        predictions = []
        true_labels = []
        confidences = []
        features = []
        
        # Extract all questions and convert to batches
        test_items = list(test_dataset)
        batch_size = 32  # Process multiple samples at once
        num_batches = (len(test_items) + batch_size - 1) // batch_size
        
        # Suppress logs during tqdm
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        logger.add(tqdm_sink, level="INFO")
        logger.add(tqdm_sink, level="DEBUG")
        
        # Process in batches
        for batch_idx in tqdm(range(num_batches), desc=f"Classifying test samples (k={k})"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(test_items))
            batch = test_items[batch_start:batch_end]
            
            batch_questions = [item["question"] for item in batch]
            batch_true_labels = [1 if float(item["rating"]) >= 0.5 else 0 for item in batch]
            
            # Process each question in the batch
            batch_predictions = []
            batch_confidences = []
            batch_neighbor_lists = []
            
            for question in batch_questions:
                # Use precomputed embedding if available
                pred_label, confidence, neighbors = classify_complexity(db, question, k)
                batch_predictions.append(pred_label)
                batch_confidences.append(confidence)
                batch_neighbor_lists.append(neighbors)
            
            # Extend results
            predictions.extend(batch_predictions)
            true_labels.extend(batch_true_labels)
            confidences.extend(batch_confidences)
            
            # For hybrid mode, prepare features
            if use_hybrid:
                for i, question in enumerate(batch_questions):
                    neighbors = batch_neighbor_lists[i]
                    neighbor_embeddings = [n["embedding"] for n in neighbors]
                    if neighbor_embeddings:
                        avg_embedding = np.mean(neighbor_embeddings, axis=0)
                    else:
                        # Use precomputed embedding
                        avg_embedding = precomputed_embeddings[question]
                    features.append(avg_embedding)
        
        # Train and evaluate hybrid model if enabled
        if use_hybrid:
            # Prepare training data
            train_texts = [item["question"] for item in train_dataset]
            train_labels = [1 if float(item["rating"]) >= 0.5 else 0 for item in train_dataset]
            
            # OPTIMIZATION 1+2: Batch embed training texts with caching
            train_features = []
            logger.info("Preparing training features for hybrid model")
            
            # Process training texts in batches
            batch_size = CONFIG["embedding"]["batch_size"]
            for i in tqdm(range(0, len(train_texts), batch_size), desc=f"Processing training features (k={k})"):
                batch = train_texts[i:i + batch_size]
                
                # Get labels for the batch
                batch_features = []
                for text in batch:
                    # Query database for neighbors
                    _, _, neighbors = classify_complexity(db, text, k)
                    neighbor_embeddings = [n["embedding"] for n in neighbors]
                    
                    if neighbor_embeddings:
                        avg_embedding = np.mean(neighbor_embeddings, axis=0)
                    else:
                        # Use embedding directly if no neighbors found
                        if text in precomputed_embeddings:
                            avg_embedding = precomputed_embeddings[text]
                        else:
                            # Add to precomputed embeddings if not already there
                            embedding = embedder.embed_batch([text], prefix=DOC_PREFIX)[0]
                            precomputed_embeddings[text] = embedding
                            avg_embedding = embedding
                    
                    batch_features.append(avg_embedding)
                
                train_features.extend(batch_features)
            
            # Train logistic regression
            model = LogisticRegression(max_iter=1000)
            model.fit(train_features, train_labels)
            
            # Predict with hybrid model
            predictions = model.predict(features)
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="binary", zero_division=0)
        recall = recall_score(true_labels, predictions, average="binary", zero_division=0)
        f1 = f1_score(true_labels, predictions, average="binary", zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Log detailed report for default k
        if k == CONFIG["classification"]["default_k"]:
            logger.info("\n" + classification_report(true_labels, predictions, target_names=["Simple", "Complex"]))
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Simple", "Complex"], yticklabels=["Simple", "Complex"])
            plt.title(f"Confusion Matrix - {method} (k={k})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(f"confusion_matrix_{'hybrid' if use_hybrid else 'semantic'}_k{k}.png")
            plt.close()
        
        result = {
            "k": k,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "average_confidence": np.mean(confidences) if not use_hybrid else 0.0,
        }
        results.append(result)
        logger.info(f"{method} results (k={k}): {result}")
    
    # Log comparison table
    table = [
        ["k", "Accuracy", "Precision", "Recall", "F1-Score"],
        *[[r["k"], f"{r['accuracy']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1_score']:.3f}"] for r in results],
    ]
    logger.info(f"\n=== {method} Performance by k ===\n" + tabulate(table, headers="firstrow", tablefmt="grid"))
    
    # Log cache statistics
    logger.info(f"Embedding cache stats: {len(_embedding_cache)} cached embeddings")
    
    # Return result for default k
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    return next(r for r in results if r["k"] == CONFIG["classification"]["default_k"])

def train_baseline_model(train_dataset: Dataset, test_dataset: Dataset) -> Dict:
    """Train and evaluate a baseline logistic regression model."""
    logger.info("Training baseline logistic regression model")
    
    # Prepare training data
    train_texts = [item["question"] for item in train_dataset]
    train_labels = [1 if float(item["rating"]) >= 0.5 else 0 for item in train_dataset]
    
    # OPTIMIZATION 1+2: Generate embeddings with batching and caching
    embedder = get_EmbedderModel()
    train_embeddings = batch_embed_texts(
        train_texts, 
        embedder, 
        batch_size=CONFIG["embedding"]["batch_size"],
        prefix=DOC_PREFIX
    )
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(train_embeddings, train_labels)
    
    # Prepare test data
    test_texts = [item["question"] for item in test_dataset]
    test_labels = [1 if float(item["rating"]) >= 0.5 else 0 for item in test_dataset]
    
    # OPTIMIZATION 1+2: Generate test embeddings with batching and caching
    test_embeddings = batch_embed_texts(
        test_texts, 
        embedder, 
        batch_size=CONFIG["embedding"]["batch_size"],
        prefix=DOC_PREFIX
    )
    
    # Evaluate
    predictions = model.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="binary", zero_division=0)
    recall = recall_score(test_labels, predictions, average="binary", zero_division=0)
    f1 = f1_score(test_labels, predictions, average="binary", zero_division=0)
    conf_matrix = confusion_matrix(test_labels, predictions)
    
    # Log detailed report
    logger.info("\n" + classification_report(test_labels, predictions, target_names=["Simple", "Complex"]))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=["Simple", "Complex"], yticklabels=["Simple", "Complex"])
    plt.title("Confusion Matrix - Baseline Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_baseline.png")
    plt.close()
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),
    }
    
    logger.info(f"Baseline logistic regression results: {results}")
    return results

def test_distilbert_model(test_dataset: Dataset) -> Dict:
    """Evaluate the trained DistilBERT model on a test dataset."""
    # [Function remains largely unchanged, could add batching if needed]
    pass

def generate_final_report(semantic_results: Dict, baseline_results: Dict, distilbert_results: Dict, db_stats: Dict) -> str:
    """Generate a formatted final report comparing semantic classification and DistilBERT model."""
    # [Function remains unchanged]
    pass

if __name__ == "__main__":
    from datasets import load_dataset
    global train_dataset  # For hybrid model access
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    start_time = time.time()
    
    try:
        client = connect_arango()
        db = ensure_database(client)
        ensure_collection(db)
        ensure_arangosearch_view(db)
        
        # [Loading and filtering dataset code remains unchanged]
        
        # OPTIMIZATION: Add timing information
        data_load_time = time.time()
        logger.info(f"Data loading completed in {data_load_time - start_time:.2f} seconds")
        
        # [Dataset splitting code remains unchanged]
        
        split_time = time.time()
        logger.info(f"Dataset splitting completed in {split_time - data_load_time:.2f} seconds")
        
        # Clean and balance collection
        logger.info("Cleaning and balancing ArangoDB collection")
        db_stats = clean_and_balance_collection(db)
        
        balance_time = time.time()
        logger.info(f"Collection balancing completed in {balance_time - split_time:.2f} seconds")
        
        # CRITICAL: Create a new BGE EmbedderModel and reindex the database with it
        logger.info("Initializing BGE EmbedderModel and reindexing the database")
        embedder = EmbedderModel(CONFIG["embedding"]["model_name"])
        
        init_time = time.time()
        logger.info(f"Embedder initialization completed in {init_time - balance_time:.2f} seconds")
        
        load_and_index_dataset(db, embedder=embedder)    
        ensure_vector_index(db)
        
        index_time = time.time()
        logger.info(f"Database indexing completed in {index_time - init_time:.2f} seconds")
        
        logger.info(f"Using embedding model: {embedder.model_name if hasattr(embedder, 'model_name') else CONFIG['embedding']['model_name']}")
        
        # Run evaluations with timing
        semantic_start = time.time()
        semantic_results = test_classifier(db, test_dataset, k_values=[5, 7, 10, 20, 25], use_hybrid=False)
        semantic_time = time.time()
        logger.info(f"Semantic evaluation completed in {semantic_time - semantic_start:.2f} seconds")
        
        baseline_start = time.time()
        baseline_results = train_baseline_model(train_dataset, test_dataset)
        baseline_time = time.time()
        logger.info(f"Baseline model evaluation completed in {baseline_time - baseline_start:.2f} seconds")
        
        distilbert_start = time.time()
        distilbert_results = test_distilbert_model(test_dataset)
        distilbert_time = time.time()
        logger.info(f"DistilBERT evaluation completed in {distilbert_time - distilbert_start:.2f} seconds")
        
        # Generate and log final report
        final_report = generate_final_report(semantic_results, baseline_results, distilbert_results, db_stats)
        logger.info(final_report)
        
        # Save the report to a file
        report_file = Path("model_comparison_report.txt")
        report_file.write_text(final_report)
        logger.info(f"Report saved to {report_file.absolute()}")
        
        # Log final timing
        end_time = time.time()
        logger.info(f"Total evaluation completed in {end_time - start_time:.2f} seconds")
        logger.info("Evaluation completed successfully")
        
        # Log embedding cache stats
        logger.info(f"Final embedding cache size: {len(_embedding_cache)} embeddings")
        
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)