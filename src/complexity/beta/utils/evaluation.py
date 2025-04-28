import sys
from typing import Dict, List
from loguru import logger
import sys
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
from complexity.beta.utils.config import CONFIG  # Import CONFIG directly for model info
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

# Load environment variables
PROJECT_ROOT = get_project_root()
load_env_file()
MODEL_PATH = f"{PROJECT_ROOT}/model"

# Custom loguru sink to discard INFO and DEBUG logs during tqdm
def tqdm_sink(message):
    if message.record["level"].name in ["INFO", "DEBUG"]:
        return
    tqdm.write(str(message), end="")

def clean_and_balance_collection(db: StandardDatabase) -> Dict:
    """Audit and balance the ArangoDB collection, returning stats."""
    logger.info("Auditing and balancing ArangoDB collection")
    col = db.collection(CONFIG["search"]["collection_name"])
    docs = list(col.find({}))
    
    # Audit labels
    valid_docs = []
    invalid_count = 0
    for doc in docs:
        if "label" not in doc or "question" not in doc or doc["question"] is None:
            logger.warning(f"Invalid document: {doc}")
            invalid_count += 1
            continue
        if doc["label"] not in [0, 1]:
            logger.warning(f"Invalid label in document: {doc}")
            invalid_count += 1
            continue
        valid_docs.append(doc)
    
    # Compute class distribution
    labels = [doc["label"] for doc in valid_docs]
    simple_count = labels.count(0)
    complex_count = labels.count(1)
    logger.info(f"Valid documents: {len(valid_docs)}, Simple: {simple_count}, Complex: {complex_count}, Invalid: {invalid_count}")
    
    # Balance classes (downsample majority class)
    target_count = min(simple_count, complex_count)
    simple_docs = [doc for doc in valid_docs if doc["label"] == 0][:target_count]
    complex_docs = [doc for doc in valid_docs if doc["label"] == 1][:target_count]
    balanced_docs = simple_docs + complex_docs
    
    # Update collection
    col.truncate()
    if balanced_docs:
        col.insert_many(balanced_docs)
        logger.info(f"Balanced collection: {len(balanced_docs)} documents (Simple: {len(simple_docs)}, Complex: {len(complex_docs)})")
    
    return {
        "original_count": len(docs),
        "valid_count": len(valid_docs),
        "invalid_count": invalid_count,
        "balanced_count": len(balanced_docs),
        "simple_count": len(simple_docs),
        "complex_count": len(complex_docs),
    }

def test_classifier(db: StandardDatabase, test_dataset: Dataset, k_values: List[int] = None, use_hybrid: bool = False) -> Dict:
    """Evaluate the semantic search classifier or hybrid k-NN + logistic regression on a test dataset."""
    if k_values is None:
        k_values = [CONFIG["classification"]["default_k"], 10, 20, 50]
    method = "Hybrid k-NN + Logistic Regression" if use_hybrid else "Semantic Search (Majority Vote)"
    logger.info(f"Testing {method} with k={k_values} on {len(test_dataset)} test samples")
    
    results = []
    EmbedderModel = get_EmbedderModel()
    
    # Print the actual embedding model being used
    if hasattr(EmbedderModel, 'model_name'):
        logger.info(f"Using embedding model: {EmbedderModel.model_name}")
    else:
        logger.info(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
    
    for k in k_values:
        logger.info(f"Evaluating with k={k}")
        predictions = []
        true_labels = []
        confidences = []
        features = []
        
        # Suppress logs during tqdm
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        logger.add(tqdm_sink, level="INFO")
        logger.add(tqdm_sink, level="DEBUG")
        with tqdm(test_dataset, desc=f"Classifying test samples (k={k})") as pbar:
            for item in pbar:
                question = item.get("question")
                true_label = 1 if float(item.get("rating")) >= 0.5 else 0
                pred_label, confidence, neighbors = classify_complexity(db, question, k)
                predictions.append(pred_label)
                true_labels.append(true_label)
                confidences.append(confidence)
                if use_hybrid:
                    neighbor_embeddings = [n["embedding"] for n in neighbors]
                    avg_embedding = np.mean(neighbor_embeddings, axis=0) if neighbor_embeddings else EmbedderModel.embed_batch([question], prefix=DOC_PREFIX)[0]
                    features.append(avg_embedding)
        
        # Train and evaluate hybrid model if enabled
        if use_hybrid:
            # Prepare training data
            train_texts = [item["question"] for item in train_dataset]
            train_labels = [1 if float(item["rating"]) >= 0.5 else 0 for item in train_dataset]
            train_features = []
            for text in tqdm(train_texts, desc=f"Extracting k-NN features (k={k})"):
                _, _, neighbors = classify_complexity(db, text, k)
                neighbor_embeddings = [n["embedding"] for n in neighbors]
                avg_embedding = np.mean(neighbor_embeddings, axis=0) if neighbor_embeddings else EmbedderModel.embed_batch([text], prefix=DOC_PREFIX)[0]
                train_features.append(avg_embedding)
            
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
    
    # Generate embeddings with tqdm
    EmbedderModel = get_EmbedderModel()
    train_embeddings = []
    for i in tqdm(range(0, len(train_texts), CONFIG["embedding"]["batch_size"]), desc="Embedding train data"):
        batch = train_texts[i:i + CONFIG["embedding"]["batch_size"]]
        batch_embs = EmbedderModel.embed_batch(batch, prefix=DOC_PREFIX)
        train_embeddings.extend(batch_embs)
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(train_embeddings, train_labels)
    
    # Prepare test data
    test_texts = [item["question"] for item in test_dataset]
    test_labels = [1 if float(item["rating"]) >= 0.5 else 0 for item in test_dataset]
    test_embeddings = []
    for i in tqdm(range(0, len(test_texts), CONFIG["embedding"]["batch_size"]), desc="Embedding test data"):
        batch = test_texts[i:i + CONFIG["embedding"]["batch_size"]]
        batch_embs = EmbedderModel.embed_batch(batch, prefix=DOC_PREFIX)
        test_embeddings.extend(batch_embs)
    
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
    logger.info(f"Testing DistilBERT model on {len(test_dataset)} test samples")
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading DistilBERT model from: {MODEL_PATH}")
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logger.info(f"DistilBERT model loaded on device: {device}")
        
        predictions = []
        true_labels = []
        
        # Suppress logs during tqdm
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        logger.add(tqdm_sink, level="INFO")
        logger.add(tqdm_sink, level="DEBUG")
        with tqdm(test_dataset, desc="Classifying test samples with DistilBERT") as pbar:
            for item in pbar:
                question = item.get("question")
                true_label = 1 if float(item.get("rating")) >= 0.5 else 0
                
                # Tokenize and classify
                inputs = tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred_label = torch.argmax(logits, dim=-1).item()
                
                predictions.append(pred_label)
                true_labels.append(true_label)
        
        # Restore logging
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="binary", zero_division=0)
        recall = recall_score(true_labels, predictions, average="binary", zero_division=0)
        f1 = f1_score(true_labels, predictions, average="binary", zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Log detailed report
        logger.info("\n" + classification_report(true_labels, predictions, target_names=["Simple", "Complex"]))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Purples", xticklabels=["Simple", "Complex"], yticklabels=["Simple", "Complex"])
        plt.title("Confusion Matrix - DistilBERT Model")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("confusion_matrix_distilbert.png")
        plt.close()
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix.tolist(),
        }
        
        logger.info(f"DistilBERT test results: {results}")
        return results
    
    except Exception as e:
        logger.exception(f"DistilBERT evaluation failed: {e}")
        raise

def generate_final_report(semantic_results: Dict, baseline_results: Dict, distilbert_results: Dict, db_stats: Dict) -> str:
    """Generate a formatted final report comparing semantic classification and DistilBERT model."""
    # Get information about embedding model 
    EmbedderModel = get_EmbedderModel()
    embedding_model = getattr(EmbedderModel, 'model_name', EMBEDDING_MODEL_NAME)
    
    # Model comparison table
    table = [
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
        [
            f"Semantic Classification ({embedding_model}, k={semantic_results['k']})",
            f"{semantic_results['accuracy']:.3f}",
            f"{semantic_results['precision']:.3f}",
            f"{semantic_results['recall']:.3f}",
            f"{semantic_results['f1_score']:.3f}",
        ],
        [
            "DistilBERT (Trained Model)",
            f"{distilbert_results['accuracy']:.3f}",
            f"{distilbert_results['precision']:.3f}",
            f"{distilbert_results['recall']:.3f}",
            f"{distilbert_results['f1_score']:.3f}",
        ],
        [
            "Logistic Regression (Baseline)",
            f"{baseline_results['accuracy']:.3f}",
            f"{baseline_results['precision']:.3f}",
            f"{baseline_results['recall']:.3f}",
            f"{baseline_results['f1_score']:.3f}",
        ],
    ]
    report = "\n=== Model Performance Comparison ===\n"
    report += tabulate(table, headers="firstrow", tablefmt="grid")
    
    # Effectiveness summary
    report += "\n=== Effectiveness Analysis ===\n"
    if distilbert_results["accuracy"] > semantic_results["accuracy"]:
        report += (
            f"The DistilBERT trained model is more effective than semantic classification with {embedding_model} (k={semantic_results['k']}), "
            f"achieving higher accuracy ({distilbert_results['accuracy']:.3f} vs. {semantic_results['accuracy']:.3f}). "
            f"DistilBERT also shows superior precision ({distilbert_results['precision']:.3f} vs. {semantic_results['precision']:.3f}), "
            f"but semantic classification has higher recall ({semantic_results['recall']:.3f} vs. {distilbert_results['recall']:.3f})."
        )
    elif distilbert_results["accuracy"] < semantic_results["accuracy"]:
        report += (
            f"Semantic classification with {embedding_model} (k={semantic_results['k']}) is more effective than the DistilBERT trained model, "
            f"achieving higher accuracy ({semantic_results['accuracy']:.3f} vs. {distilbert_results['accuracy']:.3f}). "
            f"Semantic classification also excels in recall ({semantic_results['recall']:.3f} vs. {distilbert_results['recall']:.3f}), "
            f"while DistilBERT may have better precision ({distilbert_results['precision']:.3f} vs. {semantic_results['precision']:.3f})."
        )
    else:
        report += (
            f"Semantic classification with {embedding_model} (k={semantic_results['k']}) and the DistilBERT trained model are equally effective in accuracy "
            f"({semantic_results['accuracy']:.3f}). However, differences in precision ({semantic_results['precision']:.3f} vs. "
            f"{distilbert_results['precision']:.3f}) and recall ({semantic_results['recall']:.3f} vs. {distilbert_results['recall']:.3f}) "
            f"may influence their suitability depending on the use case."
        )
    
    # Database stats
    report += "\n=== Database Audit Summary ===\n"
    report += tabulate(
        [
            ["Original Documents", db_stats["original_count"]],
            ["Valid Documents", db_stats["valid_count"]],
            ["Invalid Documents", db_stats["invalid_count"]],
            ["Balanced Documents", db_stats["balanced_count"]],
            ["Simple Documents", db_stats["simple_count"]],
            ["Complex Documents", db_stats["complex_count"]],
        ],
        headers=["Metric", "Value"],
        tablefmt="grid",
    )
    
    return report

if __name__ == "__main__":
    from datasets import load_dataset
    global train_dataset  # For hybrid model access
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    try:
        client = connect_arango()
        db = ensure_database(client)
        ensure_collection(db)
        ensure_arangosearch_view(db)
        
        # Load dataset and filter invalid entries
        logger.info("Loading dataset")
        dataset = load_dataset(
            CONFIG["dataset"]["name"], split=CONFIG["dataset"]["split"], trust_remote_code=True
        )
        logger.info(f"Original dataset size: {len(dataset)}")
        
        # Filter entries with valid ratings
        valid_data = []
        skipped = 0
        for item in dataset:
            rating = item.get("rating")
            question = item.get("question")
            if rating is None or question is None:
                logger.warning(f"Skipping document with invalid rating or question: {item}")
                skipped += 1
                continue
            try:
                float(rating)  # Ensure rating can be converted to float
                valid_data.append(item)
            except (ValueError, TypeError):
                logger.warning(f"Skipping document with non-numeric rating: {item}")
                skipped += 1
                continue
        
        if not valid_data:
            logger.error("No valid documents found in dataset")
            raise ValueError("Dataset contains no valid documents")
        
        logger.info(f"Filtered dataset size: {len(valid_data)} ({skipped} documents skipped)")
        
        # Convert to list for train_test_split
        data_list = list(valid_data)
        stratify_labels = [1 if float(item["rating"]) >= 0.5 else 0 for item in data_list]
        
        # Split dataset
        train_data, test_data = train_test_split(
            data_list,
            test_size=0.2,
            random_state=42,
            stratify=stratify_labels
        )
        
        # Convert back to Dataset
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
        
        # Clean and balance collection
        logger.info("Cleaning and balancing ArangoDB collection")
        db_stats = clean_and_balance_collection(db)
        
        # CRITICAL: Create a new BGE EmbedderModel and reindex the database with it
        logger.info("Initializing BGE EmbedderModel and reindexing the database")
        EmbedderModel = EmbedderModel(CONFIG["embedding"]["model_name"])
        load_and_index_dataset(db, EmbedderModel=EmbedderModel)    
        ensure_vector_index(db)
        
        logger.info(f"Using embedding model: {EmbedderModel.model_name if hasattr(EmbedderModel, 'model_name') else EMBEDDING_MODEL_NAME}")
        
        # Run evaluations
        semantic_results = test_classifier(db, test_dataset, k_values=[5, 7, 10, 20, 25], use_hybrid=False)
        baseline_results = train_baseline_model(train_dataset, test_dataset)
        distilbert_results = test_distilbert_model(test_dataset)
        
        # Generate and log final report
        final_report = generate_final_report(semantic_results, baseline_results, distilbert_results, db_stats)
        logger.info(final_report)
        
        # Save the report to a file
        report_file = Path("model_comparison_report.txt")
        report_file.write_text(final_report)
        logger.info(f"Report saved to {report_file.absolute()}")
        
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)