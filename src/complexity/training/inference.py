from complexity.utils.file_utils import get_project_root, load_env_file
import torch
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification
)
from loguru import logger
import os
import time  # Add this at the top with other imports

PROJECT_ROOT = get_project_root()
load_env_file()

MODEL_PATH = f"{PROJECT_ROOT}/model"


def load_model():
    """Load the trained DistilBERT model and tokenizer."""
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        logger.info(f"Model files present: {os.listdir(MODEL_PATH)}")
        
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        
        # Verify model weights
        logger.info(f"Model classifier weights: {model.classifier.weight[0,:5]}")
        
        model.eval()  # Set model to inference mode
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def classify_question(question, tokenizer, model):
    """Classify a question as simple (0) or complex (1)."""
    try:
        start_time = time.perf_counter()  # Start timer
        
        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        elapsed = time.perf_counter() - start_time  # Calculate duration
        logger.debug(f"Classification took {elapsed*1000:.2f}ms")
        
        return "Complex" if predicted_class == 1 else "Simple"
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return "Error"


if __name__ == "__main__":
    tokenizer, model = load_model()
    
    
    questions = [
        "What is the most common color of an apple?",
        "Explain the process of nuclear fission in detail.",
        "What is the half-life of uranium-238?",
        "How does a nuclear reactor generate electricity?",
        "What is the capital of France?",
        "Give me a list of all the planets in the solar system.",      
    ]
    
    # Time full pipeline
    for question in questions:
        start = time.perf_counter()
        complexity = classify_question(question, tokenizer, model)
        total_time = time.perf_counter() - start
        from tabulate import tabulate
        # Initialize table with headers if first question
        if not hasattr(classify_question, 'table'):
            classify_question.table = [["Question", "Classification", "Time (ms)"]]
        # Add row for current question
        classify_question.table.append([
            question, 
            complexity, 
            f"{total_time*1000:.2f}"
        ])
        # Print full table after last question
        if question == questions[-1]:
            logger.info("\n" + tabulate(classify_question.table, tablefmt="grid", headers="firstrow"))