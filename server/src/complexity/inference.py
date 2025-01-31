from complexity.file_utils import get_project_root, load_env_file
import torch
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification
)
from loguru import logger
import os

PROJECT_ROOT = get_project_root()
load_env_file('.env')

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

        return "Complex" if predicted_class == 1 else "Simple"
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return "Error"


if __name__ == "__main__":
    tokenizer, model = load_model()
    question = "Explain the theory of relativity step by step."
    complexity = classify_question(question, tokenizer, model)
    logger.info(f"Question Complexity: {complexity}")
