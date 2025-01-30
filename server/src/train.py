import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np
from loguru import logger
import os
from utils import preprocess_function, binarize_labels, compute_metrics
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env')

# Configurations
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./model"
HF_TOKEN = os.getenv("HF_TOKEN")  # Load the token from .env

# Check if high-speed download is enabled
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "False").lower() == "true":
    logger.info("High-speed Hugging Face downloads enabled")
else:
    logger.info("Using standard Hugging Face downloads")

def load_and_prepare_data():
    """Load and preprocess the dataset."""
    try:
        logger.info("Loading dataset...")
        dataset = load_dataset(
            "wesley7137/question_complexity_classification",
            use_auth_token=HF_TOKEN  # Pass the token here
        )

        logger.info("Tokenizing dataset...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        tokenized_datasets = dataset.map(preprocess_function(tokenizer), batched=True)
        tokenized_datasets = tokenized_datasets.map(binarize_labels)

        return tokenizer, tokenized_datasets
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def train_model():
    """Train the DistilBERT model for question complexity classification."""
    try:
        tokenizer, tokenized_datasets = load_and_prepare_data()
        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]

        logger.info("Initializing model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            push_to_hub=False,
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info(f"Saving model to {OUTPUT_DIR}...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        logger.success("Training complete!")
    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    train_model()
