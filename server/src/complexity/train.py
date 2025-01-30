import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np
from loguru import logger
import os
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter

from complexity.utils import (
    preprocess_function, 
    binarize_labels, 
    compute_metrics,
    determine_training_params
)

# Load environment variables
load_dotenv("../.env")

# Configurations
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./model"
HF_TOKEN = os.getenv("HF_TOKEN")  # Load the token from .env

# Check if high-speed download is enabled
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "False").lower() == "true":
    logger.info("High-speed HuggingFace downloads enabled")
else:
    logger.info("Using standard HuggingFace downloads")


def load_and_prepare_data():
    """Correct dataset splitting implementation"""
    try:
        logger.info("Loading dataset...")
        # Verify dataset structure first
        dataset = load_dataset(
            "wesley7137/question_complexity_classification",
            token=HF_TOKEN
        )

        # Confirm we have a DatasetDict with train split
        if "train" not in dataset:
            raise ValueError("Dataset missing required 'train' split")

        # Get actual Dataset object
        full_dataset = dataset["train"]

        # First split: 80% train+val, 20% test
        train_val_test = full_dataset.train_test_split(
            test_size=0.2,
            seed=42
        )

        # Second split: 10% of original = 12.5% of remaining 80%
        train_val = train_val_test["train"].train_test_split(
            test_size=0.125,  # 0.1 / 0.8 = 0.125
            seed=42
        )

        final_splits = {
            "train": train_val["train"],
            "val": train_val["test"],
            "test": train_val_test["test"]
        }

        # Clean each split separately
        logger.info("Cleaning dataset...")
        def is_valid(example):
            rating = example.get("rating")
            return (
                rating is not None and 
                isinstance(rating, (int, float)) and
                0 <= rating <= 1
            )

        cleaned_datasets = {
            split_name: dataset.filter(is_valid)
            for split_name, dataset in final_splits.items()
        }

        # Log cleaning results
        for split in ["train", "val", "test"]:
            orig = len(final_splits[split])
            cleaned = len(cleaned_datasets[split])
            logger.info(f"{split}: {cleaned}/{orig} examples retained")

        # Then preprocess
        logger.info("Tokenizing dataset...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        tokenized_datasets = {
            split_name: dataset.map(
                preprocess_function(tokenizer),  # Creates 'labels' from 'rating'
                batched=True,
                remove_columns=["question", "rating"]  # Only remove these original columns
            )
            for split_name, dataset in cleaned_datasets.items()
        }

        return tokenizer, tokenized_datasets
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


def train_model(num_epochs, early_stopping_patience):
    """Train the DistilBERT model for question complexity classification."""
    try:
        tokenizer, datasets = load_and_prepare_data()

        # Determine dataset size dynamically
        dataset_samples = len(datasets["train"])

        # Initialize TensorBoard
        writer = SummaryWriter(log_dir=os.path.abspath("./runs/complexity"))

        logger.info("Initializing model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )

        # Add after model initialization
        logger.info(f"Classifier weights initialized: {model.classifier.weight[0, :5]}")
        logger.info(f"Pre-classifier bias: {model.pre_classifier.bias[:5]}")

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=num_epochs,  # Dynamically determined
            weight_decay=0.01,
            push_to_hub=False,
            logging_dir="./logs",
            load_best_model_at_end=True,  # Always load the best performing model
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            logging_steps=10,
            save_total_limit=2,
            warmup_ratio=0.1,  # For cosine annealing
            report_to="tensorboard",
            lr_scheduler_type="cosine_with_restarts",  # Helps handle late-phase improvements
            save_on_each_node=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],  # Use validation set for evaluation
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            ],  # Dynamically determined early stopping
        )

        logger.info("Starting training...")
        trainer.train()

        # Log metrics to TensorBoard
        for epoch, metrics in enumerate(trainer.state.log_history):
            if "eval_loss" in metrics:
                writer.add_scalar("Loss/val", metrics["eval_loss"], epoch)
            if "eval_accuracy" in metrics:
                writer.add_scalar("Accuracy/val", metrics["eval_accuracy"], epoch)
            if "train_loss" in metrics:
                writer.add_scalar("Loss/train", metrics["train_loss"], epoch)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(datasets["test"])
        logger.info(f"Test set results: {test_results}")

        writer.close()

        logger.info(f"Saving model to {OUTPUT_DIR}...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        logger.success("Training complete!")
    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    # Example: DistilBERT (66M params), batch size 16
    model_params = 66_000_000
    batch_size = 16

    # Load tokenizer and dataset to get actual training set size
    tokenizer, datasets = load_and_prepare_data()
    dataset_samples = len(datasets["train"])

    num_epochs, early_stopping_patience = determine_training_params(
        model_params, dataset_samples, batch_size
    )

    logger.info(
        f"Determined training settings: {num_epochs} epochs, {early_stopping_patience} patience"
    )

    train_model(num_epochs, early_stopping_patience)
