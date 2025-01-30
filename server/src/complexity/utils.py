from transformers import DistilBertTokenizerFast
import evaluate
import numpy as np
import logging

logger = logging.getLogger(__name__)

def hello_world():
    print("hello world")


def preprocess_function(tokenizer):
    """Return a function that tokenizes text and creates labels from ratings"""
    def preprocess(examples):
        tokenized = tokenizer(
            examples["question"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        # Create labels from ratings
        tokenized["labels"] = [1 if r > 0.5 else 0 for r in examples["rating"]]
        return tokenized
    return preprocess


def binarize_labels(example):
    """Simplified conversion now that data is pre-cleaned"""
    example["label"] = 1 if example["rating"] > 0.5 else 0
    return example


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Compute accuracy for model evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


def determine_training_params(
    model_size: int, dataset_size: int, batch_size: int
) -> tuple:
    """
    Dynamically determine the number of training epochs and early stopping patience.

    Args:
        model_size (int): Number of parameters in the model.
        dataset_size (int): Number of training samples.
        batch_size (int): Training batch size.

    Returns:
        tuple: (num_epochs, early_stopping_patience)
    """

    # Base settings
    base_epochs = 10  # Minimum number of epochs
    base_patience = 5  # Minimum patience

    # Adjust for model size (larger models need more epochs)
    if model_size < 100_000_000:  # Small models (DistilBERT, TinyBERT)
        extra_epochs = 5
        extra_patience = 5
    elif model_size < 300_000_000:  # Medium models (BERT-Base, RoBERTa-Base)
        extra_epochs = 10
        extra_patience = 10
    else:  # Large models (BERT-Large, T5-Large)
        extra_epochs = 20
        extra_patience = 15

    # Adjust for dataset size (larger datasets need more patience)
    if dataset_size < 50_000:  # Small dataset
        dataset_factor = 0.8
    elif dataset_size < 500_000:  # Medium dataset
        dataset_factor = 1.0
    else:  # Large dataset
        dataset_factor = 1.5

    # Adjust for batch size (small batches need more training)
    batch_factor = 1.0 if batch_size >= 32 else 1.2

    # Final calculation
    num_epochs = int((base_epochs + extra_epochs) * dataset_factor * batch_factor)
    early_stopping_patience = int(
        (base_patience + extra_patience) * dataset_factor * batch_factor
    )

    return num_epochs, early_stopping_patience


if __name__ == "__main__":
    hello_world()
