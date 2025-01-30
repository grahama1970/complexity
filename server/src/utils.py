from transformers import DistilBertTokenizerFast
import evaluate
import numpy as np


def preprocess_function(tokenizer):
    """Return a function that tokenizes the text for use in datasets.map()"""

    def preprocess(examples):
        return tokenizer(
            examples["question"], truncation=True, padding="max_length", max_length=128
        )

    return preprocess


def binarize_labels(example):
    """Convert continuous complexity scores into binary labels."""
    example["label"] = 1 if example["complexity"] > 0.5 else 0
    return example


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Compute accuracy for model evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)
