from transformers import DistilBertTokenizerFast
import evaluate
import numpy as np
import logging
from loguru import logger
import torch
import os
import transformers
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
    """Enhanced metrics calculation for classification"""
    metric = evaluate.combine([
        evaluate.load("accuracy"),
        evaluate.load("precision"),
        evaluate.load("recall"),
        evaluate.load("f1")
    ])
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted"  # Use weighted averaging for imbalanced classes
    )


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


def log_system_metrics(datasets, dataset_name):
    """Log system configuration and dataset metrics"""
    # System hardware metrics
    logger.info("\nSystem Configuration:")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Total memory: {torch.cuda.get_device_properties(i).total_memory/1e9:.2f}GB")
    logger.info(f"Training device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info(f"CPU cores: {os.cpu_count()}")

    # Dataset metrics
    logger.info("\nDataset Configuration:")
    logger.info(f"HuggingFace Dataset: https://huggingface.co/datasets/{dataset_name}")
    for split in ["train", "val", "test"]:
        logger.info(f"{split.capitalize()} samples: {len(datasets[split])}")

    # Memory metrics
    try:
        import psutil
        ram = psutil.virtual_memory()
        logger.info(f"\nSystem RAM: {ram.total/1e9:.2f} GB total, {ram.available/1e9:.2f} GB available")
    except ImportError:
        logger.warning("psutil not installed - RAM information unavailable")

    # Precision support
    logger.info(f"Mixed precision support: {torch.cuda.amp.is_available() if torch.cuda.is_available() else 'N/A'}")


class TrainingSummaryCallback(transformers.TrainerCallback):
    """Logs final training statistics and metrics"""
    def on_train_end(self, args, state, control, **kwargs):
        logger = logging.getLogger(__name__)
        if state.is_world_process_zero:
            # Calculate total training time
            training_time = state.log_history[-1]["train_runtime"]
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = int(training_time % 60)
            
            # Get final metrics
            final_metrics = state.log_history[-1]
            
            logger.info("\n🚀 Training Complete - Final Metrics:")
            logger.info(f"⏱️  Total training time: {hours}h {minutes}m {seconds}s")
            logger.info(f"📊 Samples/second: {final_metrics.get('train_samples_per_second', 'N/A'):.1f}")
            logger.info(f"🔢 Total steps: {state.num_train_epochs:.0f} epochs")
            logger.info(f"🎯 Final validation loss: {final_metrics.get('eval_loss', 'N/A'):.4f}")
            logger.info(f"🏆 Best validation accuracy: {state.best_metric:.4f}")
            logger.info(f"📈 Total samples processed: {state.num_training_samples * state.num_train_epochs:.0f}")
            
            # Log all metrics
            for key, value in final_metrics.items():
                if key not in ["epoch", "train_runtime", "train_samples_per_second"]:
                    logger.info(f"{key}: {value:.4f}")


if __name__ == "__main__":
    hello_world()
