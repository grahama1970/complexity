# src/complexity/beta/rag/rag_classifier.py
"""
Module Description:
Provides the EmbedderModel class for generating text embeddings using transformer models.
This class handles loading models (e.g., BGE, sentence-transformers) and generating
normalized embeddings suitable for semantic search or classification tasks.

Links:
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
- PyTorch: https://pytorch.org/docs/stable/index.html
- Loguru: https://loguru.readthedocs.io/en/stable/

Sample Input (for EmbedderModel().embed_single):
  text = "This is a sample sentence."

Expected Output (structure):
  List[float] of length corresponding to the model's embedding dimension (e.g., 768).
  Example (conceptual): [0.123, -0.456, ..., 0.789]
"""

import torch
import sys
from typing import List, Optional
from loguru import logger
from transformers import AutoTokenizer, AutoModel

# Import the model name from config.py
from complexity.beta.utils.config import CONFIG

# Use model name from CONFIG
DOC_PREFIX = "search_document: "

class EmbedderModel:
    """A class for generating text embeddings using a transformer model."""
    
    def __init__(self, model_name: Optional[str] = None): # Changed str to Optional[str]
        """
        Initialize the EmbedderModel with a specified model.
        
        Args:
            model_name (str): Name of the transformer model to use.
                             If None, uses the model from CONFIG.
        """
        # If no model specified, use from config
        if model_name is None:
            model_name = CONFIG["embedding"]["model_name"]
            
        logger.info(f"Loading embedding model: {model_name}")
        try:
            # Store the model name as an attribute for reference
            self.model_name = model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Model loaded on device: {self.device}")
        except Exception as e:
            logger.exception(f"Failed to load model {model_name}: {e}")
            raise
    
    def embed_batch(self, texts: List[str], prefix: Optional[str] = None) -> List[List[float]]:
        """
        Embed a batch of texts.
        
        Args:
            texts (List[str]): List of texts to embed.
            prefix (Optional[str]): Prefix to prepend to each text (e.g., for search context).
        
        Returns:
            List[List[float]]: List of embeddings for the input texts.
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []
        
        try:
            # Apply prefix if provided
            if prefix:
                texts = [f"{prefix}{text}" for text in texts]
            
            # Generate embeddings
            logger.debug(f"Embedding {len(texts)} texts on {self.device}")
            
            # For BGE models, use specific handling
            if self.model_name is None:
                 # This case should ideally not happen due to __init__ logic
                 logger.error("self.model_name is unexpectedly None in embed_batch")
                 # Raise an error or return early if model_name is None
                 raise ValueError("Model name is not set before embedding")

            # Now self.model_name is confirmed not None by the check above
            if "bge" in self.model_name.lower():
                # Tokenize inputs for the BGE model
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Get model outputs
                    outputs = self.model(**inputs)
                    
                    # For BGE models, use the CLS token embedding (first token)
                    embeddings = outputs.last_hidden_state[:, 0]
                    
                    # Normalize embeddings to unit length (important for BGE models)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    # Convert to list of lists for compatibility
                    embeddings_list = embeddings.cpu().tolist()
            else:
                # Default embedding logic for other models
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Use mean pooling for other models
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    
                    # Sum token embeddings and divide by attention mask
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = input_mask_expanded.sum(1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    
                    # Normalize
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    embeddings_list = embeddings.cpu().tolist()
            
            logger.debug(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list
        
        except Exception as e:
            logger.exception(f"Embedding failed: {e}")
            raise
    
    def embed_single(self, text: str, prefix: Optional[str] = None) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text (str): Text to embed.
            prefix (Optional[str]): Prefix to prepend to the text.
        
        Returns:
            List[float]: Embedding for the input text.
        """
        embeddings = self.embed_batch([text], prefix=prefix)
        return embeddings[0] if embeddings else []


if __name__ == "__main__":
    """Minimal real-world usage and validation."""
    logger.info("Starting standalone execution and validation...")

    # --- Configuration ---
    # Use the default model from config for testing
    # Ensure CONFIG is loaded correctly
    try:
        model_name_from_config = CONFIG["embedding"]["model_name"]
        # A simple known dimension for a common model type, adjust if needed
        # Ideally, fetch this dynamically or store it reliably.
        # For 'BAAI/bge-small-en-v1.5', dimension is 384
        # For 'nomic-ai/nomic-embed-text-v1', dimension is 768
        # Let's assume a default or fetch dynamically if possible
        # Fallback dimension if model name doesn't match known ones
        expected_dimension = 768 # Renamed to lowercase
        if "bge-small" in model_name_from_config:
             expected_dimension = 384 # Renamed to lowercase
        elif "nomic-embed-text" in model_name_from_config:
             expected_dimension = 768 # Renamed to lowercase
        # Add more known model dimensions here if necessary

        logger.info(f"Using model: {model_name_from_config}")
        logger.info(f"Expecting dimension: {expected_dimension}")

    except KeyError as e:
        logger.error(f"❌ VALIDATION FAILED: Missing key in CONFIG: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ VALIDATION FAILED: Error loading config: {e}")
        sys.exit(1)

    sample_text = "This is a test sentence for embedding validation."
    validation_passed = True
    validation_failures = {}

    # --- Initialization ---
    try:
        logger.info("Initializing EmbedderModel...")
        embedder = EmbedderModel() # Uses model from CONFIG by default
        logger.info("EmbedderModel initialized successfully.")
    except Exception as e:
        logger.error(f"❌ VALIDATION FAILED: Could not initialize EmbedderModel: {e}")
        validation_passed = False
        validation_failures["initialization"] = f"Failed with error: {e}"
        # Cannot proceed if initialization fails
        print("❌ VALIDATION FAILED - EmbedderModel initialization error.")
        print(f"FAILURE DETAILS:\n  - initialization: {validation_failures['initialization']}")
        sys.exit(1)


    # --- Embedding Generation ---
    embedding = None
    try:
        logger.info(f"Generating embedding for: '{sample_text}'")
        embedding = embedder.embed_single(sample_text)
        logger.info("Embedding generated successfully.")
    except Exception as e:
        logger.error(f"❌ VALIDATION FAILED: embed_single failed: {e}")
        validation_passed = False
        validation_failures["embedding_generation"] = f"embed_single failed with error: {e}"


    # --- Validation ---
    if validation_passed and embedding is not None:
        logger.info("Validating embedding...")
        # 1. Check if embedding is empty (Removed redundant isinstance check)
        if not embedding:
            validation_passed = False
            validation_failures["emptiness"] = {
                "expected": "non-empty list",
                "actual": "empty list"
            }
            logger.error("Validation Error: Embedding is empty.")

        # 2. Check embedding dimension
        elif len(embedding) != expected_dimension: # Use lowercase variable
            validation_passed = False
            validation_failures["dimension"] = {
                "expected": expected_dimension, # Use lowercase variable
                "actual": len(embedding)
            }
            logger.error(f"Validation Error: Dimension mismatch. Expected {expected_dimension}, Got {len(embedding)}") # Use lowercase variable

        # 3. Check if elements are floats (check first few)
        else:
            all_floats = all(isinstance(x, float) for x in embedding[:5]) # Check first 5 elements
            if not all_floats:
                 validation_passed = False
                 validation_failures["element_type"] = {
                     "expected": "float",
                     "actual": "non-float found in first 5 elements"
                 }
                 logger.error("Validation Error: Non-float element found in embedding.")
            else:
                 logger.info("Embedding structure and dimension validated.")

    # --- Reporting ---
    if validation_passed:
        print("✅ VALIDATION COMPLETE - EmbedderModel generated valid embedding.")
        logger.success("Standalone execution and validation successful.")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Issues detected during EmbedderModel validation.")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            if isinstance(details, dict):
                 print(f"  - {field}: Expected: {details.get('expected', 'N/A')}, Got: {details.get('actual', 'N/A')}")
            else:
                 print(f"  - {field}: {details}")
        logger.error("Standalone execution and validation failed.")
        sys.exit(1)