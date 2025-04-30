# src/complexity/arangodb/embedding_utils.py
import os
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Union
import time
from loguru import logger

# Import HuggingFace for BAAI/bge model
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    has_transformers = True
    logger.info("Transformers library is available for embeddings")
except ImportError:
    has_transformers = False
    logger.warning("Transformers library not available, will use fallback embedding method")

# Import config
try:
    from complexity.arangodb.config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
except ImportError:
    logger.warning("Failed to import config, using default embedding settings")
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSIONS = 1024

# Initialize BAAI/bge model if available
_model = None
_tokenizer = None

def _initialize_model():
    """Initialize the BAAI/bge embedding model and tokenizer."""
    global _model, _tokenizer
    
    if not has_transformers:
        logger.warning("Transformers library not available, cannot initialize model")
        return False
    
    try:
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        _model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            _model = _model.to("cuda")
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        return False

def get_embedding(text: str, model: str = None) -> Optional[List[float]]:
    """
    Get an embedding vector for a text string using BAAI/bge model.
    
    Args:
        text: The text to embed
        model: Optional model name (defaults to config value)
        
    Returns:
        List of embedding values or None if embedding failed
    """
    global _model, _tokenizer
    
    # Use default model if not specified
    model = model or EMBEDDING_MODEL
    
    logger.info(f"Generating embedding for text: {text[:50]}...")
    
    # We only use BGE model - no OpenAI
    if has_transformers:
        # Initialize model if not done yet
        if _model is None or _tokenizer is None:
            success = _initialize_model()
            if not success:
                return _fallback_embedding(text)
        
        try:
            # Prepare inputs
            encoded_input = _tokenizer(text, padding=True, truncation=True, 
                                      return_tensors='pt', max_length=512)
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                encoded_input = {k: v.to("cuda") for k, v in encoded_input.items()}
            
            # Compute embedding
            with torch.no_grad():
                model_output = _model(**encoded_input)
                embedding = model_output.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embedding with {EMBEDDING_MODEL}: {e}")
            # Fall back to hash-based method
            return _fallback_embedding(text)
    
    # Use fallback method if transformers not available
    return _fallback_embedding(text)

def _fallback_embedding(text: str) -> List[float]:
    """
    Generate a deterministic fallback embedding using text hash.
    This is only used when the primary embedding method fails.
    
    Args:
        text: The text to embed
        
    Returns:
        List of embedding values
    """
    logger.warning("Using fallback embedding method (hash-based)")
    
    # Use text hash as seed
    text_hash = hashlib.md5(text.encode()).hexdigest()
    seed = int(text_hash, 16) % (2**32)
    
    # Generate deterministic random vector
    np.random.seed(seed)
    embedding = np.random.normal(0, 1, EMBEDDING_DIMENSIONS)
    
    # Normalize the vector
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.tolist()

def get_EmbedderModel():
    """
    Return the embedding model information.
    
    Returns:
        Dict with model name and dimensions
    """
    return {
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS
    }