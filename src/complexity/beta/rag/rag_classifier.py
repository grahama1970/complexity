import torch
from typing import List, Optional
from loguru import logger
from transformers import AutoTokenizer, AutoModel

# Import the model name from config.py
from complexity.beta.utils.config import CONFIG

# Use model name from CONFIG
DOC_PREFIX = "search_document: "

class EmbedderModel:
    """A class for generating text embeddings using a transformer model."""
    
    def __init__(self, model_name: str = None):
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