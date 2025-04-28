import torch
from typing import List, Optional
from loguru import logger
from sentence_transformers import SentenceTransformer

# Constants
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
DOC_PREFIX = "search_document: "

class ModernBertEmbedder:
    """A class for generating text embeddings using a transformer model."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the embedder with a specified model.
        
        Args:
            model_name (str): Name of the transformer model to use.
        """
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
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
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.device
            )
            
            # Convert to list of lists for compatibility
            embeddings_list = embeddings.tolist()
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
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    try:
        # Test the embedder
        embedder = ModernBertEmbedder()
        test_texts = [
            "What is the capital of France?",
            "How does quantum computing work?"
        ]
        embeddings = embedder.embed_batch(test_texts, prefix=DOC_PREFIX)
        logger.info(f"Embedded {len(test_texts)} texts. First embedding length: {len(embeddings[0])}")
        single_embedding = embedder.embed_single(test_texts[0], prefix=DOC_PREFIX)
        logger.info(f"Single embedding length: {len(single_embedding)}")
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        sys.exit(1)
