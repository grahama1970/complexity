"""
Utilities for generating and comparing text embeddings using various models.

Links to Documentation:
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Sentence Transformers: https://www.sbert.net/docs/package_reference/SentenceTransformer.html

Sample Input/Output:
    Input: "This is a test document"
    Output: List[float] - Vector of dimension 768 (local) or 1536 (OpenAI)
"""

import sys
import os
import json
import time
import math
from typing import List, Optional, Any, Dict, Union, Tuple
from loguru import logger

# Default configuration
EMBEDDING_MODEL = "text-embedding-ada-002"  # Default OpenAI model
EMBEDDING_DIMENSIONS = 1536  # Ada-002 dimension

# Initialize availability flags
openai_available = False
sentence_transformers_available = False
openai = None  # Define at module level to avoid unbound reference
SentenceTransformer = None  # Define at module level to avoid unbound reference

# Import OpenAI if available, otherwise provide a degraded experience
try:
    import openai
    openai_available = True
except ImportError:
    logger.warning("OpenAI package not available - using fallback embedding method")

# Import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    logger.warning("sentence-transformers package not available")

# Try to load API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (0-1)
    """
    # Check if vectors have the same length
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have the same length: {len(vec1)} != {len(vec2)}")

    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    # Check for zero magnitudes
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    # Ensure the value is between 0 and 1
    return max(0.0, min(1.0, similarity))

def get_local_embedding(text: str, model_name: str = "nomic-ai/modernbert-embed-base") -> List[float]:
    """
    Get embedding vector for text using local sentence-transformers model.

    Args:
        text: Text to embed
        model_name: Name of the sentence-transformers model to use

    Returns:
        List of float values representing the embedding vector
    """
    # Check if text is empty
    if not text:
        logger.error("Empty text provided for embedding")
        # Return a zero vector of default dimension as fallback
        return [0.0] * 768  # Default BERT dimension

    # Try to use sentence-transformers if available
    if sentence_transformers_available and SentenceTransformer is not None:
        try:
            # Load model
            model = SentenceTransformer(model_name)
            
            # Generate embedding
            logger.info(f"Generating local embedding for text: {text[:50]}...")
            embedding = model.encode([text], convert_to_tensor=False)[0]
            
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Sentence-transformers error: {e}")
            # Fall through to fallback method

    # Fallback to deterministic hash-based embedding
    logger.warning("Using fallback embedding method (hash-based)")
    return get_embedding(text, model="fallback")

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Get embedding vector for a text string using OpenAI API.

    Args:
        text: Text to embed
        model: Embedding model to use

    Returns:
        List of float values representing the embedding vector
    """
    # Check if text is empty
    if not text:
        logger.error("Empty text provided for embedding")
        # Return a zero vector of the right dimension as fallback
        return [0.0] * EMBEDDING_DIMENSIONS

    # Try to use OpenAI if available
    if openai_available and OPENAI_API_KEY and model != "fallback" and openai is not None:
        try:
            # Instantiate the OpenAI client with the API key
            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            # Call OpenAI API using the new client interface
            logger.info(f"Generating embedding for text: {text[:50]}...")

            response = client.embeddings.create(model=model, input=text)

            # Extract the embedding from the response using the new structure
            embedding = response.data[0].embedding

            return embedding
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")

    # Fallback to deterministic hash-based embedding
    logger.warning("Using fallback embedding method (hash-based)")

    import hashlib
    import struct
    import numpy as np

    # Create a hash of the text
    hasher = hashlib.sha512()
    hasher.update(text.encode("utf-8"))
    hash_bytes = hasher.digest()

    # Convert the hash to a list of floats
    # Each 8 bytes becomes a double precision float
    step = 8  # 8 bytes per double
    floats = []

    # Generate enough bytes for the required dimensions
    bytes_needed = EMBEDDING_DIMENSIONS * step
    current_bytes = hash_bytes

    while len(current_bytes) < bytes_needed:
        hasher.update(current_bytes)
        current_bytes += hasher.digest()

    # Convert bytes to floats
    for i in range(0, EMBEDDING_DIMENSIONS * step, step):
        if i + step <= len(current_bytes):
            value = struct.unpack("d", current_bytes[i : i + step])[0]
            # Normalize to a reasonable range (-1 to 1)
            value = np.tanh(value)
            floats.append(value)

    # Normalize the vector to unit length
    norm = math.sqrt(sum(f * f for f in floats))
    if norm > 0:
        floats = [f / norm for f in floats]

    return floats[:EMBEDDING_DIMENSIONS]

def validate_embeddings(test_embeddings: List[Dict[str, Any]], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate embedding generation against expected results.

    Args:
        test_embeddings: List of test embeddings to validate
        fixture_path: Path to the fixture file containing expected results

    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    # Load fixture data if it exists
    try:
        with open(fixture_path, "r") as f:
            expected_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Fixture file not found: {fixture_path}")
        return False, {"fixture_error": {"expected": "Fixture file exists", "actual": "File not found"}}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode fixture JSON: {e}")
        return False, {"fixture_error": {"expected": "Valid JSON", "actual": f"JSON decode error: {e}"}}
    except Exception as e:
        logger.error(f"Failed to load fixture data: {e}")
        return False, {"fixture_error": {"expected": "Fixture loaded successfully", "actual": str(e)}}

    # Track all validation failures
    validation_failures = {}

    # Check dimensions
    expected_dimensions = expected_data.get("embedding_dimensions", EMBEDDING_DIMENSIONS)

    for item in test_embeddings:
        if "text" in item and "embedding" in item:
            text = item["text"]
            embedding = item["embedding"]

            # Check dimensions
            if len(embedding) != expected_dimensions:
                validation_failures[f"dimensions_{text}"] = {
                    "expected": expected_dimensions,
                    "actual": len(embedding),
                }

            # Check if text exists in expected data
            if text in expected_data.get("test_values", {}):
                expected_hash = expected_data["test_values"][text].get("hash")

                if expected_hash:
                    # Calculate hash of the current embedding
                    import hashlib
                    embedding_str = json.dumps(embedding)
                    hash_object = hashlib.md5(embedding_str.encode())
                    actual_hash = hash_object.hexdigest()

                    # If using OpenAI, hashes won't match exactly, so this is informational
                    if actual_hash != expected_hash and not openai_available:
                        validation_failures[f"embedding_hash_{text}"] = {
                            "expected": expected_hash,
                            "actual": actual_hash,
                        }

    # If we're using OpenAI, we don't expect the hashes to match exactly
    # So only consider the validation failures that are about dimensions
    if openai_available:
        filtered_failures = {k: v for k, v in validation_failures.items() if "dimensions" in k}
        return len(filtered_failures) == 0, filtered_failures

    return len(validation_failures) == 0, validation_failures

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Define expected validation results
    EXPECTED_RESULTS = {
        "similarity_tests": {
            "self_similarity": 1.0,
            "min_cross_similarity": 0.0,
            "max_cross_similarity": 1.0
        },
        "embedding_dimensions": {
            "local": 768,  # BERT base dimension
            "openai": 1536  # Ada-002 dimension
        },
        "validation_criteria": {
            "self_similarity_tolerance": 0.0001,
            "dimension_match": True,
            "value_range": [-1.0, 1.0]
        }
    }

    try:
        # Generate test embeddings using both methods
        test_texts = [
            "This is a test document about artificial intelligence.",
            "Python programming is fun and versatile.",
            "Data science combines statistics and programming.",
        ]

        all_validation_passed = True
        validation_errors = []

        # Test both embedding methods
        for method in ["local", "openai"]:
            test_embeddings = []
            embedding_func = get_local_embedding if method == "local" else get_embedding
            expected_dim = EXPECTED_RESULTS["embedding_dimensions"][method]

            for text in test_texts:
                embedding = embedding_func(text)
                test_embeddings.append({"text": text, "embedding": embedding})

                # Validate embedding properties
                if len(embedding) != expected_dim:
                    all_validation_passed = False
                    validation_errors.append(f"{method} embedding dimension mismatch: {len(embedding)} != {expected_dim}")

                # Check value range
                min_val = min(embedding)
                max_val = max(embedding)
                if min_val < EXPECTED_RESULTS["validation_criteria"]["value_range"][0] or \
                   max_val > EXPECTED_RESULTS["validation_criteria"]["value_range"][1]:
                    all_validation_passed = False
                    validation_errors.append(f"{method} embedding values out of range: [{min_val}, {max_val}]")

                # Test self-similarity
                self_similarity = cosine_similarity(embedding, embedding)
                if abs(self_similarity - EXPECTED_RESULTS["similarity_tests"]["self_similarity"]) > \
                   EXPECTED_RESULTS["validation_criteria"]["self_similarity_tolerance"]:
                    all_validation_passed = False
                    validation_errors.append(
                        f"{method} self-similarity test failed: {self_similarity} != "
                        f"{EXPECTED_RESULTS['similarity_tests']['self_similarity']}"
                    )

                logger.info(f"Generated {method} embedding for '{text[:20]}...' "
                          f"with {len(embedding)} dimensions")
                logger.info(f"Value range: {min_val:.4f} to {max_val:.4f}")
                logger.info(f"Self-similarity: {self_similarity:.4f}")

            # Test cross-similarities
            for i in range(len(test_embeddings)):
                for j in range(i + 1, len(test_embeddings)):
                    text1 = test_embeddings[i]["text"]
                    text2 = test_embeddings[j]["text"]
                    embed1 = test_embeddings[i]["embedding"]
                    embed2 = test_embeddings[j]["embedding"]

                    similarity = cosine_similarity(embed1, embed2)
                    logger.info(
                        f"{method} similarity between '{text1[:10]}...' and '{text2[:10]}...': {similarity:.4f}"
                    )

                    # Validate similarity is in valid range
                    if similarity < EXPECTED_RESULTS["similarity_tests"]["min_cross_similarity"] or \
                       similarity > EXPECTED_RESULTS["similarity_tests"]["max_cross_similarity"]:
                        all_validation_passed = False
                        validation_errors.append(
                            f"{method} cross-similarity out of range: {similarity}"
                        )

        # Report final validation status
        if all_validation_passed:
            print("✅ VALIDATION COMPLETE - All embedding functions work as expected")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Embedding functions don't match expected behavior")
            print("Validation Errors:")
            for error in validation_errors:
                print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Unexpected error in __main__: {e}")
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        sys.exit(1)
