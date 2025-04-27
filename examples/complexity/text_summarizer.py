"""
Text Summarizer Module

Purpose:
    Provides efficient text summarization using LLM with configurable chunk sizes,
    overlap, and validation against expected outputs. Handles long text via MapReduce.
"""

import sys
import json
import asyncio # Ensure asyncio is imported
from typing import Dict, Any, Optional, List, Tuple, Union
from difflib import SequenceMatcher
from pathlib import Path

from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from typing import Union, Optional, Dict, Any
from pydantic import BaseModel, Field
import litellm # Import the base library
from litellm import completion
from litellm.types.utils import ModelResponse as LiteLLMResponse
class ModelMessage(BaseModel):
    """Message from an LLM response"""
    role: str = Field(default="assistant")
    content: str = Field(default="")

class ModelChoice(BaseModel):
    """Single choice from an LLM response"""
    message: ModelMessage
    index: int = Field(default=0)
    finish_reason: str = Field(default="stop")

class CustomStreamWrapper(BaseModel):
    """Wrapper for streamed responses"""
    choices: list[ModelChoice] = Field(default_factory=list)
    id: str = Field(default="")

class ModelResponse(BaseModel):
    """Response from an LLM call"""
    id: str = Field(default="")
    choices: list[ModelChoice] = Field(default_factory=list)

    @classmethod
    def from_stream_wrapper(cls, wrapper: CustomStreamWrapper) -> "ModelResponse":
        """Convert a stream wrapper to a ModelResponse"""
        return cls(
            id=wrapper.id,
            choices=wrapper.choices
        )
import nltk
import os
import torch
from sentence_transformers import SentenceTransformer, util
try:
    import openai
    openai_available = True # Renamed to lowercase
except ImportError:
    logger.warning("OpenAI library not found. OpenAI embedding provider will not be available.")
    openai_available = False # Renamed to lowercase
    openai = None # Define openai as None if import fails

# Try to load API key from environment for OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Check for key presence here as well
openai_available = openai_available and bool(OPENAI_API_KEY)
if not openai_available:
     # Log reason if either library or key is missing
     if not openai:
          logger.warning("OpenAI library not found. OpenAI embedding provider disabled.")
     elif not OPENAI_API_KEY:
          logger.warning("OPENAI_API_KEY environment variable not set. OpenAI embedding provider disabled.")

# Use absolute imports starting from src
from src.complexity.utils.json_utils import clean_json_string
from src.complexity.utils.estimate_token_count import estimate_token_count
from src.complexity.utils.summarizer.utils.load_wikipedia_text import load_wikipedia_text # Import wikipedia loader
# Import multimodal utils (assuming it will be enhanced later)
# from summarizer.utils.multimodal_utils import format_multimodal_messages, is_multimodal # File not found
# Import log utils for truncation
# from summarizer.utils.log_utils import truncate_large_value # File not found

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# --- Embedding Model Management ---
local_embedding_model = None # For SentenceTransformer instance

def get_local_embedding_model(model_name: str):
    """Loads the local SentenceTransformer model if not already loaded."""
    global local_embedding_model
    # Check if the correct model is already loaded
    if local_embedding_model is None or getattr(local_embedding_model, '_model_name', None) != model_name:
        logger.info(f"Loading local embedding model: {model_name}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            local_embedding_model = SentenceTransformer(model_name, device=device)
            setattr(local_embedding_model, '_model_name', model_name) # Store model name for checking
            logger.info(f"Local embedding model loaded successfully on {device}.")
        except Exception as e:
            logger.warning(f"Failed to load local model on {device}: {e}. Trying CPU...")
            local_embedding_model = SentenceTransformer(model_name, device='cpu')
            setattr(local_embedding_model, '_model_name', model_name)
            logger.info("Local embedding model loaded successfully on CPU.")
    return local_embedding_model

def get_openai_embedding(text: str, model_name: str) -> List[float]:
    """Gets a single embedding from OpenAI."""
    # Use the boolean flag which already incorporates the API key check
    if not openai_available or not openai:
        raise RuntimeError("OpenAI provider selected, but library or API key is not available.")
    # No need for separate OPENAI_API_KEY check here anymore
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(model=model_name, input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        # Re-raise the exception to be handled by the caller
        raise RuntimeError(f"OpenAI API call failed: {e}")

async def generate_embeddings(texts: List[str], embedding_config: Dict[str, Any]) -> List[List[float]]:
    """Generates embeddings using the configured provider."""
    provider = embedding_config.get("provider", "local") # Default to local
    embeddings = []

    if provider == "local":
        model_name = embedding_config.get("local_model", "nomic-ai/modernbert-embed-base")
        model = get_local_embedding_model(model_name)
        # Add prefix for modernbert if it's the selected local model
        if "modernbert" in model_name:
             texts_to_embed = [f"search_document: {text}" for text in texts]
             logger.debug("Added 'search_document:' prefix for ModernBERT.")
        else:
             texts_to_embed = texts
        # Run encode in a separate thread as it can be CPU/GPU intensive
        embeddings = await asyncio.to_thread(model.encode, texts_to_embed)
        embeddings = embeddings.tolist() # Convert numpy array to list
    elif provider == "openai":
        if not openai_available: # Use renamed variable
             raise ValueError("OpenAI provider selected, but library or API key is unavailable.")
        model_name = embedding_config.get("openai_model", "text-embedding-ada-002")
        logger.info(f"Using OpenAI embedding model: {model_name}")
        # Use asyncio.gather for potential concurrency with OpenAI API calls
        tasks = [asyncio.to_thread(get_openai_embedding, text, model_name) for text in texts]
        try:
            results = await asyncio.gather(*tasks)
            embeddings = list(results)
        except RuntimeError as e:
             logger.error(f"Failed to get OpenAI embeddings: {e}")
             raise ValueError(f"Failed to generate OpenAI embeddings: {e}")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    # Basic validation
    if not embeddings or len(embeddings) != len(texts):
         raise ValueError("Failed to generate embeddings for all input texts.")

    return embeddings

# --- LLM Call Handling ---
def should_retry_call(exception: BaseException) -> bool:
    """Determine if an exception should trigger a retry based on status code."""
    # Check for both attributes since litellm exceptions might be structured differently
    status_code = getattr(exception, "status_code", None) or getattr(exception, "code", None)
    if not status_code:
        logger.debug(f"No status code found in exception: {type(exception)}")
        return False
    # Retry on rate limits and server errors
    should_retry = status_code in [429] or (500 <= status_code < 600)
    if should_retry:
        logger.info(f"Will retry due to status code {status_code}")
    return should_retry

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(should_retry_call)
)
def reliable_completion(**kwargs: Any) -> LiteLLMResponse:
    """Make LLM completion call with robust retry handling. Uses LiteLLM's ModelResponse type."""
    messages = kwargs.get("messages", [])
    # Check if any message content is a list (indicating multimodal)
    is_multimodal_call = any(isinstance(msg.get("content"), list) for msg in messages)

    # Always use LiteLLM
    
    # For multimodal calls, ensure user content is a list. System content remains string.
    if is_multimodal_call and "messages" in kwargs:
        # We only need to ensure the user message content is a list,
        # which should already be handled by summarize_text/summarize_chunk.
        # System message content should remain a string.
        # Let's log the structure being sent.
        truncated_messages = truncate_large_value(kwargs["messages"])
        logger.debug(f"Making LiteLLM API call (multimodal) with messages: {truncated_messages}")
        log_kwargs = {k: v for k, v in kwargs.items() if k != 'messages'}
        logger.debug(f"Other LiteLLM API call kwargs (multimodal): {log_kwargs}")
    elif "messages" in kwargs: # Standard text call
        truncated_messages = truncate_large_value(kwargs["messages"])
        logger.debug(f"Making LiteLLM API call (text) with messages: {truncated_messages}")
        log_kwargs = {k: v for k, v in kwargs.items() if k != 'messages'}
        logger.debug(f"Other LiteLLM API call kwargs (text): {log_kwargs}")

    # Log the final kwargs structure just before the call
    log_kwargs_final = {k: truncate_large_value(v) if k == 'messages' else v for k, v in kwargs.items()}
    logger.debug(f"Final kwargs for litellm.completion: {log_kwargs_final}")

# Explicitly log the messages structure being passed
    logger.debug(f"Passing messages to litellm.completion: {truncate_large_value(kwargs.get('messages', []))}")
    response = completion(**kwargs, timeout=60)

    # Safely extract content using duck typing instead of strict type checking
    choices = getattr(response, "choices", [])
    if not choices:
        logger.error("Empty response - no choices returned")
        raise ValueError("Empty model response - no choices returned")

    # Handle both regular and streaming choices
    choice = choices[0]
    content = None

    # Try different ways to access content based on response structure
    if hasattr(choice, "message") and hasattr(choice.message, "content"):
        content = choice.message.content
    elif hasattr(choice, "delta") and hasattr(choice.delta, "content"):
        content = choice.delta.content
    else:
        logger.error(f"Unexpected response structure: {choice}")
        raise ValueError("Could not extract content from response")

    if not content or not content.strip():
        raise ValueError("Empty message content in response")

    content = content.strip()
    logger.debug(f"Got valid response with content preview: {content[:100]}...")
    
    # Extract id from response or generate new one
    response_id = getattr(response, 'id', '') or str(uuid.uuid4())
    
    # Create response using our model classes
    message = ModelMessage(role="assistant", content=content)
    choice = ModelChoice(message=message, finish_reason="stop")
    llm_response = LiteLLMResponse(id=response_id, choices=[choice])
    
    logger.debug("Created LiteLLM response object")
    return llm_response


    return response

# --- Validation Logic ---
async def validate_summary(summary: str, validation_data: Dict[str, Any], embedding_config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    """
    Validate summary against expected results or original text.
    Returns: (passed, failures_dict, metrics_dict)
    If 'expected_summary' is same as 'input_text', assumes validation against original.
    """
    validation_failures = {}
    metrics = {} # Dictionary to store calculated metrics like similarity
    summary_lower = summary.lower()
    original_text = validation_data["input_text"]
    is_validation_against_original = validation_data.get("expected_summary") == original_text

    # --- Semantic Content Validation ---
    logger.info("Performing semantic similarity validation...")
    target_text_for_similarity = validation_data["expected_summary"] # Could be original text or a specific expected summary
    try:
        embeddings = await generate_embeddings([summary, target_text_for_similarity], embedding_config)
        if len(embeddings) != 2:
             raise ValueError(f"Expected 2 embeddings, but got {len(embeddings)}")
        # Calculate similarity using sentence_transformers util
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item() # Get scalar value
        metrics["semantic_similarity"] = similarity # Store the calculated similarity
    except (ValueError, RuntimeError) as e:
         logger.error(f"Embedding generation failed during validation: {e}")
         validation_failures["embedding_generation"] = {"error": str(e)}
         # Cannot proceed with similarity check if embeddings failed
         # Return empty metrics dict on embedding failure
         return False, validation_failures, {}

    # Determine threshold based on validation type
    default_threshold = 0.6 if is_validation_against_original else 0.7
    threshold = validation_data["expected_properties"].get("semantic_similarity_threshold", default_threshold)
    logger.debug(f"Semantic similarity score: {similarity:.4f} (threshold: {threshold})")

    if similarity < threshold:
        validation_failures["semantic_similarity"] = {
            "expected": f">= {threshold}",
            "actual": f"{similarity:.4f}"
        }

    # --- Other Validations (Only if NOT validating against original text) ---
    if not is_validation_against_original:
        logger.info("Performing compression ratio and key concept validation...")
        # Compression ratio validation
        input_len = len(original_text.split())
        summary_len = len(summary.split())
        if input_len > 0:
            compression = summary_len / input_len
            expected_compression = validation_data["expected_properties"].get("compression_ratio")
            if expected_compression is not None:
                tolerance = 0.4 # Allow 40% deviation
                if abs(compression - expected_compression) > tolerance:
                    validation_failures["compression_ratio"] = {
                        "expected": expected_compression,
                        "actual": f"{compression:.2f}"
                    }
        else:
             logger.warning("Input text has zero length, skipping compression ratio check.")


        # Key concepts validation
        expected_concepts = validation_data["expected_properties"].get("key_concepts", [])
        if expected_concepts:
            missing_concepts = []
            for concept in expected_concepts:
                if concept.lower() not in summary_lower:
                    missing_concepts.append(concept)
            if missing_concepts:
                validation_failures["missing_concepts"] = {
                    "expected": expected_concepts,
                    "actual_missing": missing_concepts,
                    "found": [c for c in expected_concepts if c not in missing_concepts]
                }
    else:
         logger.info("Skipping compression ratio and key concept validation (validating against original text).")


    logger.debug(f"Validation failures: {validation_failures}")
    logger.debug(f"Validation metrics: {metrics}")
    return len(validation_failures) == 0, validation_failures, metrics

# --- Chunking Logic ---
def create_chunks_with_overlap(sentences: List[str], chunk_size: int, overlap_size: Optional[int]=None) -> List[List[str]]:
    """Creates chunks of sentences with overlap, handling long sentences effectively."""
    if not sentences:
        return []
    if overlap_size is None:
        overlap_size = max(1, len(sentences) // 20) # Default overlap based on sentence count
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt', quiet=True)

    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        # Handle sentence longer than chunk size
        if sentence_tokens > chunk_size:
            if current_chunk: # Add previous chunk if exists
                chunks.append(current_chunk)
            chunks.append([sentence]) # Long sentence becomes its own chunk
            current_chunk = []
            current_chunk_tokens = 0
            logger.warning(f"Sentence with {sentence_tokens} tokens exceeds chunk size {chunk_size}. Treating as separate chunk.")
        # If adding sentence exceeds chunk size, finalize current chunk
        elif current_chunk_tokens + sentence_tokens > chunk_size:
            if current_chunk: # Ensure chunk is not empty before adding
                 chunks.append(current_chunk)
            # Start new chunk with overlap
            start_index = max(0, len(current_chunk) - overlap_size) if overlap_size > 0 else len(current_chunk)
            current_chunk = current_chunk[start_index:]
            current_chunk_tokens = sum(estimate_token_count(s) for s in current_chunk)
            # Add current sentence to new chunk
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens
        # Otherwise, add sentence to current chunk
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

# --- Summarization Core Logic ---

async def summarize_chunk(
    chunk_text: str,
    config: Dict[str, Any],
    prompt: str,
    image_inputs: Optional[List[Dict[str, Any]]] = None, # Added optional images for final reduction
    code_metadata: Optional[str] = None # Added code metadata for final reduction context
) -> str:
     """Summarizes a single text chunk using reliable_completion.
     Can include images and code metadata for final reduction.
     Returns only the content string from the response."""
     model = config.get("model", "gpt-4o-mini")
     temperature = config.get("temperature", 0.7)
     # Estimate input tokens for max_tokens calculation
     input_tokens = estimate_token_count(chunk_text)
     # Allow slightly more tokens for chunk summaries, maybe 80%? Cap at 1000?
     max_tokens = min(max(150, int(input_tokens * 0.8)), 1000)

     logger.debug(f"Summarizing chunk ({input_tokens} tokens) with max_output={max_tokens}...")
     # Use reliable_completion (run sync function in thread for async context)
     try:
         response = await asyncio.to_thread(
             reliable_completion,
             model=model,
             messages=[
                 {"role": "system", "content": prompt},
                 # Construct user message content including metadata if provided (mainly for final reduction)
                 {"role": "user", "content": (
                     # Case 1: Metadata and Images
                     [
                         {"type": "text", "text": f"{code_metadata}\n\n--- Text to Summarize ---\n{chunk_text}"}
                     ] + image_inputs
                     if code_metadata and image_inputs
                     # Case 2: Only Metadata
                     else f"{code_metadata}\n\n--- Text to Summarize ---\n{chunk_text}"
                     if code_metadata
                     # Case 3: Only Images
                     else [{"type": "text", "text": chunk_text}] + image_inputs
                     if image_inputs
                     # Case 4: Only Text
                     else chunk_text
                 )}
             ],
             temperature=temperature,
             max_tokens=max_tokens,
         )
         # Safely extract content from response
         choices = getattr(response, "choices", [])
         if not choices:
             raise ValueError("No choices in chunk summary response")
         
         choice = choices[0]
         content = None
         
         if hasattr(choice, "message") and hasattr(choice.message, "content"):
             content = choice.message.content
         elif hasattr(choice, "delta") and hasattr(choice.delta, "content"):
             content = choice.delta.content
         else:
             raise ValueError("Could not extract content from response")
         
         if not content or not content.strip():
             raise ValueError("Empty or whitespace-only chunk summary")
             
         summary = content.strip()
         logger.debug(f"Chunk summary ({len(summary.split())} tokens): {summary[:100]}...")
         return summary
     except Exception as e:
          logger.error(f"Error summarizing chunk: {e}", exc_info=True)
          # Return error message or raise? Let's return an error indicator for gather
          return f"Error summarizing chunk: {e}"


# --- Recursive Reduction Helper ---

async def _recursive_reduce(
    text_to_reduce: str,
    config: Dict[str, Any],
    final_summary_prompt: str,
    image_inputs: Optional[List[Dict[str, Any]]],
    code_metadata: Optional[str],
    current_depth: int = 0
) -> str:
    """
    Recursively reduces text if it exceeds context limits, otherwise summarizes.
    """
    # Get config values
    context_limit_threshold = config.get("context_limit_threshold", 3800)
    max_recursion_depth = config.get("max_recursion_depth", 3) # Added config
    chunk_size = config.get("chunk_size", 3500)
    overlap_size = config.get("overlap_size", 2)
    default_chunk_prompt = "Summarize the key points of this text segment:"
    chunk_summary_prompt = config.get("chunk_summary_prompt", default_chunk_prompt) # Needed for recursive step
    # No need to log here as it's logged in the caller (summarize_text)

    input_tokens = estimate_token_count(text_to_reduce)

    if input_tokens <= context_limit_threshold:
        # Base case: Text fits, perform final summarization
        logger.debug(f"Recursion depth {current_depth}: Text fits ({input_tokens} <= {context_limit_threshold}). Performing final summarization.")
        try:
            final_summary = await summarize_chunk(
                text_to_reduce,
                config,
                final_summary_prompt,
                image_inputs=image_inputs,
                code_metadata=code_metadata
            )
            return final_summary
        except Exception as e:
            logger.error(f"Final reduction summarization failed at depth {current_depth}: {e}", exc_info=True)
            raise ValueError(f"Final reduction summarization failed: {e}")
    else:
        # Recursive case: Text too long
        logger.debug(f"Recursion depth {current_depth}: Combined text too long ({input_tokens} > {context_limit_threshold}).")

        if current_depth >= max_recursion_depth:
            # Max depth reached, truncate
            logger.warning(f"Max recursion depth ({max_recursion_depth}) reached. Truncating text ({input_tokens} tokens) for final pass.")
            # Use a slightly more robust truncation based on estimated characters per token
            estimated_chars_per_token = 4 # A common estimate
            truncation_limit_chars = int(context_limit_threshold * estimated_chars_per_token)
            truncated_text = text_to_reduce[:truncation_limit_chars]
            logger.debug(f"Truncated text to approx {estimate_token_count(truncated_text)} tokens.")
            try:
                 # Summarize the truncated text
                 final_summary = await summarize_chunk(
                     truncated_text,
                     config,
                     final_summary_prompt,
                     image_inputs=image_inputs,
                     code_metadata=code_metadata
                 )
                 return final_summary
            except Exception as e:
                 logger.error(f"Final summarization failed after truncation at max depth {current_depth}: {e}", exc_info=True)
                 raise ValueError(f"Final summarization failed after truncation: {e}")
        else:
            # Initiate recursive reduction
            logger.debug(f"Recursive reduction initiated at depth {current_depth}. Re-chunking...")
            # Ensure NLTK data is available (might be redundant if create_chunks does it, but safe)
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK 'punkt' tokenizer for recursive step...")
                nltk.download('punkt', quiet=True)

            sentences = nltk.sent_tokenize(text_to_reduce)
            recursive_chunks = create_chunks_with_overlap(sentences, chunk_size, overlap_size)
            if not recursive_chunks:
                 logger.warning(f"Recursive chunking at depth {current_depth} resulted in zero chunks. Returning original text for this level.")
                 # Fallback: try summarizing the original text for this level, hoping it works (might fail if too long)
                 try:
                     final_summary = await summarize_chunk(
                         text_to_reduce, config, final_summary_prompt, image_inputs=image_inputs, code_metadata=code_metadata
                     )
                     return final_summary
                 except Exception as e:
                     logger.error(f"Fallback summarization failed at depth {current_depth} after empty recursive chunks: {e}", exc_info=True)
                     raise ValueError(f"Recursive reduction failed due to empty chunks and fallback failure: {e}")


            recursive_chunk_texts = [' '.join(chunk) for chunk in recursive_chunks]

            logger.info(f"Recursively summarizing {len(recursive_chunk_texts)} new chunks at depth {current_depth}...")
            recursive_summaries_results = await asyncio.gather(
                *(summarize_chunk(chunk_text, config, chunk_summary_prompt) for chunk_text in recursive_chunk_texts),
                return_exceptions=True
            )

            # Filter errors
            recursive_summaries = []
            for i, result in enumerate(recursive_summaries_results):
                 if isinstance(result, Exception):
                      logger.error(f"Error summarizing recursive chunk {i} at depth {current_depth}: {result}", exc_info=True)
                 elif isinstance(result, str) and result.startswith("Error summarizing chunk:"):
                      logger.error(f"Error string returned for recursive chunk {i} at depth {current_depth}: {result}")
                 elif isinstance(result, str):
                      recursive_summaries.append(result)
                 else:
                      logger.error(f"Unexpected result type for recursive chunk {i} at depth {current_depth}: {type(result)} - {result}")

            if not recursive_summaries:
                 raise ValueError(f"All recursive chunk summarization attempts failed at depth {current_depth}.")

            new_combined_text = "\n\n".join(recursive_summaries)
            logger.debug(f"Combined {len(recursive_summaries)} recursive summaries at depth {current_depth}. New combined tokens: {estimate_token_count(new_combined_text)}")

            # Recursive call
            return await _recursive_reduce(
                new_combined_text,
                config,
                final_summary_prompt,
                image_inputs,
                code_metadata,
                current_depth + 1
            )


# --- Main Summarization Function ---

async def summarize_text(
    text: str,
    config: Dict[str, Any],
    image_inputs: Optional[List[Dict[str, Any]]] = None, # Added: Expects list of formatted image dicts
    code_metadata: Optional[str] = None # Added code metadata string
) -> str:
    """
    Summarize text (and optionally images, with code metadata context) using LLM.
    Handles long text via chunking (MapReduce) and recursive reduction.
    """
    if not text:
        raise ValueError("Input text cannot be empty")

    model_name = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.7)
    # Define a reasonable context limit threshold
    context_limit_threshold = config.get("context_limit_threshold", 3800)
    chunk_size = config.get("chunk_size", 3500) # Size for individual chunks
    overlap_size = config.get("overlap_size", 2) # Number of sentences overlap

    input_tokens = estimate_token_count(text)
    logger.info(f"Input text estimated tokens: {input_tokens}")

    # --- Direct Summarization for Short Text ---
    if input_tokens <= context_limit_threshold:
        logger.info("Input text is within context limit. Summarizing directly.")
        # Choose prompt based on whether it's code
        is_code = config.get("is_code_summary", False)
        if is_code:
            default_code_prompt = "Summarize the following code, explaining its purpose, inputs, outputs, and key logic. Use the provided code structure context if available."
            system_prompt = config.get("code_system_prompt", default_code_prompt)
            if system_prompt != default_code_prompt:
                logger.info("Using custom code system prompt for direct summarization.")
            # Allow potentially longer summaries for code
            max_tokens = min(max(200, int(input_tokens * 0.7)), 800)
        else:
            default_system_prompt = "Summarize the following text concisely, preserving key information."
            system_prompt = config.get("system_prompt", default_system_prompt)
            if system_prompt != default_system_prompt:
                logger.info("Using custom system prompt for direct summarization.")
            # Adjust max_tokens calculation for better summary length
            max_tokens = min(max(100, int(input_tokens * 0.6)), 500) # Aim for 60% length, capped

        # Construct user content, potentially including metadata
        user_content = text
        if is_code and code_metadata:
            user_content = f"{code_metadata}\n\n--- Code to Summarize ---\n{text}"
            logger.debug("Prepending code metadata to user prompt for direct summarization.")

        # Format for multimodal if images are present
        if image_inputs:
            user_content_final = [{"type": "text", "text": user_content}] + image_inputs
        else:
            user_content_final = user_content

        try:
            response = await asyncio.to_thread( # Run sync reliable_completion in thread
                reliable_completion,
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content_final} # Use potentially modified user content
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Safely extract summary
            choices = getattr(response, "choices", [])
            if not choices: raise ValueError("No choices in direct summary response")
            message = getattr(choices[0], "message", None)
            if not message: raise ValueError("No message in direct summary first choice")
            summary = getattr(message, "content", "").strip()
            if not summary: raise ValueError("Empty or whitespace-only direct summary")

            output_tokens = len(summary.split())
            logger.info(
                f"Direct Summarization stats: Input={input_tokens} tokens, "
                f"Output={output_tokens} tokens, "
                f"Reduction={(1 - output_tokens/input_tokens)*100:.1f}%" if input_tokens > 0 else "N/A"
            )
            return summary
        except Exception as e:
             logger.error(f"Direct summarization failed: {e}", exc_info=True)
             raise ValueError(f"Direct summarization failed: {e}")


    # --- Chunking Summarization for Long Text (MapReduce) ---
    else:
        logger.info("Input text exceeds context limit. Using chunking (MapReduce).")

        # 1. Split into sentences
        sentences = nltk.sent_tokenize(text)

        # 2. Create chunks
        chunks = create_chunks_with_overlap(sentences, chunk_size, overlap_size)
        if not chunks:
             logger.warning("Text chunking resulted in zero chunks. Returning original text.")
             return text # Or raise error?
        chunk_texts = [' '.join(chunk) for chunk in chunks]

        # 3. Summarize each chunk (Map)
        # Code chunks are summarized like text chunks for the Map step
        default_chunk_prompt = "Summarize the key points of this text segment:"
        chunk_summary_prompt = config.get("chunk_summary_prompt", default_chunk_prompt)
        if chunk_summary_prompt != default_chunk_prompt:
            logger.info("Using custom chunk summary prompt.")
        # Use asyncio.gather for concurrent chunk summarization
        logger.info(f"Summarizing {len(chunk_texts)} chunks concurrently...")
        chunk_summaries_results = await asyncio.gather(
            *(summarize_chunk(chunk_text, config, chunk_summary_prompt) for chunk_text in chunk_texts),
            return_exceptions=True # Return exceptions instead of raising immediately
        )

        # Filter out errors and log them
        chunk_summaries = []
        for i, result in enumerate(chunk_summaries_results):
             if isinstance(result, Exception):
                  # Log the exception traceback for better debugging
                  logger.error(f"Error summarizing chunk {i}: {result}", exc_info=True)
             # Explicitly check if result is a string before using 'in'
             elif isinstance(result, str) and result.startswith("Error summarizing chunk:"):
                  logger.error(f"Error string returned for chunk {i}: {result}")
             elif isinstance(result, str): # Check if it's a valid string summary
                  chunk_summaries.append(result)
             else: # Handle unexpected types if necessary
                  logger.error(f"Unexpected result type for chunk {i}: {type(result)} - {result}")

        if not chunk_summaries:
             raise ValueError("All chunk summarization attempts failed.")

        # 4. Combine chunk summaries
        combined_summary_text = "\n\n".join(chunk_summaries)
        combined_tokens = estimate_token_count(combined_summary_text)
        logger.info(f"Combined {len(chunk_summaries)} chunk summaries. Combined tokens: {combined_tokens}")

        # 5. Reduce the combined summaries (potentially recursively)
        is_code = config.get("is_code_summary", False)
        if is_code:
            default_code_final_prompt = "Synthesize the following code segment summaries into a single, coherent explanation of the overall code's purpose and structure. Use the provided overall code structure context if available."
            # Note: text_summarizer expects 'final_summary_prompt' or 'code_system_prompt', let's align keys
            # We'll use 'code_system_prompt' if provided, otherwise the specific default.
            # If 'final_summary_prompt' is ALSO provided, it takes precedence for the final step even for code.
            final_summary_prompt = config.get("final_summary_prompt", config.get("code_system_prompt", default_code_final_prompt))
            if final_summary_prompt != default_code_final_prompt:
                 logger.info("Using custom final prompt (or code system prompt) for code reduction.")
        else:
            default_final_prompt = "Synthesize the following summaries into a single, coherent summary:"
            final_summary_prompt = config.get("final_summary_prompt", default_final_prompt)
            if final_summary_prompt != default_final_prompt:
                 logger.info("Using custom final summary prompt.")

        try:
            # Call the recursive reduction helper function
            final_summary = await _recursive_reduce(
                text_to_reduce=combined_summary_text,
                config=config,
                final_summary_prompt=final_summary_prompt,
                image_inputs=image_inputs, # Pass images through
                code_metadata=code_metadata, # Pass metadata through
                current_depth=0 # Start recursion depth at 0
            )
        except Exception as e:
             # Error already logged in _recursive_reduce, just re-raise
             raise ValueError(f"Recursive reduction process failed: {e}")


        output_tokens = estimate_token_count(final_summary) # Use estimate_token_count for consistency
        logger.info(
            f"Chunked Summarization stats: Input={input_tokens} tokens, "
            f"Output={output_tokens} tokens, "
            f"Reduction={(1 - output_tokens/input_tokens)*100:.1f}%" if input_tokens > 0 else "N/A"
        )
        return final_summary

# --- Main Execution Block ---
if __name__ == "__main__":

    # Define an async main function
    async def main(run_fixture_test: bool = False): # Add flag to run fixture test optionally
        # --- Configuration ---
        config = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            # Chunking config
            "context_limit_threshold": 3800,
            "chunk_size": 3500,
            "overlap_size": 2, # Sentences
            # Embedding config (defaulting to local)
            "embedding_config": {
                "provider": "local",
                "local_model": "nomic-ai/modernbert-embed-base"
            },
        }

        validation_failures = {} # Initialize here
        if run_fixture_test:
            logger.info("--- Running Fixture Validation ---")
            try:
                # Use relative path assuming tests/fixtures exists at project root
                fixture_path = Path("tests/fixtures/summarize_text_expected.json")
                logger.debug(f"Loading test fixture from: {fixture_path}")
                with open(fixture_path, "r") as f:
                    test_data = json.load(f)
                logger.info("Successfully loaded test fixture")
            except Exception as e:
                logger.error(f"Failed to load test fixture: {e}")
                sys.exit(1)

            validation_failures = {}
            for i, test_case in enumerate(test_data["test_cases"]):
                 logger.info(f"Processing fixture test case {i+1}")
                 logger.debug(f"Input text: {test_case['input_text'][:100]}...")
                 # Summarize the short fixture text (should use direct summarization)
                 summary = await summarize_text(test_case["input_text"], config)
                 logger.debug(f"Generated summary: {summary}")
                 # Validate against the fixture data, unpack metrics (though not used here)
                 passed, failures, _ = await validate_summary(summary, test_case, config["embedding_config"])
                 if not passed:
                      validation_failures[f"fixture_test_case_{i}"] = failures

            if not validation_failures:
                 print("\n✅ FIXTURE VALIDATION COMPLETE - All fixture tests passed.")
            else:
                 print("\n❌ FIXTURE VALIDATION FAILED - Fixture tests failed.")
                 # Print details (similar to below)
                 for test_case, failures in validation_failures.items():
                      print(f"\nTest Case: {test_case}")
                      for field, details in failures.items():
                           print(f"  - {field}: Expected: {details.get('expected')}, Got: {details.get('actual', details.get('error'))}")
                 sys.exit(1) # Exit if fixture tests fail

        # --- Long Text Summarization & Validation (Default Run) ---
        logger.info("--- Running Long Text Validation ---")
        wiki_title = "Artificial Intelligence"
        max_wiki_tokens = 6000 # Limit input size for testing

        logger.info(f"Loading Wikipedia text for '{wiki_title}' (max_tokens={max_wiki_tokens})...")
        original_text, token_count = load_wikipedia_text(wiki_title, max_tokens=max_wiki_tokens)
        if not original_text:
            logger.error("Failed to load Wikipedia text. Exiting.")
            sys.exit(1)
        logger.info(f"Loaded {token_count} tokens.")

        logger.info("Starting summarization...")
        try:
            summary = await summarize_text(original_text, config)
            logger.info(f"Final Summary generated ({len(summary.split())} words).")
            logger.debug(f"Final Summary Preview: {summary[:200]}...")
        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            sys.exit(1)

        logger.info("Validating summary against original text (Semantic Similarity)...")
        validation_data = {
            "input_text": original_text,
            "expected_summary": original_text,
            "expected_properties": {
                "semantic_similarity_threshold": 0.6,
                "compression_ratio": 0, # Dummy
                "key_concepts": [] # Dummy
            }
        }

        long_text_validation_failures = {}
        try:
            # Unpack metrics (though not used here)
            passed, failures, _ = await validate_summary(summary, validation_data, config["embedding_config"])
            if not passed:
                 long_text_validation_failures = failures

        except Exception as e:
             logger.error(f"Validation step failed with exception: {e}", exc_info=True)
             print("\n❌ LONG TEXT VALIDATION ERROR - An exception occurred during validation.")
             sys.exit(1)

        if not long_text_validation_failures:
            print("\n✅ LONG TEXT VALIDATION COMPLETE - Summary generated and semantic similarity check passed.")
            # Only exit 0 if fixture tests also passed (or weren't run)
            if not run_fixture_test or not validation_failures:
                 sys.exit(0)
            else:
                 # Fixture tests failed earlier, so exit 1
                 sys.exit(1)
        else:
            print("\n❌ LONG TEXT VALIDATION FAILED - Summary semantic similarity check failed.")
            print("FAILURE DETAILS:")
            for field, details in long_text_validation_failures.items():
                 if field == "semantic_similarity":
                      print(f"  - {field}: Expected: {details.get('expected')}, Got: {details.get('actual')}")
                 elif field == "embedding_generation":
                      print(f"  - {field}: Error: {details.get('error')}")
                 else:
                      print(f"  - {field}: {details}")
            sys.exit(1)

    # Run the async main function
    try:
        # Add argument parsing to optionally run fixture test
        import argparse
        parser = argparse.ArgumentParser(description="Run summarizer validation.")
        parser.add_argument('--run-fixture-test', action='store_true', help='Run validation against the fixture file.')
        args = parser.parse_args()

        asyncio.run(main(run_fixture_test=args.run_fixture_test))
    except Exception as e:
        logger.error(f"❌ MAIN EXECUTION ERROR: {str(e)}", exc_info=True)
        sys.exit(1)
