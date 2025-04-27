from smolagent import Tool
from smolagent.llm.litellm_utils import RetryingLiteLLMModel, initialize_litellm_cache_sync
from smolagent.utils.json_utils import clean_json_string
from smolagent.summarizer.utils.estimate_token_count import estimate_token_count
from typing import Any, Dict, List, Optional
import asyncio
import nltk
import os
from loguru import logger

class SummarizationTool(Tool):
    """
    Advanced document summarization tool with hierarchical processing
    and rolling window context maintenance.
    """
    
    name = "hierarchical_summarizer"
    description = "Summarizes large documents using chunking and multi-step synthesis"
    inputs = {
        "text": {"type": "string", "description": "Text content to summarize"},
        "config": {
            "type": "object",
            "description": "Configuration for summarization process",
            "default": {
                "chunk_size": 3000,
                "overlap_size": 100,
                "llm_params": {"model": "anthropic/claude-3-5-sonnet-20240620"},
                "max_retries": 3
            }
        }
    }
    output_type = "string"

    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.default_config = default_config or {
            "chunk_size": 3000,
            "overlap_size": 100,
            "llm_params": {"model": "anthropic/claude-3-5-sonnet-20240620"},
            "max_retries": 3
        }
        
        self.llm = None
        self._initialize_llm()
        initialize_litellm_cache_sync()
        nltk.download('punkt', quiet=True)

    def _initialize_llm(self):
        """Initialize the LLM with automatic retries"""
        try:
            self.llm = RetryingLiteLLMModel(
                self.default_config["llm_params"]["model"],
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            logger.success("Summarization LLM initialized with retry capabilities")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise RuntimeError("Failed to initialize summarization LLM")

    async def forward(self, text: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Execute full hierarchical summarization workflow"""
        merged_config = {**self.default_config, **(config or {})}
        
        try:
            # Phase 1: Document chunking
            sentences = nltk.sent_tokenize(text)
            chunks = self._create_chunks_with_overlap(
                sentences,
                merged_config["chunk_size"],
                merged_config["overlap_size"]
            )
            
            # Phase 2: Parallel chunk summarization
            chunk_tasks = [
                self._summarize_chunk(" ".join(chunk), merged_config)
                for chunk in chunks
            ]
            chunk_summaries = await asyncio.gather(*chunk_tasks)
            
            # Phase 3: Final synthesis
            return await self._final_summarization("\n".join(chunk_summaries), merged_config)
            
        except Exception as e:
            logger.error(f"Summarization pipeline failed: {e}")
            raise RuntimeError(f"Summarization error: {str(e)}")

    def _create_chunks_with_overlap(self, sentences: List[str], chunk_size: int, overlap_size: int) -> List[List[str]]:
        """Create context-aware chunks with rolling window"""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = estimate_token_count(sentence)
            
            # Handle oversized sentences
            if sentence_tokens > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                chunks.append([sentence])
                current_chunk = []
                current_tokens = 0
                continue
                
            # Add to current chunk if space allows
            if current_tokens + sentence_tokens <= chunk_size:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                chunks.append(current_chunk)
                current_chunk = current_chunk[-overlap_size:] + [sentence]
                current_tokens = sum(estimate_token_count(s) for s in current_chunk)
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    async def _summarize_chunk(self, chunk: str, config: Dict[str, Any]) -> str:
        """Summarize individual chunk using retry-equipped LLM"""
        try:
            response = await self.llm.run([
                {"role": "system", "content": "Summarize this chunk preserving key facts and context:"},
                {"role": "user", "content": chunk}
            ])
            return clean_json_string(response)
        except Exception as e:
            logger.error(f"Chunk summarization failed: {e}")
            raise

    async def _final_summarization(self, partial_summaries: str, config: Dict[str, Any]) -> str:
        """Synthesize partial summaries into final document summary"""
        try:
            response = await self.llm.run([
                {"role": "system", "content": "Synthesize these partial summaries into a coherent final summary:"},
                {"role": "user", "content": partial_summaries}
            ])
            return clean_json_string(response)
        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")
            raise

# Usage Example
async def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    summarizer = SummarizationTool()
    
    with open("long_document.txt") as f:
        summary = await summarizer.forward(f.read())
        print(f"Final Summary:\n{summary}")

if __name__ == "__main__":
    asyncio.run(main())