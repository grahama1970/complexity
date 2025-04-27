import asyncio
from typing import Any, Dict, List, Optional

import nltk
import pandas as pd
from llm_client.handle_user_query import handle_user_query
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from app.backend.summarize.utils.estimate_token_count import estimate_token_count
from app.backend.utils.json_cleaner import clean_json_string
from app.backend.utils.load_json_file import load_json_file


class LLMParams(BaseModel):
    model: str = Field(default='openai/gpt-4o-mini', description='The model to use for summarization.')
    json_mode: bool = Field(default=False, description='Whether to use JSON mode for the LLM response.')

    class Config:
        validate_assignment = True

class SummarizerConfig(BaseModel):
    context_length: int = Field(default=5000, description='Maximum number of tokens in a chunk.')
    overlap_size: Optional[int] = Field(default=None, description='Number of tokens to overlap between chunks.')
    text_field: str = Field(default='line_text', description='Field name in JSON containing text data.')
    description_field: Optional[str] = Field(default='description', description='Field name in JSON containing description data, if present.')
    table_field: Optional[str] = Field(default='table_data', description='Field name in JSON containing table data, if present.')
    type_field: str = Field(default='type', description='Field name in JSON containing the type of data.')
    llm_params: LLMParams = Field(default_factory=LLMParams, description='LLM parameters for model selection and settings.')

    class Config:
        validate_assignment = True

def create_chunks_with_overlap(sentences: List[str], chunk_size: int, overlap_size: Optional[int]=None) -> List[List[str]]:
    """
    Creates chunks of sentences with overlap, handling long sentences more effectively.

    Args:
        sentences (List[str]): List of sentences to be chunked.
        chunk_size (int): Maximum number of tokens in a chunk.
        overlap_size (Optional[int]): Number of sentences to overlap between chunks.
                                     If None, it will default to a value based on content length.

    Returns:
        List[List[str]]: List of chunks, where each chunk is a list of sentences.
    """
    if overlap_size is None:
        overlap_size = max(1, len(sentences) // 20)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        if sentence_tokens > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            chunks.append([sentence])
            current_chunk = []
            current_chunk_tokens = 0
        elif current_chunk_tokens + sentence_tokens > chunk_size:
            chunks.append(current_chunk)
            current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
            current_chunk_tokens = sum((estimate_token_count(s) for s in current_chunk))
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

async def summarize_chunk(chunk: str, llm_params: dict, db=None) -> str:
    """
    Summarizes a single chunk of text.

    Args:
        chunk (str): The chunk of text to be summarized.

    Returns:
        str: A summary of the chunk.
    """
    messages = [{'role': 'system', 'content': 'Summarize the following text concisely:'}, {'role': 'user', 'content': chunk}]
    try:
        response = await handle_user_query(messages, llm_params, db=db)
        if isinstance(response, object) and hasattr(response, 'choices'):
            message_content = response.choices[0].get('message', {}).get('content')
            message_content = clean_json_string(message_content, return_dict=False)
            if message_content:
                return message_content.strip()
            else:
                logger.error(f'Unexpected LLM response structure for summary: {response}')
                return 'Error: Unexpected response structure'
        else:
            logger.error(f'Invalid response structure: {response}')
            return 'Error: Invalid response structure'
    except Exception as e:
        logger.error(f'Error getting LLM response: {str(e)}')
        return f'Error: {str(e)}'

async def summarize_structured_json_data(json_data: List[Dict[str, Any]], summarizer_config: dict, db=None):
    try:
        validated_config = SummarizerConfig(**summarizer_config)
        logger.debug(f'Validated config: {validated_config}')
    except ValidationError as e:
        logger.error(f'Invalid configuration provided: {e}')
        return f'Error: Invalid configuration - {str(e)}'
    token_count = 0
    combined_content = []
    for node in json_data:
        content_type = node.get(validated_config.type_field)
        line_text = node.get(validated_config.text_field) if content_type == 'text' else None
        table_data = node.get(validated_config.table_field) if content_type == 'table' else None
        description_data = node.get(validated_config.description_field) if content_type in ['table', 'image'] and node.get(validated_config.description_field) else None
        if line_text:
            sentences = nltk.sent_tokenize(line_text)
            token_count += sum((estimate_token_count(sentence) for sentence in sentences))
            combined_content.extend(sentences)
        if table_data:
            table_csv = pd.DataFrame(table_data).to_csv(index=False)
            table_summary = f'Table Data:\n{table_csv}'
            tokens_in_table = estimate_token_count(table_summary)
            token_count += tokens_in_table
            combined_content.append(table_summary)
        if description_data:
            description_summary = f'Description: {description_data}'
            tokens_in_description = estimate_token_count(description_summary)
            token_count += tokens_in_description
            combined_content.append(description_summary)
    if token_count <= validated_config.context_length:
        unified_content = ' '.join(combined_content)
        final_summary = await summarize_chunk(unified_content, validated_config.llm_params.model_dump(), db=db)
        logger.info(f'Final Summary: {final_summary}')
        return final_summary
    else:
        chunks_to_process = create_chunks_with_overlap(combined_content, validated_config.context_length, validated_config.overlap_size)
        chunk_summaries = await asyncio.gather(*(summarize_chunk(' '.join(chunk), validated_config.llm_params.model_dump(), db=db) for chunk in chunks_to_process))
        combined_chunk_summaries = ' '.join(chunk_summaries)
        final_summary = await summarize_chunk(combined_chunk_summaries, validated_config.llm_params.model_dump())
        logger.info(f'Final Summary: {final_summary}')
        return final_summary
if __name__ == '__main__':
    from app.backend.utils.loguru_setup import setup_logger
    setup_logger(log_level='INFO')
    section_content = [{'type': 'text', 'page': 1, 'bbox': [72, 82, 316, 588], 'text': '4.1.5.4. BHT (Branch History Table) submodule BHT is implemented as a memory which is composed of BHTDepth configuration parameter entries. The lower address bits of the virtual address point to the memory entry. When a branch instruction is resolved by the EX_STAGE module, the branch PC and the taken (or not taken) status information is stored in the Branch History Table. The Branch History Table is a table of two-bit saturating counters that takes the virtual address of the current fetched instruction by the CACHE. It states whether the current branch request should be taken or not. The two bit counter is updated by the successive execution of the instructions as shown in the following figure. When a branch instruction is pre-decoded by instr_scan submodule, the BHT valids whether the PC address is in the BHT and provides the taken or not prediction. The BHT is never flushed.', 'text_type': 'paragraph'}, {'type': 'image', 'page': 1, 'bbox': [73.5, 338.99365234375, 541.5, 489.7436828613281], 'id': '4afdfc56-4a32-4823-8dd3-f548224d11b7', 'image_file': '/home/grahama/dev/embedding_network/verifaix/pymupdf_extractor/images/BHT_CV32A65X_page_1_block_17.png', 'description': None}, {'type': 'table', 'page': 1, 'bbox': [72.0, 611.04, 541.92, 685.92], 'table_data': [{'0': 'Signal', '1': 'IO', '2': 'Description', '3': 'connexion', '4': 'Type', 'section_title': '4.1.5.4. BHT (Branch History Table) submodule'}, {'0': 'clk_i', '1': 'in', '2': 'Subsystem Clock', '3': 'SUBSYSTEM', '4': 'logic', 'section_title': '4.1.5.4. BHT (Branch History Table) submodule'}, {'0': 'rst_ni', '1': 'in', '2': 'Asynchronous reset active low', '3': 'SUBSYSTEM', '4': 'logic', 'section_title': '4.1.5.4. BHT (Branch History Table) submodule'}, {'0': 'vpc_i', '1': 'in', '2': 'Virtual PC', '3': 'CACHE', '4': 'logic[CVA6Cfg.VLEN-1:0]', 'section_title': '4.1.5.4. BHT (Branch History Table) submodule'}, {'0': 'bht_update_i', '1': 'in', '2': 'Update bht with resolved address', '3': 'EXECUTE', '4': 'bht_update_t', 'section_title': '4.1.5.4. BHT (Branch History Table) submodule'}, {'0': 'bht_prediction_o', '1': 'out', '2': 'Prediction from bht', '3': 'FRONTEND', '4': 'ariane_pkg::bht_prediction_t[CVA6Cfg.INSTR_PER_FETCH-1:0]', 'section_title': '4.1.5.4. BHT (Branch History Table) submodule'}], 'description': None}]
    section_summary = load_json_file('/home/grahama/dev/embedding_network/verifaix/pymupdf_extractor/data/section_summary.json')
    config = {'context_length': 5000, 'overlap_size': 50, 'text_field': 'text', 'table_field': 'table_data', 'type_field': 'type', 'llm_params': {'model': 'openai/gpt-4o-mini', 'json_mode': False}}
    asyncio.run(summarize_structured_json_data(section_content, config))