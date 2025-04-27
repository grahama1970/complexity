import json
import os

import nltk
import wikipedia
from loguru import logger
from nltk.tokenize import word_tokenize

from app.backend.utils.loguru_setup import setup_logger

setup_logger()
nltk.download('punkt', quiet=True)

def load_wikipedia_text(title, max_tokens=6000, save_path='documents'):
    """
    Load Wikipedia text, save it to a file, or load from file if it exists.
    
    Args:
        title (str): The title of the Wikipedia page.
        max_tokens (int): Maximum number of tokens to include.
        save_path (str): Directory to save/load the text file.
    
    Returns:
        tuple: (text content, token count)
    """
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"{title.replace(' ', '_')}.json")
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return (data['content'], data['token_count'])
        except Exception as e:
            logger.error(f'Error loading file {filename}: {str(e)}')
    try:
        page = wikipedia.page(title)
        paragraphs = page.content.split('\n\n')
        (chunks, current_chunk, current_token_count) = ([], [], 0)
        for paragraph in paragraphs:
            paragraph_tokens = word_tokenize(paragraph)
            if current_token_count + len(paragraph_tokens) > max_tokens:
                break
            current_chunk.append(paragraph)
            current_token_count += len(paragraph_tokens)
        content = ' '.join(current_chunk)
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump({'content': content, 'token_count': current_token_count}, file, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f'Error saving file {filename}: {str(e)}')
        return (content, current_token_count)
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
        logger.error(f"Error fetching Wikipedia page '{title}': {str(e)}")
        return ('', 0)

def usage_example(title):
    """
    Demonstrate the usage of load_wikipedia_text function.
    
    Args:
    title (str): The title of the Wikipedia page to extract text from.
    
    Returns:
    None
    """
    (text, token_count) = load_wikipedia_text(title)
    print(f'Extracted text (first 100 characters): {text[:100]}...')
    print(f'Token count: {token_count}')

def main():
    """
    Main function to run the usage example.
    """
    title = 'Artificial Intelligence'
    usage_example(title)
if __name__ == '__main__':
    main()