def estimate_token_count(text: str) -> int:
    import nltk
    'Estimate the number of tokens in a given text.'
    return len(nltk.word_tokenize(text))