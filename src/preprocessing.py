import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

NEGATION_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'none', 'cannot', 'cant',
    "n't", 'don', "don't", 'doesn', "doesn't", 'didnt', "didn't",
    'isnt', "isn't", 'arent', "aren't", 'wasnt', "wasn't", 'werent', "weren't",
    'havent', "haven't", 'hasnt', "hasn't", 'hadnt', "hadn't",
    'wont', "won't", 'wouldnt', "wouldn't", 'shant', "shan't", 'shouldnt', "shouldn't",
    'musnt', "mustn't", 'couldnt', "couldn't", 'against', 'without'
}

# Ensure required NLTK resources are downloaded
def _init_nltk():
    import os
    tmp_dir = os.path.join("/tmp", "nltk_data")
    if os.path.exists("/tmp"):
        try:
            os.makedirs(tmp_dir, exist_ok=True)
        except Exception:
            pass
        if tmp_dir not in nltk.data.path:
            nltk.data.path.append(tmp_dir)
        
    resources = ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except Exception:
            try:
                nltk.download(resource, download_dir=tmp_dir if os.path.exists("/tmp") else None, quiet=True)
            except Exception:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass

_init_nltk()

_lemmatizer = WordNetLemmatizer()
try:
    _stop_words = set(stopwords.words('english')) - NEGATION_WORDS
except Exception:
    _init_nltk()
    _stop_words = set(stopwords.words('english')) - NEGATION_WORDS


def minimal_clean_text(text: str) -> str:
    """
    Minimal text normalization for Transformer tokenizers.
    Unescapes HTML entities, normalizes whitespace.
    Preserves casing, punctuation, stopwords, and negations intact.
    """
    if not text or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    """
    Preprocesses raw input text for classical sentiment analysis.
    
    Steps:
    1. Lowercase text
    2. Remove URLs, user mentions (@username), HTML tags, and special characters/numbers
    3. Tokenize into words
    4. Remove English stopwords (PRESERVING negation words like not, no, never, n't)
    5. Lemmatize tokens
    6. Rejoin clean tokens into a normalized string
    
    Args:
        text (str): Raw input text string
        
    Returns:
        str: Preprocessed, cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # Expand negation contractions to preserve polarity before punctuation stripping
    text = re.sub(r"\bcan['’]?t\b", "cannot", text)
    text = re.sub(r"\bwon['’]?t\b", "will not", text)
    text = re.sub(r"\bshan['’]?t\b", "shall not", text)
    text = re.sub(r"\bn['’]t\b", " not", text)
    
    # 2. Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Strip URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
    
    # 4. Strip user mentions (@username)
    text = re.sub(r'@\w+', ' ', text)
    
    # 5. Strip punctuation, numbers, and special characters (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 6. Tokenize (fallback to simple split if word_tokenize has issues)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
        
    # 7. Remove non-negation stopwords and 8. Lemmatize (with fallback if WordNet lookup fails)
    cleaned_tokens = []
    for token in tokens:
        if token not in _stop_words and len(token) > 1:
            try:
                cleaned_tokens.append(_lemmatizer.lemmatize(token))
            except Exception:
                cleaned_tokens.append(token)
    
    return " ".join(cleaned_tokens)

