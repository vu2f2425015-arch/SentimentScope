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
    2. Expand negation contractions (don't -> do not, can't -> cannot, etc.)
    3. Strip HTML tags, URLs, user mentions (@username), numbers, special characters
    4. Tokenize into words
    5. Remove non-negation stopwords and lemmatize
    6. Generate compound negation tokens (e.g. 'not_nice', 'not_good') when negations occur
       so linear Bag-of-Words models decouple negated terms from positive unigram/char subwords.
    
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
        
    # 7. Remove non-negation stopwords, lemmatize, and generate compound negation tokens
    cleaned_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in NEGATION_WORDS and i + 1 < len(tokens):
            next_token = tokens[i+1]
            if next_token not in _stop_words:
                try:
                    lem_next = _lemmatizer.lemmatize(next_token)
                except Exception:
                    lem_next = next_token
                cleaned_tokens.append("not")
                cleaned_tokens.append(f"not_{lem_next}")
                i += 2
                continue
        if token not in _stop_words and len(token) > 1:
            try:
                cleaned_tokens.append(_lemmatizer.lemmatize(token))
            except Exception:
                cleaned_tokens.append(token)
        i += 1
    
    return " ".join(cleaned_tokens)

