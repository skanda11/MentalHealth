# preprocessing/preprocess.py
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from utils.helpers import load_data, save_data # Adjusted import path as utils is in parent dir relative to this file

print("DEBUG: preprocess.py - Module loaded.")

# Ensure NLTK data is downloaded
try:
    print("DEBUG: preprocess.py - Checking for NLTK 'wordnet' and 'punkt' resources.")
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('tokenizers/punkt')
    print("DEBUG: preprocess.py - NLTK resources found.")
except LookupError:
    print("DEBUG: preprocess.py - NLTK resources not found. Downloading...")
    nltk.download('wordnet')
    nltk.download('punkt')
    print("DEBUG: preprocess.py - NLTK resources downloaded.")


def clean_text(text, lemmatize=True):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
    return text

def preprocess_data(data_config): # Takes only data_config for clarity
    print("DEBUG: preprocess.py - Entering preprocess_data function.")
    raw_data_path = data_config['raw_data_path']
    processed_data_path = data_config['processed_data_path']
    text_column = data_config['text_column']
    target_column = data_config['target_column']
    lemmatize_text = data_config['lemmatize']

    print(f"DEBUG: preprocess.py - Loading raw data from {raw_data_path}...")
    df = load_data(raw_data_path)
    print(f"DEBUG: preprocess.py - Raw data loaded. Shape: {df.shape}")

    df = df.rename(columns={target_column: 'label'})
    df['label'] = df['label'].apply(lambda x: 1 if x.lower() == 'suicide' else 0)
    print("DEBUG: preprocess.py - Renamed target column and converted labels.")

    print(f"DEBUG: preprocess.py - Cleaning text data (lemmatize={lemmatize_text})...")
    df['cleaned_text'] = df[text_column].apply(lambda x: clean_text(x, lemmatize=lemmatize_text))
    print("DEBUG: preprocess.py - Text cleaning complete.")

    processed_df = df[['cleaned_text', 'label']].copy()
    save_data(processed_df, processed_data_path)
    print(f"DEBUG: preprocess.py - Processed data saved to {processed_data_path}.")
    print("DEBUG: preprocess.py - Exiting preprocess_data function.")
