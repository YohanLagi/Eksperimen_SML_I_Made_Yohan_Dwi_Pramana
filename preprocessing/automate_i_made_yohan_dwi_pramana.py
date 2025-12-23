import os
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

STOP_ID = set(stopwords.words('indonesian'))
STOP_EN = set(stopwords.words('english'))
STOP_CUSTOM = {'iya', 'yaa', 'loh', 'sih', 'nya', 'ga', 'ya'}
STOPWORDS_ALL = STOP_ID | STOP_EN | STOP_CUSTOM

def load_slangwords(csv_path):
    slang_df = pd.read_csv(csv_path)

    # Validasi kolom wajib
    required_cols = {'@', 'di'}
    if not required_cols.issubset(slang_df.columns):
        raise ValueError(f"Kolom slangword tidak sesuai. Ditemukan: {slang_df.columns}")

    slang_df = slang_df[['@', 'di']].dropna()
    slang_df['@'] = slang_df['@'].astype(str)
    slang_df['di'] = slang_df['di'].astype(str)

    return dict(zip(slang_df['@'], slang_df['di']))


SLANGWORDS = load_slangwords("preprocessing/indonesian-slangwords.csv")


def cleaning_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'RT\s+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def casefolding_text(text):
    return text.lower()


def normalize_slang(text):
    return ' '.join(SLANGWORDS.get(w, w) for w in text.split())


def filtering_text(words):
    return [w for w in words if w not in STOPWORDS_ALL]


def preprocess_text(text):
    text = cleaning_text(str(text))
    text = casefolding_text(text)
    text = normalize_slang(text)
    tokens = text.split()
    tokens = filtering_text(tokens)
    return ' '.join(tokens)


def preprocess_dataframe(df, text_column='content'):
    if text_column not in df.columns:
        raise ValueError(f"Kolom '{text_column}' tidak ditemukan")

    df = df.copy()
    df['text_final'] = df[text_column].apply(preprocess_text)
    return df


def preprocess_csv(input_path, output_path): 
if not os.path.exists(input_path): 
    raise FileNotFoundError("File input tidak ditemukan") 
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    df = pd.read_csv(input_path) 
    df_clean = preprocess_dataframe(df) 
    df_clean.to_csv(output_path, index=False) 
    return df_clean 
    
    if __name__ == "__main__": 
        preprocess_csv( 
            input_path='Kredivo.csv', 
            output_path='preprocessing/preprocessed_kredivo.csv' )
