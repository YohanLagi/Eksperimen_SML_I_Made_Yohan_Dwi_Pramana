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
    if not os.path.exists(csv_path):
        raise FileNotFoundError("File slangword CSV tidak ditemukan")

    df = pd.read_csv(csv_path)
    return dict(zip(df['slang'], df['formal']))


SLANGWORDS = load_slangwords("preprocessing/indonesian_slangwords.csv")


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


def preprocess_folder(input_dir, output_dir, text_column='content'):
    if not os.path.exists(input_dir):
        raise FileNotFoundError("Folder input tidak ditemukan")

    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_files:
        raise ValueError("Tidak ada file CSV di folder input")

    for file in csv_files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(
            output_dir,
            file.replace('.csv', '_preprocessed.csv')
        )

        df = pd.read_csv(input_path)
        df_clean = preprocess_dataframe(df, text_column)
        df_clean.to_csv(output_path, index=False)

        print(f"[OK] {file} → {output_path}")

if __name__ == "__main__":
    preprocess_folder(
        input_dir="Kredivo",
        output_dir="preprocessed_kredivo",
        text_column="content"
    )
