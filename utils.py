import json
import re

import pandas as pd

STOPWORDS = set(('yang dan di ke dari pada untuk dengan atau ini itu adalah juga dapat bisa akan oleh dalam '
                 'sebagai karena seperti serta tidak ada pasien mengalami merasakan keluhan penderita').split())

def tokenize(text):
    # Originally from Nihal's part
    return [t for t in re.findall(r'\w+', text.lower()) if t not in STOPWORDS and len(t) > 1]

def load_doc_df(path: str, corpus = 'all'):
    # Originally from Nihal's part
    df = pd.read_csv(path).fillna('')
    df['has_symptoms'] = df['has_symptoms'].astype(str).str.lower().eq('true')
    if corpus == 'with_symptoms':
        df = df[df['has_symptoms']]
    df = df.reset_index(drop=True)

    def build_doc_text(row):
        parts = [row['name'] + '.', row['description'], row['symptoms']]
        return ' '.join(p for p in parts if p).strip()

    df['doc_text'] = df.apply(build_doc_text, axis=1)
    print(len(df), 'diseases in the corpus,', int(df['has_symptoms'].sum()), 'with symptom text')
    print(df[['name', 'doc_text']].head())
    return df

def doc_df_to_docs(df: pd.DataFrame):
    return df['doc_text'].tolist()

def load_entities_dict(path: str):
    # Originally from Nihal's part
    return {k.lower(): v for k, v in json.load(open(path, encoding='utf-8')).items()}

def get_symptoms_from_entities_dict(entities: dict, disease_name: str) -> list[str]:
    return entities.get(disease_name.lower(), {}).get('SYMPTOM', [])
