# Originally from Nihal's part
import json
import re

import numpy as np
import pandas as pd

from retriever_generic import RetrieveProcess, RetrieverQueryState
from utils import doc_df_to_docs, load_doc_df, tokenize

GENERIC_NAME_WORDS = {'penyakit', 'sindrom', 'gangguan', 'infeksi', 'kanker', 'tumor', 'radang', 'pada', 'anak', 'bayi'}

# a phrase must contain at least one of these to count as a symptom
SYMPTOM_WORDS = re.compile(
    r'\b(nyeri|sakit|demam|batuk|pilek|mual|muntah|pusing|lemas|lelah|letih|ruam|gatal|bengkak|pembengkakan|'
    r'sesak|diare|sembelit|konstipasi|kemerahan|merah|perdarahan|berdarah|mimisan|kejang|kesemutan|mati rasa|'
    r'kebas|kabur|buram|sulit|susah|kesulitan|benjolan|benjol|luka|lepuh|lepuhan|bintik|bercak|bisul|nanah|'
    r'menggigil|berkeringat|keringat|berat badan|nafsu makan|kram|bersin|hidung tersumbat|tersumbat|gelisah|'
    r'cemas|bingung|linglung|pingsan|berdebar|keputihan|kering|panas|perih|kaku|lemah|kelemahan|bengkok|'
    r'kuning|pucat|kebiruan|gangguan|kehilangan|hilang|sering buang air|buang air|dahak|lendir|serak|'
    r'mengi|sendawa|kembung|begah|tidak nyaman|ketidaknyamanan|terbakar|berdenyut|gemetar|tremor|'
    r'lumpuh|kelumpuhan|bau|penurunan|peningkatan|insomnia|sulit tidur|mengantuk|halusinasi)',
    re.IGNORECASE)

TRIGGER = re.compile(r'.*?\b(ditandai dengan|ditandai oleh|seperti|meliputi|berupa|antara lain|yaitu|yakni|'
                     r'mengalami|menimbulkan|menyebabkan|di antaranya|termasuk)\b', re.IGNORECASE)
SPLITTER = re.compile(r'[,;:.()]|\b(?:dan|serta|atau|maupun|hingga|sampai|disertai|diikuti)\b')

# filler words removed from the start / end of a phrase
LEAD = re.compile(r'^(rasa|terasa|merasa|merasakan|dirasakan|mengalami|timbul|timbulnya|muncul|munculnya|'
                  r'kemunculan|adanya|terjadinya|terjadi|berupa|yaitu|terutama|sering|kadang|kadang-kadang|'
                  r'biasanya|juga|bisa|dapat|akan|menjadi|mulai|sangat|cukup|lebih|yang|di|pada|ke|dengan|'
                  r'penderita|penderitanya|anak|bayi|seseorang|gejala|gejalanya|keluhan)\s+', re.IGNORECASE)
TRAIL = re.compile(r'\s+(yang|dan|atau|serta|pada|di|ke|dengan|tertentu|tersebut|ini|itu|juga|saja|lain|'
                   r'secara|sebagainya|lainnya)$', re.IGNORECASE)

# phrases with these words are advice, treatment, hedging or context, not symptoms
REJECT = re.compile(
    r'\b(jangan|berikan|memberikan|obat|dokter|segera|sebaiknya|perlu|harus|disarankan|pengobatan|'
    r'penyakit|kondisi|penderita|pasien|disebabkan|akibat|risiko|komplikasi|tersebut|umumnya|'
    r'rumah sakit|terapi|fisioterapi|operasi|mengurangi|meredakan|meringankan|menghilangkan|mengatasi|'
    r'mencegah|diobati|ditangani|meski|meskipun|walaupun|namun|tanpa gejala|tanpa disertai|'
    r'tidak menimbulkan|tidak terasa|tidak selalu|tidak ada|jarang|beberapa|sebagian|kasus|'
    r'misalnya|contohnya|salah satunya)\b', re.IGNORECASE)

def name_tokens(name):
    # words of the disease name that must not appear in a query (symptom words like 'demam' are allowed)
    return {t for t in re.findall(r'\w+', name.lower())
            if len(t) > 2 and t not in GENERIC_NAME_WORDS and not SYMPTOM_WORDS.match(t)}

def strip_edges(piece):
    prev = None
    while prev != piece:
        prev = piece
        piece = LEAD.sub('', piece).strip()
        piece = TRAIL.sub('', piece).strip()
    return piece

def near_duplicate(a, b):
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(1, len(ta | tb)) >= 0.8

def extract_phrases(symptom_text, disease_name='', max_words=6):
    banned = name_tokens(disease_name)
    phrases = []
    for sent in re.split(r'(?<=[.!?])\s+', symptom_text):
        sent = TRIGGER.sub('', sent, count=1)            # drop openings like 'Impetigo ditandai dengan'
        for piece in SPLITTER.split(sent):
            piece = re.sub(r'\s+', ' ', piece or '').strip(' -"\'').lower()
            piece = strip_edges(piece)
            words = piece.split()
            if not (1 <= len(words) <= max_words):
                continue
            if banned & set(words) or REJECT.search(piece):
                continue
            if not SYMPTOM_WORDS.search(piece):
                continue
            if any(near_duplicate(piece, p) for p in phrases):
                continue
            phrases.append(piece)
    return phrases

def present_symptoms(state):
    return [s['name'] for s in state.get('symptoms', []) if s.get('present')]

def absent_symptoms(state):
    return [s['name'] for s in state.get('symptoms', []) if not s.get('present')]

def query_entities(case: RetrieverQueryState):
    return present_symptoms(case.to_dict()), absent_symptoms(case.to_dict())

def to_token_sets(phrases):
    return [s for s in (frozenset(tokenize(p)) for p in phrases) if s]

MATCH_THRESHOLD = 0.6   # token overlap needed for two symptom phrases to count as the same
def hits(query_sets, doc_sets):
    return sum(any(len(q & d) / min(len(q), len(d)) >= MATCH_THRESHOLD for d in doc_sets) for q in query_sets)

class EntityBasedRetrieveProcess(RetrieveProcess):
    def __init__(self, doc_df: pd.DataFrame, entity_path: str):
        E = {k.lower(): v for k, v in json.load(open(entity_path, encoding='utf-8')).items()}
        load_entities = lambda i: E.get(doc_df.loc[i, 'name'].lower(), {}).get('SYMPTOM', [])

        self.docs = doc_df_to_docs(doc_df)
        self.doc_ent = [to_token_sets(load_entities(entity_path)) for i in range(len(doc_df))]
        self.absent_penalty = 0.5

    def __call__(self, query_state_list: list[RetrieverQueryState]):
        S = np.zeros((len(query_state_list), len(self.docs)))
        for qi, case_ in enumerate(query_state_list):
            present, absent = query_entities(case_)
            P, A = to_token_sets(present), to_token_sets(absent)
            for j, D in enumerate(self.doc_ent):
                if D and P:
                    S[qi, j] = hits(P, D) / len(P) - self.absent_penalty * (hits(A, D) / len(A) if A else 0)
        return S
