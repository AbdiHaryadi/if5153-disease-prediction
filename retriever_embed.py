# Originally from Nihal's part
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from transformers import MarianMTModel, MarianTokenizer

from retriever_generic import RetrieveProcess, RetrieverQueryState
from utils import doc_df_to_docs

@dataclass
class EmbeddingModelConfig:
    name: str
    translate: bool
    q_prefix: str
    d_prefix: str

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class Translator:
    def __init__(self):
        RUN_NAME = 'run_03_clean_phrases'
        OUT_DIR = Path('results/retrieval') / RUN_NAME
        
        self.cache_path = OUT_DIR / 'translation_cache.json'
        self._cache = json.load(open(self.cache_path, encoding='utf-8')) if self.cache_path.exists() else {}
        self._mt: tuple | None = None
        self.translation_model_name = 'Helsinki-NLP/opus-mt-id-en'
        self.device = get_device()

    def __call__(self, sentences, batch_size=16):
        todo = [s for s in dict.fromkeys(sentences) if s not in self._cache]
        if todo:
            if self._mt is None:
                tok = MarianTokenizer.from_pretrained(self.translation_model_name)
                mt_model = MarianMTModel.from_pretrained(self.translation_model_name)
                mt_model.to(self.device)  # type: ignore
                mt_model.eval()
                self._mt = (tok, mt_model)
            
            tok, mt_model = self._mt
            for b in range(0, len(todo), batch_size):
                batch = todo[b:b + batch_size]
                enc = tok(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                with torch.no_grad():
                    out = mt_model.generate(**enc, max_new_tokens=256)
                for src, tgt in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
                    self._cache[src] = tgt
            json.dump(self._cache, open(self.cache_path, 'w', encoding='utf-8'), ensure_ascii=False)
        return [self._cache[s] for s in sentences]

class EmbeddingBasedRetrieveProcess(RetrieveProcess):
    def __init__(self, doc_df: pd.DataFrame, config: EmbeddingModelConfig, batch_size: int = 64):
        docs = doc_df_to_docs(doc_df)
        self.config = config
        self.translator = Translator()
        self.model = SentenceTransformer(config.name, device=get_device())
        self.doc_texts = self.translate_if_needed(docs)
        self.batch_size = batch_size

    def translate_if_needed(self, texts: list[str]):
        if self.config.translate:
            return self.translator(texts)

        return texts

    def query_state_list_to_text_list(self, query_state_list: list[RetrieverQueryState]):
        texts = [q.to_text() for q in query_state_list]
        return self.translate_if_needed(texts)

    def __call__(self, query_state_list: list[RetrieverQueryState]):
        kw = dict(
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        query_texts = self.query_state_list_to_text_list(query_state_list)
        q = self.model.encode([self.config.q_prefix + t for t in query_texts], **kw)  # type: ignore
        d = self.model.encode([self.config.d_prefix + t for t in self.doc_texts], **kw)  # type: ignore
        return q @ d.T   # cosine similarity, since vectors are normalised
