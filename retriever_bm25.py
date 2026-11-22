# Originally from Nihal's part
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from retriever_generic import RetrieveProcess, RetrieverQuery
from utils import doc_df_to_docs, tokenize

class BM25RetrieveProcess(RetrieveProcess):
    def __init__(self, doc_df: pd.DataFrame):
        docs = doc_df_to_docs(doc_df)
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

    def __call__(self, query_list: list[RetrieverQuery]):
        return np.vstack([self.bm25.get_scores(tokenize(q.text)) 
                          for q in query_list])
