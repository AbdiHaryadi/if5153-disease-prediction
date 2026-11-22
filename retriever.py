from dataclasses import dataclass

import pandas as pd

from retriever_bm25 import BM25RetrieveProcess
from retriever_embed import EmbeddingBasedRetrieveProcess, EmbeddingModelConfig
from retriever_entity import EntityBasedRetrieveProcess
from retriever_fusion import FusionRetrieveProcess

@dataclass
class RetrieverSymptomInput:
    name: str
    present: bool

class RetrieverDisease:
    def __init__(self, name: str):
        self.name = name

    def get_symptoms(self) -> list[str]:
        return []

class Retriever:
    def __init__(self, doc_df: pd.DataFrame, entity_path: str):
        # Used process: Fusion
        # Use the best result from Nihal
        self.process = FusionRetrieveProcess(
            processes=[
                BM25RetrieveProcess(doc_df),
                EmbeddingBasedRetrieveProcess(
                    doc_df,
                    config=EmbeddingModelConfig(
                        name="intfloat/multilingual-e5-base",
                        translate=False,
                        q_prefix="query: ",
                        d_prefix="passage: "
                    )
                ),
                EntityBasedRetrieveProcess(doc_df, entity_path)
            ],
            weights=[0.3, 0.4, 0.3]
        )

    def retrieve_diseases(self, symptoms: list[RetrieverSymptomInput]) -> list[RetrieverDisease]:
        raise NotImplementedError("Not done yet")
        # scores = self.process(symptoms)
        S = SCORES[split][method]
        top3 = np.argsort(-S, axis=1)[:, :3]
        rows = [{'query': q['text'], 'correct': df.loc[g, 'name'], 'top3': [df.loc[j, 'name'] for j in t]}
                for q, t, g in zip(splits[split], top3, GOLD[split]) if t[0] != g]
        print(len(rows), 'of', len(S), 'queries wrong at top-1 for', method)
        return pd.DataFrame(rows).head(n)
        return []

# class FusionRetrieveProcess:
#     def __init__(
#             self,
#             doc_df: pd.DataFrame,
#             embedding_model_config: EmbeddingModelConfig,
#             entity_path: str,
#             weights,
#     ):
#         self.bm25_process = BM25RetrieveProcess(doc_df)
#         self.embed_process = EmbeddingBasedRetrieveProcess(doc_df, embedding_model_config)
#         self.entity_process = EntityBasedRetrieveProcess(doc_df, entity_path)
#         self.weights = weights

#     def __call__(self, query_texts: list[str]) -> float:
#         all_scores = [
#             self.bm25_process(query_texts),
#             self.embed_process(query_texts),
#             self.entity_process(query_texts),
#         ]
#         normalized_scores = [minmax(scores) for scores in all_scores]
#         return sum(w * scores for scores, w in zip(normalized_scores, self.weights))
