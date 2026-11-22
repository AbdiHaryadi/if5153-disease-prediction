import numpy as np
import pandas as pd

from retriever_bm25 import BM25RetrieveProcess
from retriever_embed import EmbeddingBasedRetrieveProcess, EmbeddingModelConfig
from retriever_entity import EntityBasedRetrieveProcess
from retriever_fusion import FusionRetrieveProcess
from retriever_generic import RetrieverQuery, RetrieverSymptomState

RetrieverSymptomInput = RetrieverSymptomState

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

        self.doc_df = doc_df

    def retrieve_diseases(self, symptoms: list[RetrieverSymptomState]) -> list[RetrieverDisease]:
        # Originally from Nihal
        query_list = [RetrieverQuery.from_symptom_states(symptoms)]
        S = self.process(query_list)
        top3 = np.argsort(-S, axis=1)[:, :3]

        df = self.doc_df
        top3 = [str(df.loc[j, 'name']) for j in top3[0]]
        diseases = [RetrieverDisease(str(df.loc[j, 'name'])) for j in top3[0]]
        return diseases
