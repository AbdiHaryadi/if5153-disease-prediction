import numpy as np
import pandas as pd

from retriever_bm25 import BM25RetrieveProcess
from retriever_embed import EmbeddingBasedRetrieveProcess, EmbeddingModelConfig
from retriever_entity import EntityBasedRetrieveProcess
from retriever_fusion import FusionRetrieveProcess
from retriever_generic import RetrieverQuery, RetrieverSymptomState
from utils import get_symptoms_from_entities_dict

RetrieverSymptomInput = RetrieverSymptomState

class RetrieverDisease:
    def __init__(self, name: str, symptoms: list[str]):
        self.name = name
        self.symptoms = symptoms

    def get_symptoms(self) -> list[str]:
        return self.symptoms

class Retriever:
    def __init__(self, doc_df: pd.DataFrame, entities: dict):
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
                EntityBasedRetrieveProcess(doc_df, entities)
            ],
            weights=[0.3, 0.4, 0.3]
        )

        self.doc_df = doc_df
        self.entities = entities

    def retrieve_diseases_from_symptoms(self, symptoms: list[RetrieverSymptomState]) -> list[RetrieverDisease]:
        query_list = [RetrieverQuery.from_symptom_states(symptoms)]
        return self._retrieve_diseases(query_list)

    def _retrieve_diseases(self, query_list: list[RetrieverQuery]) -> list[RetrieverDisease]:
        # Originally from Nihal
        S = self.process(query_list)
        top3 = np.argsort(-S, axis=1)[:, :3]

        df = self.doc_df
        top3 = [str(df.loc[j, 'name']) for j in top3[0]]
        diseases = [RetrieverDisease(name, self.get_symptoms(name)) for name in top3]
        return diseases

    def get_symptoms(self, disease_name: str):
        return get_symptoms_from_entities_dict(self.entities, disease_name)

    def retrieve_diseases_from_text(self, text: str) -> list[RetrieverDisease]:
        query_list = [RetrieverQuery.from_text(text)]
        return self._retrieve_diseases(query_list)
