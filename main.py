import os

import pandas as pd

from ner import NER
from retriever import Retriever, RetrieverDisease, RetrieverSymptomInput
from utils import load_doc_df, load_entities_dict

class Response:
    def handle(self) -> Response | None:
        raise NotImplementedError

class Session:
    def __init__(
            self,
            retriever: Retriever,
            symptom_presences: dict[str, bool | None],
    ):
        self.retriever = retriever
        self.symptom_presences = symptom_presences

    def make_response(self) -> Response:
        for s, presence in self.symptom_presences.items():
            if presence is None:
                return AskSymptomOccurence(self, s)

        retriever_symptoms = [
            RetrieverSymptomInput(name=name, present=present)
            for name, present in self.symptom_presences.items()
            if present is not None
        ]
        retriever_diseases = self.retriever.retrieve_diseases_from_symptoms(
            retriever_symptoms
        )
        for d in retriever_diseases:
            for s in d.get_symptoms():
                if s not in self.symptom_presences:
                    self.symptom_presences[s] = None

        for s, presence in self.symptom_presences.items():
            if presence is None:
                return AskSymptomOccurence(self, s)

        return DiseasePrediction(retriever_diseases)

class AskSymptomOccurence(Response):
    def __init__(self, session: Session, asked_symptom: str):
        self.session = session
        self.asked_symptom = asked_symptom

    def handle(self) -> Response | None:
        occured_input = ""
        while occured_input != "y" and occured_input != "t":
            occured_input = input(f"{self.asked_symptom}? (y/t)")
        occured = occured_input == "y"

        self.session.symptom_presences[self.asked_symptom] = occured
        return self.session.make_response()

class DiseasePrediction(Response):
    def __init__(self, diseases: list[RetrieverDisease]):
        self.diseases = diseases

    def handle(self):
        print(f"Prediksi: ({len(self.diseases)})")
        for d in self.diseases:
            print(f"- {d.name}")

class App:
    def __init__(self, doc_df: pd.DataFrame, entities: dict, use_ner: bool = True):
        self.retriever = Retriever(doc_df, entities)
        self.ner = NER() if use_ner else None

    def tell(self, complaint: str) -> Response:
        symptom_presences: dict[str, bool | None] = {}
        if self.ner is None:
            symptom_presences = self.init_symptom_presences_with_retriever(complaint)
        else:
            symptom_presences = self.init_symptom_presences_with_ner(complaint)
        return Session(self.retriever, symptom_presences).make_response()

    def init_symptom_presences_with_retriever(self, complaint: str):
        retriever_diseases = self.retriever.retrieve_diseases_from_text(
            complaint
        )
        symptom_presences: dict[str, bool | None] = {}
        for d in retriever_diseases:
            for s in d.get_symptoms():
                if s not in symptom_presences:
                    symptom_presences[s] = None
        return symptom_presences

    def init_symptom_presences_with_ner(self, complaint: str):
        assert self.ner is not None
        symptoms: list[str] = self.ner.extract_symptom_entities(complaint)
        symptom_presences: dict[str, bool | None] = {}
        for s in symptoms:
            symptom_presences[s] = True
        return symptom_presences

if __name__ == "__main__":
    doc_df = load_doc_df(os.environ["DATA_PATH"])
    entities = load_entities_dict(os.environ["ENTITY_PATH"])
    app = App(doc_df, entities, use_ner=False)
    complaint = input("Masukkan komplain: ")
    response = app.tell(complaint)
    while response is not None:
        response = response.handle()
