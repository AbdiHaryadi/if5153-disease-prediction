import os

import pandas as pd

from ner import NER
from retriever import Retriever, RetrieverDisease, RetrieverSymptomInput
from utils import load_doc_df

class Response:
    def handle(self) -> Response | None:
        raise NotImplementedError

class Session:
    def __init__(
            self,
            retriever: Retriever,
            symptom_presences: dict[str, bool],
    ):
        self.retriever = retriever
        self.symptom_presences = symptom_presences

    def make_response(self) -> Response:
        retriever_symptoms = [
            RetrieverSymptomInput(name=name, present=present)
            for name, present in self.symptom_presences.items()
        ]
        retriever_diseases = self.retriever.retrieve_diseases(
            retriever_symptoms
        )
        for d in retriever_diseases:
            for s in d.get_symptoms():
                if s not in self.symptom_presences:
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
    def __init__(self, doc_df: pd.DataFrame, entity_path: str):
        self.ner = NER()
        self.retriever = Retriever(doc_df, entity_path)

    def tell(self, complaint: str) -> Response:
        symptoms: list[str] = self.ner.extract_symptom_entities(complaint)
        return Session(
            retriever=self.retriever,
            symptom_presences={
                s: True for s in symptoms
            }
        ).make_response()

if __name__ == "__main__":
    doc_df = load_doc_df(os.environ["DATA_PATH"])
    entity_path = os.environ["ENTITY_PATH"]
    app = App(doc_df, entity_path)
    complaint = input("Masukkan komplain: ")
    response = app.tell(complaint)
    while response is not None:
        response = response.handle()
