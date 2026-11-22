from ner import NER
from retriever import Retriever, RetrieverDisease, RetrieverSymptomInput

class Response:
    def handle(self) -> Response | None:
        raise NotImplementedError

class Session:
    def __init__(
            self,
            possible_diseases: list[RetrieverDisease],
            symptom_presences: dict[str, bool | None],
    ):
        self.possible_diseases = possible_diseases
        self.symptom_presences = symptom_presences

    def make_response(self) -> Response:
        for symptom, presence in self.symptom_presences.items():
            if presence is None:
                return AskSymptomOccurence(self, symptom)

        return DiseasePrediction(self.possible_diseases)

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
    def __init__(self):
        self.ner = NER()
        self.retriever = Retriever()

    def tell(self, complaint: str) -> Response:
        symptoms: list[str] = self.ner.extract_symptom_entities(complaint)
        retriever_symptoms = [
            RetrieverSymptomInput(name=s, present=True)
            for s in symptoms
        ]
        retriever_diseases = self.retriever.retrieve_diseases(
            retriever_symptoms
        )
        symptom_presences: dict[str, bool | None] = {
            rs.name: rs.present for rs in retriever_symptoms
        }
        for d in retriever_diseases:
            for s in d.get_symptoms():
                if s not in symptom_presences:
                    symptom_presences[s] = None
        
        return Session(
            possible_diseases=retriever_diseases,
            symptom_presences=symptom_presences
        ).make_response()

if __name__ == "__main__":
    app = App()
    complaint = input("Masukkan komplain: ")
    response = app.tell(complaint)
    while response is not None:
        response = response.handle()
