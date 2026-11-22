from dataclasses import dataclass

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
    def __init__(self):
        pass

    def retrieve_diseases(self, symptoms: list[RetrieverSymptomInput]) -> list[RetrieverDisease]:
        return []
