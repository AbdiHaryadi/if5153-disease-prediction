from dataclasses import dataclass
from typing import Any

@dataclass
class RetrieverSymptomState:
    name: str
    present: bool

    def to_dict(self):
        return {
            "name": self.name,
            "present": self.present,
        }

TEMPLATES = ['Pasien mengalami {}.', 'Keluhan pasien adalah {}.', 'Pasien merasakan {}.']

def describe(symptom):
    text = symptom['name']
    attrs = symptom.get('attributes') or {}
    if 'type' in attrs:
        text += ' ' + str(attrs['type'])
    if 'duration_days' in attrs:
        days = attrs['duration_days']
        text += f' selama {days} hari'
    return text

def join_id(items):
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' dan ' + items[-1]

def serialize(state, template=0):
    present = [describe(s) for s in state.get('symptoms', []) if s.get('present')]
    if not present:
        return ''
    return TEMPLATES[template % len(TEMPLATES)].format(join_id(present))

@dataclass
class RetrieverQueryState:
    symptoms: list[RetrieverSymptomState]

    def to_text(self):
        return serialize(self.to_dict())

    def to_dict(self):
        return {"symptoms": [s.to_dict() for s in self.symptoms]}

class RetrieveProcess:
    def __call__(self, query_state_list: list[RetrieverQueryState]) -> Any:
        raise NotImplementedError
