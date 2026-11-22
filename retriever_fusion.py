# Originally from Nihal's part
from typing import Any

import numpy as np

from retriever_generic import RetrieveProcess, RetrieverQueryState


def minmax(S):
    lo, hi = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
    return (S - lo) / np.where(hi - lo == 0, 1, hi - lo)

class FusionRetrieveProcess(RetrieveProcess):
    def __init__(
            self,
            processes: list[RetrieveProcess],
            weights: list[float],
    ):
        self.processes = processes
        self.weights = weights

    def __call__(self, query_state_list: list[RetrieverQueryState]) -> Any:
        all_scores = [p(query_state_list) for p in self.processes]
        normalized_scores = [minmax(scores) for scores in all_scores]
        return sum(w * scores for scores, w in zip(normalized_scores,
                                                   self.weights, strict=True))
