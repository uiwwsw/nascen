"""A compact, ordered sentence-space student."""

import torch
from torch import nn
from torch.nn.functional import cosine_similarity, cross_entropy, normalize

from .semantic_dataset import RELATIONS

TEMPERATURE = 0.12


class ConceptModel(nn.Module):
    def __init__(self, width: int, seed: int, hidden: int = 48):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.project = nn.Sequential(nn.Linear(width, hidden), nn.Tanh())
            self.sequence = nn.GRU(hidden, hidden, batch_first=True)
            self.next_sentence = nn.Linear(hidden, width)
            self.relation = nn.Linear(4 * hidden, len(RELATIONS))

    def context(self, sentences: torch.Tensor) -> torch.Tensor:
        _, state = self.sequence(self.project(sentences))
        return state.squeeze(0)

    def forward(self, sentences: torch.Tensor) -> torch.Tensor:
        return normalize(self.next_sentence(self.context(sentences)), dim=-1)

    def relation_logits(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = self.project(first), self.project(second)
        return self.relation(torch.cat((left, right, left - right, left * right), dim=-1)), cosine_similarity(left, right)


def continuation_scores(prediction: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
    return cosine_similarity(prediction.unsqueeze(-2), options, dim=-1)


def continuation_loss(prediction: torch.Tensor, options: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    scores = continuation_scores(prediction, options)
    observed = scores.gather(1, answers[:, None]).mean()
    return cross_entropy(scores / TEMPERATURE, answers) + 1 - observed


def relation_loss(student: ConceptModel, first: torch.Tensor, second: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits, similarity = student.relation_logits(first, second)
    equivalent = labels == RELATIONS.index("equivalent")
    contradiction = labels == RELATIONS.index("contradiction")
    anchors = (1 - similarity) * equivalent + similarity.clamp(min=0) * contradiction
    return cross_entropy(logits, labels) + anchors.mean()
