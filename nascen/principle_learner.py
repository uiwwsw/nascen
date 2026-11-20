"""Use the actual teacher prose as supervision for shared student weights."""

import torch
from torch.nn.functional import cross_entropy, normalize

from .concept_model import TEMPERATURE, ConceptModel, continuation_loss
from .sentence_encoder import SentenceVectors
from .semantic_dataset import Scene
from .semantic_teacher import PrincipleLesson


def principle_texts(lessons: dict[str, PrincipleLesson]):
    for lesson in lessons.values():
        yield lesson.when
        yield lesson.then
        yield lesson.otherwise_when
        yield lesson.otherwise_then
        yield lesson.explanation


class PrincipleTargets:
    def __init__(self, lessons: dict[str, PrincipleLesson], scenes: tuple[Scene, ...], vectors: SentenceVectors):
        self.explanations = torch.stack([vectors[lesson.explanation] for lesson in lessons.values()])
        rule_index = {key: i for i, key in enumerate(lessons)}
        self.scene_rules = torch.tensor([rule_index.get(scene.principle, -1) for scene in scenes])
        self.conditions = torch.stack([
            vectors[text] for lesson in lessons.values() for text in (lesson.when, lesson.otherwise_when)
        ]).unsqueeze(1)
        self.consequences = torch.stack([
            torch.stack((vectors[lesson.then], vectors[lesson.otherwise_then]))
            for lesson in lessons.values() for _ in range(2)
        ])
        self.branches = torch.tensor([0, 1] * len(lessons))

    def loss(self, student: ConceptModel, contexts: torch.Tensor) -> torch.Tensor:
        known = self.scene_rules >= 0
        representations = normalize(student.context(contexts[known]), dim=-1)
        explanations = normalize(student.project(self.explanations), dim=-1)
        alignment = cross_entropy(representations @ explanations.T / TEMPERATURE, self.scene_rules[known])
        application = continuation_loss(student(self.conditions), self.consequences, self.branches)
        return alignment + application
