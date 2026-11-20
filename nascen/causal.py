"""Discover small probabilistic programs from measured interventions."""

from collections.abc import Sequence
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from math import exp, log

from .semantic import PUSH_FIELDS, Offset, PushEvent, PushObservation

# Loss weights are fixed before collecting outcomes in Experiment 001.
PREDICATE_COST = 1.25
CONTRADICTION_COST = 0.35
PLANNER_PROGRAMS = 8
Predicate = tuple[int, str]


@dataclass
class Program:
    predicates: tuple[Predicate, ...]
    counts: tuple[Counter[Offset], Counter[Offset]] = field(default_factory=lambda: (Counter(), Counter()))

    def branch(self, features: tuple[str, ...]) -> int:
        return int(all(features[index] == value for index, value in self.predicates))

    def distribution(self, features: tuple[str, ...], offsets: Sequence[Offset]) -> dict[Offset, float]:
        observed = self.counts[self.branch(features)]
        normalizer = observed.total() + len(offsets)
        return {offset: (observed[offset] + 1) / normalizer for offset in offsets}

    def objective(self, offset_count: int) -> float:
        loss = PREDICATE_COST * len(self.predicates)
        for observed in self.counts:
            total = observed.total()
            if total:
                loss -= sum(count * log((count + 1) / (total + offset_count)) for count in observed.values())
                loss += CONTRADICTION_COST * (total - max(observed.values()))
        return loss

    def describe(self, offsets: Sequence[Offset]) -> dict:
        rules = []
        for branch, observed in enumerate(self.counts):
            if observed:
                total = observed.total()
                rules.append({
                    "when": "conditions_match" if branch else "otherwise",
                    "effects": [
                        {"offset": offset, "evidence": observed[offset],
                         "probability": (observed[offset] + 1) / (total + len(offsets))}
                        for offset in offsets
                    ],
                    "counterexamples": total - max(observed.values()),
                })
        return {
            "conditions": [{"field": PUSH_FIELDS[index], "equals": value} for index, value in self.predicates],
            "branches": rules, "objective": self.objective(len(offsets)),
        }


class CausalKernel:
    def __init__(self, observations: Sequence[PushObservation]):
        # Enumerating predicates uses visible states only. All outcome counts start at zero.
        predicates = {()}
        for observation in observations:
            terms = tuple(enumerate(observation.features))
            predicates.update((term,) for term in terms)
            predicates.update(combinations(terms, 2))
        self.programs = [Program(terms) for terms in sorted(predicates, key=lambda terms: (len(terms), terms))]
        self.ranked = [(0.0, program) for program in self.programs]
        self.exposures: dict[tuple[int, str], int] = {}
        self.offsets: list[Offset] = [(0, 0)]
        self.events: list[PushEvent] = []
        self.planner_predictions = 0

    @property
    def program(self) -> Program:
        return self.ranked[0][1]

    def observe(self, event: PushEvent) -> None:
        features = event.before.features
        offset = event.offset
        if offset not in self.offsets:
            self.offsets.append(offset)
        self.events.append(event)
        for term in enumerate(features):
            self.exposures[term] = self.exposures.get(term, 0) + 1
        for program in self.programs:
            program.counts[program.branch(features)][offset] += 1
        self.ranked = sorted(
            ((program.objective(len(self.offsets)), program) for program in self.programs),
            key=lambda scored: scored[0],
        )

    def predict(self, observation: PushObservation) -> dict[Offset, float]:
        return self.program.distribution(observation.features, self.offsets)

    def select(self, options: dict[int, PushObservation], tie_break: dict[int, float]) -> int:
        head = self.ranked[:PLANNER_PROGRAMS]
        weights = [exp(head[0][0] - score) for score, _ in head]
        total = sum(weights)
        self.planner_predictions += len(head) * len(options)

        def value(item: tuple[int, PushObservation]) -> tuple[float, float]:
            key, observation = item
            features = observation.features
            predictions = [program.distribution(features, self.offsets).values() for _, program in head]
            disagreement = 0.0
            for probabilities in zip(*predictions, strict=True):
                weighted = tuple(zip(weights, probabilities, strict=True))
                average = sum(weight * probability for weight, probability in weighted) / total
                disagreement += sum(weight * (p - average) ** 2 for weight, p in weighted) / total
            novelty = sum(1 / (1 + self.exposures.get(term, 0)) for term in enumerate(features)) / len(features)
            return disagreement / len(self.offsets) + 0.12 * novelty, tie_break[key]

        return max(options.items(), key=value)[0]
