"""Evidence-backed hypotheses and a separate decision about what to train."""

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import sqrt

from .world import SPRITE_SIZE, factored_pixels

MIN_SUPPORT = 2
MIN_LEARNING_SCORE = 0.32


@dataclass(frozen=True)
class Verification:
    status: str
    accepted: str | None
    confidence: float
    votes: dict[str, int]


@dataclass(frozen=True)
class LearningValue:
    novelty: float
    value: float
    score: float


class Knowledge:
    def __init__(self, state: dict | None = None):
        state = state or {}
        self.evidence: dict[str, dict[str, str]] = state.get("evidence", {})
        self.training: dict[str, dict[str, int]] = state.get("training", {})

    def observe(self, claim: str) -> None:
        self.evidence.setdefault(claim, {})

    def record(self, claim: str, origin: str, outcome: str) -> Verification:
        self.observe(claim)
        self.evidence[claim][origin] = outcome
        return self.verify(claim)

    def verify(self, claim: str) -> Verification:
        votes = Counter(self.evidence.get(claim, {}).values())
        ranked = votes.most_common(2)
        if not ranked:
            return Verification("pending", None, 0.0, {})
        candidate, support = ranked[0]
        rival = ranked[1][1] if len(ranked) > 1 else 0
        confidence = (support + 1) / (sum(votes.values()) + 2)
        if support >= MIN_SUPPORT and support > rival:
            return Verification("verified", candidate, confidence, dict(votes))
        status = "conflicted" if rival else "pending"
        return Verification(status, None, confidence, dict(votes))

    def check(self, claim: str, outcome: str) -> tuple[str, Verification]:
        verification = self.verify(claim)
        if verification.accepted is not None and verification.accepted != outcome:
            return "refuted", verification
        return verification.status, verification

    def mark_trained(self, claim: str, outcome: str) -> None:
        counts = self.training.setdefault(claim, {})
        counts[outcome] = counts.get(outcome, 0) + 1

    def trained(self, claim: str, outcome: str) -> int:
        return self.training.get(claim, {}).get(outcome, 0)

    def checkpoint(self) -> dict:
        return {"evidence": self.evidence, "training": self.training}

    def counts(self) -> dict[str, int]:
        totals = dict.fromkeys(("pending", "conflicted", "verified"), 0)
        for claim in self.evidence:
            totals[self.verify(claim).status] += 1
        totals["consolidations"] = sum(sum(values.values()) for values in self.training.values())
        return totals

    def findings(self) -> list[dict]:
        return [
            {
                "claim": claim, **asdict(self.verify(claim)),
                "evidence": dict(ballots), "consolidations": dict(self.training.get(claim, {})),
            }
            for claim, ballots in sorted(self.evidence.items())
        ]

    def assess(self, claim: str, outcome: str, confidence: Callable[[], float]) -> dict:
        status, verification = self.check(claim, outcome)
        decision = {"status": status, "reliability": verification.confidence, "votes": verification.votes}
        if status == "verified":
            value = learning_value(confidence(), self.trained(claim, outcome))
            decision.update(asdict(value))
        return decision


def learning_value(confidence: float, training_count: int) -> LearningValue:
    novelty = 1 / sqrt(1 + training_count)
    value = 1 - confidence
    return LearningValue(novelty, value, 0.35 * novelty + 0.65 * value)


def appearance_claim(pixels: tuple[float, ...]) -> str:
    """Group similar observations using RGB and a translation-normalized silhouette."""
    red, green, blue, *mask = factored_pixels(pixels)
    occupied = [(index % SPRITE_SIZE, index // SPRITE_SIZE) for index, value in enumerate(mask) if value]
    origin_x = min((x for x, _ in occupied), default=0)
    origin_y = min((y for _, y in occupied), default=0)
    silhouette = sum(1 << ((y - origin_y) * SPRITE_SIZE + x - origin_x) for x, y in occupied)
    color_bins = "-".join(str(round(component * 6)) for component in (red, green, blue))
    return f"appearance:{color_bins}:{silhouette:x}"


def label_outcome(label: tuple[int, int] | list[int]) -> str:
    return f"{label[0]}:{label[1]}"
