"""Question scoring from observations and previously observed learning gains."""

from collections.abc import Sequence
from random import Random

import torch

from .knowledge import Verification
from .model import Learner

QUESTION_STRATEGIES = ("random", "uncertainty", "progress")


def verification_bonus(verification: Verification, new_evidence: bool) -> float:
    """Prefer independent rechecks of open hypotheses."""
    if not new_evidence:
        return -0.25
    if verification.status == "verified":
        return -0.10
    return 0.22 if verification.votes else 0.0


@torch.no_grad()
def question_scores(
    learner: Learner,
    features: Sequence[list[float]],
    strategy: str,
    gains: Sequence[tuple[list[float], float]],
    rng: Random,
) -> list[float]:
    if not features:
        return []
    if strategy == "random":
        return [rng.random() for _ in features]

    observations = torch.tensor(features)
    probabilities = [head.softmax(dim=-1) for head in learner.concepts(observations)]
    uncertainty = 1 - sum(probability.amax(dim=-1) for probability in probabilities) / len(probabilities)

    if strategy == "progress" and gains:
        examples, improvements = zip(*gains)
        distances = torch.cdist(observations, torch.tensor(examples)) / observations.shape[-1] ** 0.5
        distance, neighbor = distances.topk(min(3, len(examples)), largest=False)
        weights = (distance + 0.05).reciprocal()
        expected_gain = (weights * torch.tensor(improvements)[neighbor]).sum(dim=-1) / weights.sum(dim=-1)
        # Keep a route to unfamiliar observations when previous gains are small or absent.
        uncertainty = 0.4 * uncertainty + 0.6 * (4 * expected_gain).clamp(0, 1)

    return uncertainty.tolist()
