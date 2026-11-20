"""Compare full, random-budget, and evidence-gated training on 1,000 observations."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import platform
from random import Random
from statistics import mean
from time import process_time_ns

import torch

from .benchmark_v02 import BATCH_SIZE, NEW_PAIRS, OLD_PAIRS, Dataset, Trial, observe_stream
from .knowledge import MIN_LEARNING_SCORE, MIN_SUPPORT, Knowledge, appearance_claim, label_outcome
from .selective import SELECTIVE_VERSION
from .store import MAX_SEED
from .world import COLORS, HELD_OUT, SHAPES

PHASE_OBSERVATIONS = 500
OBSERVATIONS = 2 * PHASE_OBSERVATIONS
LABEL_SPACING = 4
WINDOW = 50
UPDATES_PER_WINDOW = 5
SCENARIOS = ("clean", "noisy")
ARMS = ("full", "random", "selective")
LABELS = tuple((color, shape) for color in range(len(COLORS)) for shape in range(len(SHAPES)))


@dataclass(frozen=True)
class Candidate:
    index: int
    claim: str
    outcome: str
    answer: tuple[int, int]
    features: list[float]
    score: float


class Selector:
    def __init__(self):
        self.knowledge = Knowledge()
        self.assessments: Counter[str] = Counter()
        self.consolidated: list[tuple[str, str]] = []

    def choose(
        self, trial: Trial, pixels: list[tuple[float, ...]], answers: dict[int, tuple[int, int]], start: int,
    ) -> list[Candidate]:
        claims = [appearance_claim(image) for image in pixels]
        for claim in claims:
            self.knowledge.observe(claim)
        for index, answer in answers.items():
            self.knowledge.record(claims[index - start], f"observation:{index}", label_outcome(answer))

        ranked = []
        for index, answer in answers.items():
            claim = claims[index - start]
            outcome = label_outcome(answer)
            features = trial.learner.features(pixels[index - start])
            decision = self.knowledge.assess(
                claim, outcome, lambda: trial.learner.label_probability(features, list(answer)),
            )
            self.assessments[decision["status"]] += 1
            if decision.get("score", 0) >= MIN_LEARNING_SCORE:
                ranked.append(Candidate(index, claim, outcome, answer, features, decision["score"]))

        selected = {}
        for candidate in sorted(ranked, key=lambda item: (-item.score, item.index)):
            selected.setdefault(candidate.claim, candidate)
            if len(selected) == UPDATES_PER_WINDOW:
                break
        return list(selected.values())

    def replay_ok(self, index: int) -> bool:
        claim, outcome = self.consolidated[index]
        return self.knowledge.verify(claim).accepted == outcome

    def train(self, trial: Trial, candidate: Candidate) -> None:
        trial.learn(candidate.features, candidate.answer, replay_ok=self.replay_ok)
        self.consolidated.append((candidate.claim, candidate.outcome))
        self.knowledge.mark_trained(candidate.claim, candidate.outcome)


def experiment(
    seed: int, arm: str, pixels: list[tuple[float, ...]], reports: list[tuple[int, int]],
    offered: set[int], corrupted: set[int], probes: Dataset, measure_time: bool,
    budget: list[int] | None = None,
) -> dict:
    trial = Trial(seed, "factored", True, probes)
    selection = Selector() if arm == "selective" else None
    rng = Random(seed * 193 + 13)
    labels_received = 0
    selected_ids = []
    update_counts = []
    selection_time, training_time = 0, 0
    curve = [trial.point()]

    for block, start in enumerate(range(0, len(pixels), WINDOW)):
        started = process_time_ns()
        indices = range(start, start + WINDOW)
        answers = {index: reports[index] for index in indices if arm == "full" or index in offered}
        trial.observations += WINDOW
        labels_received += len(answers)
        if selection is not None:
            candidates = selection.choose(trial, pixels[start:start + WINDOW], answers, start)
            chosen = [candidate.index for candidate in candidates]
        elif arm == "random":
            chosen = sorted(rng.sample(list(answers), budget[block]))
        else:
            chosen = list(answers)
        selection_time += process_time_ns() - started

        started = process_time_ns()
        if selection is not None:
            for candidate in candidates:
                selection.train(trial, candidate)
        else:
            for index in chosen:
                trial.learn(trial.learner.features(pixels[index]), reports[index])
        training_time += process_time_ns() - started

        update_counts.append(len(chosen))
        selected_ids.extend(chosen)
        curve.append(trial.point() | {"questions": labels_received})
    initial_old = curve[PHASE_OBSERVATIONS // WINDOW]["old_accuracy"]
    report = {
        "seed": seed, "condition": arm, "model_version": SELECTIVE_VERSION, "encoder": "factored", "replay": True,
        "curve": curve, "updates_by_window": update_counts, "selected_observations": selected_ids,
        "labels_received": labels_received, "corrupted_labels_trained": sum(index in corrupted for index in selected_ids),
        "old_accuracy_at_switch": initial_old, "old_accuracy_change": curve[-1]["old_accuracy"] - initial_old,
    }
    if selection is not None:
        report.update(verification=dict(selection.assessments), knowledge=selection.knowledge.counts())
    if measure_time:
        report["cpu_ms"] = {
            "selection": selection_time / 1e6, "training": training_time / 1e6, "total": (selection_time + training_time) / 1e6,
        }
    return report


def summarize(results: list[dict], measure_time: bool) -> list[dict]:
    summaries = []
    for arm in ARMS:
        group = [trial for trial in results if trial["condition"] == arm]
        row = {
            "condition": arm,
            "mean_final": {key: mean(trial["curve"][-1][key] for trial in group) for key in group[0]["curve"][-1]},
            "mean_old_accuracy_at_switch": mean(trial["old_accuracy_at_switch"] for trial in group),
            "mean_old_accuracy_change": mean(trial["old_accuracy_change"] for trial in group),
            "mean_corrupted_labels_trained": mean(trial["corrupted_labels_trained"] for trial in group),
        }
        if measure_time:
            row["mean_cpu_ms"] = {
                stage: mean(trial["cpu_ms"][stage] for trial in group) for stage in ("selection", "training", "total")
            }
        summaries.append(row)
    return summaries


def run(seeds: list[int], scenarios: tuple[str, ...] = SCENARIOS, *, measure_time: bool = False) -> dict:
    probes = {
        "held_out": observe_stream(3017, HELD_OUT, 160, evaluation=True),
        "old": observe_stream(3019, OLD_PAIRS, 100, evaluation=True),
        "new": observe_stream(3023, NEW_PAIRS, 96, evaluation=True),
    }
    results = {scenario: [] for scenario in scenarios}
    for seed in seeds:
        stream = [
            *observe_stream(seed * 5 + 17, OLD_PAIRS, PHASE_OBSERVATIONS),
            *observe_stream(seed * 5 + 23, NEW_PAIRS, PHASE_OBSERVATIONS),
        ]
        pixels = [example.pixels for example in stream]
        rng = Random(seed * 197 + 11)
        offered = {start + rng.randrange(LABEL_SPACING) for start in range(0, OBSERVATIONS, LABEL_SPACING)}
        for scenario in scenarios:
            corrupted = set(rng.sample(range(OBSERVATIONS), OBSERVATIONS // 5)) if scenario == "noisy" else set()
            reports = [example.answer for example in stream]
            for index in sorted(corrupted):
                reports[index] = rng.choice([label for label in LABELS if label != reports[index]])
            shared = pixels, reports, offered, corrupted, probes, measure_time
            selected = experiment(seed, "selective", *shared)
            random = experiment(seed, "random", *shared, budget=selected["updates_by_window"])
            full = experiment(seed, "full", *shared)
            results[scenario].extend((full, random, selected))
    return {
        "experiment": "Selective learning", "model_version": SELECTIVE_VERSION, "seeds": seeds,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine()},
        "protocol": {
            "observations": OBSERVATIONS, "phase_observations": PHASE_OBSERVATIONS,
            "offered_labels": OBSERVATIONS // LABEL_SPACING, "window": WINDOW,
            "updates_per_window_cap": UPDATES_PER_WINDOW, "noisy_fraction": 0.2, "batch_size": BATCH_SIZE,
            "minimum_corroboration": MIN_SUPPORT, "minimum_learning_score": MIN_LEARNING_SCORE,
            "held_out_pairs": HELD_OUT, "evaluation_sizes": {name: len(dataset) for name, dataset in probes.items()},
        },
        "scenarios": {
            name: {"summary": summarize(trials, measure_time), "trials": trials} for name, trials in results.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 42, 55])
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--measure-time", action="store_true", help="measure CPU use for selection and training")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed <= MAX_SEED for seed in args.seeds):
        parser.error(f"seeds must be distinct and between 0 and {MAX_SEED}")
    torch.set_num_threads(1)
    selected = (args.scenario,) if args.scenario else SCENARIOS
    print(json.dumps(run(args.seeds, selected, measure_time=args.measure_time), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
