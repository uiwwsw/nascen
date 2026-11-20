"""Controlled comparisons for embodied observation, questions, and retention."""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import platform
from random import Random
from statistics import mean

import torch

from .body import Action, EmbodiedWorld
from .embodied import MODEL_VERSION, EmbodiedConfig
from .experiment import MEMORY_SIZE
from .model import ENCODERS, Learner
from .policy import QUESTION_STRATEGIES, question_scores
from .store import MAX_SEED
from .world import HELD_OUT, SLOTS, TRAIN_PAIRS, Object, concept_target

EXPERIMENTS = ("A", "B", "C")
VISION_ANSWERS = 120
QUESTION_BUDGET = 10
QUESTION_WINDOW = 8
PHASE_ANSWERS = 80
BATCH_SIZE = 8
QUESTION_COST = EmbodiedConfig().question_cost
OLD_PAIRS = tuple(pair for pair in TRAIN_PAIRS if pair[1] != "triangle")
NEW_PAIRS = tuple(pair for pair in TRAIN_PAIRS if pair[1] == "triangle")
EVAL_SLOTS = ((0, 3), (0, 1), (3, 1), (5, 3))
# All four objects become visible along this valid walk in either layout.
WALK = ("north", "north", "east", "east", "east", "south")
Pair = tuple[str, str]
Dataset = dict[str, Sequence["Example"]]


@dataclass(frozen=True)
class Example:
    pixels: tuple[float, ...]
    answer: tuple[int, int]


def observe_stream(seed: int, pairs: tuple[Pair, ...], count: int, *, evaluation: bool = False) -> list[Example]:
    """Give all conditions the same pixel crops from the same local observation walk."""
    if not pairs or count < 0:
        raise ValueError("A stream needs at least one pairing and a non-negative size.")
    rng = Random(seed)
    slots = EVAL_SLOTS if evaluation else SLOTS
    shuffled: list[Pair] = []
    while len(shuffled) < count + len(slots):
        subset = list(pairs)
        rng.shuffle(subset)
        shuffled.extend(subset)

    result = []
    for start in range(0, count, len(slots)):
        objects = [
            Object(index + 1, color, shape, x, y, rng.randrange(1_000_000) + (3_000_000 if evaluation else 0))
            for index, ((color, shape), (x, y)) in enumerate(
                zip(shuffled[start:start + len(slots)], slots, strict=True)
            )
        ]
        world = EmbodiedWorld(seed, start, objects)
        observed = world.look().objects
        for direction in WALK:
            world.perform(Action(f"walk {direction}"))
            observed.update(world.look().objects)
        # Keep the balanced stream order regardless of when an object first came into view.
        result.extend(Example(observed[obj.id].pixels, concept_target(obj)) for obj in objects)
    return result[:count]


class Trial:
    def __init__(self, seed: int, encoder: str, replay: bool, probes: Dataset):
        self.seed = seed
        self.encoder = encoder
        self.replay = replay
        self.learner = Learner(seed, encoder)
        self.rng = Random(seed * 97 + 181)
        self.labeled: list[tuple[list[float], list[int]]] = []
        self.gains: list[tuple[list[float], float]] = []
        self.probes = {
            name: (
                torch.tensor([self.learner.features(example.pixels) for example in dataset]),
                torch.tensor([example.answer for example in dataset]),
            )
            for name, dataset in probes.items()
        }
        self.observations = 0

    def learn(
        self, features: list[float], answer: tuple[int, int], *, replay_ok: Callable[[int], bool] | None = None,
    ) -> None:
        fresh = features, list(answer)
        memory = self.labeled[-MEMORY_SIZE:]
        if replay_ok is not None:
            start = len(self.labeled) - len(memory)
            memory = [example for index, example in enumerate(memory, start) if replay_ok(index)]
        # Repeating the newest example preserves a single-example mean gradient.
        # This keeps processed examples and optimizer steps equal across replay arms.
        past = self.rng.choices(memory, k=BATCH_SIZE - 1) if self.replay and memory else [fresh] * (BATCH_SIZE - 1)
        before = self.learner.label_probability(*fresh)
        self.learner.train_concepts([fresh, *past])
        self.labeled.append(fresh)
        improvement = max(0.0, self.learner.label_probability(*fresh) - before)
        self.gains.append((features, improvement))

    def point(self) -> dict:
        answers = len(self.labeled)
        return {
            "observations": self.observations, "questions": answers, "optimizer_steps": answers,
            "examples_processed": answers * BATCH_SIZE,
            **{f"{name}_accuracy": self.learner.joint_accuracy(*dataset) for name, dataset in self.probes.items()},
        }

    def result(self, condition: str, curve: list[dict], **extra) -> dict:
        vision = (self.learner.color, self.learner.shape)
        if self.encoder == "shared":
            vision += (self.learner.visual_encoder,)
        parameters = sum(weight.numel() for module in vision for weight in module.parameters())
        return {
            "seed": self.seed, "condition": condition, "model_version": MODEL_VERSION,
            "encoder": self.encoder, "replay": self.replay, "vision_parameters": parameters,
            "curve": curve, **extra,
        }


def vision_trials(seed: int, probes: Dataset) -> list[dict]:
    stream = observe_stream(seed * 3 + 17, TRAIN_PAIRS, VISION_ANSWERS)
    results = []
    for encoder in ENCODERS:
        trial = Trial(seed, encoder, True, probes)
        curve = [trial.point()]
        for example in stream:
            trial.observations += 1
            trial.learn(trial.learner.features(example.pixels), example.answer)
            if trial.observations % 20 == 0:
                curve.append(trial.point())
        results.append(trial.result(encoder, curve))
    return results


def question_trials(seed: int, probes: Dataset) -> list[dict]:
    stream = observe_stream(seed * 3 + 23, TRAIN_PAIRS, QUESTION_BUDGET * QUESTION_WINDOW)
    results = []
    for encoder in ENCODERS:
        for strategy in QUESTION_STRATEGIES:
            results.append(acquire_labels(seed, encoder, strategy, stream, probes))
    return results


def acquire_labels(seed: int, encoder: str, strategy: str, stream: list[Example], probes: Dataset) -> dict:
    trial = Trial(seed, encoder, True, probes)
    rng = Random(seed * 97 + 19)
    curve = [trial.point()]
    selected = []
    for start in range(0, len(stream), QUESTION_WINDOW):
        window = stream[start:start + QUESTION_WINDOW]
        features = [trial.learner.features(example.pixels) for example in window]
        trial.observations += len(features)
        scores = question_scores(trial.learner, features, strategy, trial.gains, rng)
        index = max(range(len(scores)), key=scores.__getitem__)
        selected.append(start + index)
        trial.learn(features[index], window[index].answer)
        curve.append(trial.point())
    return trial.result(
        f"{encoder}/{strategy}", curve, selected_observations=selected,
        question_strategy=strategy, acquisition_cost=QUESTION_BUDGET * QUESTION_COST,
    )


def retention_trials(seed: int, probes: Dataset) -> list[dict]:
    first = observe_stream(seed * 3 + 29, OLD_PAIRS, PHASE_ANSWERS)
    second = observe_stream(seed * 3 + 31, NEW_PAIRS, PHASE_ANSWERS)
    results = []
    for replay in (False, True):
        trial = Trial(seed, "shared", True, probes)
        curve = [trial.point()]
        before_switch = None
        for example in [*first, *second]:
            trial.observations += 1
            trial.learn(trial.learner.features(example.pixels), example.answer)
            if trial.observations % 20 == 0:
                curve.append(trial.point())
            if trial.observations == PHASE_ANSWERS:
                before_switch = curve[-1]
                trial.replay = replay
        results.append(trial.result(
            "replay" if replay else "newest", curve, before_switch=before_switch,
            old_accuracy_change=curve[-1]["old_accuracy"] - before_switch["old_accuracy"],
        ))
    return results


def summarize(trials: list[dict]) -> list[dict]:
    summary = []
    for condition in dict.fromkeys(trial["condition"] for trial in trials):
        group = [trial for trial in trials if trial["condition"] == condition]
        points = list(zip(*(trial["curve"] for trial in group), strict=True))
        mean_curve = [{key: mean(point[key] for point in sample) for key in sample[0]} for sample in points]
        result = {"condition": condition, "mean_curve": mean_curve}
        if "old_accuracy_change" in group[0]:
            result["mean_old_accuracy_change"] = mean(trial["old_accuracy_change"] for trial in group)
        summary.append(result)
    return summary


def run(seeds: list[int], experiments: tuple[str, ...] = EXPERIMENTS) -> dict:
    probes = {
        "held_out": observe_stream(917, HELD_OUT, 160, evaluation=True),
        "train_split": observe_stream(918, TRAIN_PAIRS, 160, evaluation=True),
        "old": observe_stream(919, OLD_PAIRS, 100, evaluation=True),
        "new": observe_stream(923, NEW_PAIRS, 96, evaluation=True),
    }
    runners = {"A": vision_trials, "B": question_trials, "C": retention_trials}
    results = {}
    for name in experiments:
        trials = [trial for seed in seeds for trial in runners[name](seed, probes)]
        results[name] = {"summary": summarize(trials), "trials": trials}
    return {
        "model_version": MODEL_VERSION, "seeds": seeds,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine()},
        "protocol": {
            "vision_answers": VISION_ANSWERS, "question_budget": QUESTION_BUDGET, "question_window": QUESTION_WINDOW,
            "retention_phase_answers": PHASE_ANSWERS, "batch_size": BATCH_SIZE, "held_out_pairs": HELD_OUT,
            "question_cost": QUESTION_COST, "replay_capacity": MEMORY_SIZE, "retention_phase_one_replay": True,
            "evaluation_sizes": {name: len(dataset) for name, dataset in probes.items()},
        },
        "experiments": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 42, 55])
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    args = parser.parse_args()
    if any(not 0 <= seed <= MAX_SEED for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error(f"seeds must be distinct and between 0 and {MAX_SEED}")
    torch.set_num_threads(1)
    experiments = (args.experiment,) if args.experiment else EXPERIMENTS
    print(json.dumps(run(args.seeds, experiments), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
