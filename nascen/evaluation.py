"""Experiment 001 — discover a push rule through interventions in the local simulator."""

import argparse
from dataclasses import asdict, dataclass
from functools import cache
import json
import platform
from random import Random
from statistics import mean
from time import process_time_ns

import torch

from . import __version__
from .body import Action, EmbodiedWorld, Position, shift
from .causal import CONTRADICTION_COST, PREDICATE_COST, CausalKernel
from .model import Learner, context_features
from .semantic import Offset, PushEvent, PushObservation
from .store import MAX_SEED
from .world import DIRECTIONS, HEIGHT, HELD_OUT, MENTOR, TRAIN_PAIRS, WALLS, WIDTH, Object

DEFAULT_BUDGET = 24
TRAIN_PER_TILE = 24
TEST_PER_TILE = 24
TEST_TILES = ("empty", "wall", "object")
CONDITIONS = ("causal_active", "causal_random", "neural_random")
DISCOVERY_THRESHOLD = 0.95
DISCOVERY_STEPS = 3


@dataclass(frozen=True)
class Fixture:
    objects: tuple[Object, ...]
    agent: Position
    direction: str

    def world(self) -> EmbodiedWorld:
        return EmbodiedWorld(0, 0, list(self.objects), self.agent)

    def preview(self) -> PushObservation:
        return PushObservation.from_view(self.world().look(), 1, self.direction)

    def execute(self) -> PushEvent:
        world = self.world()
        before = PushObservation.from_view(world.look(), 1, self.direction)
        world.perform(Action(self.direction, 1))
        after = world.look().objects[1].position
        return PushEvent(before, after)


def fixture_pool(seed: int, evaluation: bool) -> list[Fixture]:
    rng = Random(seed)
    pairs = HELD_OUT if evaluation else TRAIN_PAIRS
    columns = range(WIDTH // 2, WIDTH) if evaluation else range(WIDTH // 2)
    blocked = {*WALLS, MENTOR}
    by_tile: dict[str, list[Fixture]] = {tile: [] for tile in TEST_TILES}
    for x in columns:
        for y in range(HEIGHT):
            position = (x, y)
            if position in blocked:
                continue
            for direction, (dx, dy) in DIRECTIONS.items():
                agent = (x - dx, y - dy)
                if agent in blocked or not (0 <= agent[0] < WIDTH and 0 <= agent[1] < HEIGHT):
                    continue
                color, shape = rng.choice(pairs)
                obj = Object(1, color, shape, x, y, rng.randrange(1_000_000))
                landing = shift(position, direction)
                if landing == MENTOR:
                    continue
                fixture = Fixture((obj,), agent, direction)
                tile = fixture.preview().destination
                by_tile[tile].append(fixture)
                if tile == "empty":
                    blocker = Object(2, color, shape, *landing, rng.randrange(1_000_000))
                    by_tile["object"].append(Fixture((obj, blocker), agent, direction))
    limit = TEST_PER_TILE if evaluation else TRAIN_PER_TILE
    fixtures = []
    for options in by_tile.values():
        rng.shuffle(options)
        fixtures.extend(options[:limit])
    rng.shuffle(fixtures)
    return fixtures


class NeuralMotion:
    def __init__(self, seed: int):
        self.learner = Learner(seed)
        self.records = []

    def observe(self, event: PushEvent) -> None:
        context = context_features(event.before.neighbors)
        direction = tuple(DIRECTIONS).index(event.before.direction)
        self.records.append((context, direction, event.offset != (0, 0)))
        # Comparable small online updates with replay over past interventions.
        self.learner.train_motion(self.records[-8:])

    @torch.no_grad()
    def predict(self, observation: PushObservation) -> dict[Offset, float]:
        move = self.learner.motion(
            torch.tensor(context_features(observation.neighbors)),
            torch.tensor(tuple(DIRECTIONS).index(observation.direction)),
        ).sigmoid().item()
        return {(0, 0): 1 - move, (1, 0): move}


@cache
def held_events(seed: int) -> tuple[PushEvent, ...]:
    return tuple(fixture.execute() for fixture in fixture_pool(seed, evaluation=True))


def score(learner: CausalKernel | NeuralMotion, held: tuple[PushEvent, ...], step: int) -> dict:
    correct, squares = 0, 0.0
    for event in held:
        prediction = learner.predict(event.before)
        offset = max(prediction, key=prediction.__getitem__)
        correct += int(event.before.position_after(offset) == event.after)
        squares += (1 - prediction[(0, 0)] - (event.offset != (0, 0))) ** 2
    return {"step": step, "position_accuracy": correct / len(held), "brier": squares / len(held)}


def discovery_step(curve: list[dict]) -> int | None:
    for start in range(len(curve) - DISCOVERY_STEPS + 1):
        section = curve[start:start + DISCOVERY_STEPS]
        if all(point["position_accuracy"] >= DISCOVERY_THRESHOLD for point in section):
            return section[0]["step"]
    return None


def trial(seed: int, condition: str, fixtures: list[Fixture], budget: int, measure_time: bool) -> dict:
    held = held_events(101)
    rng = Random(seed * 13 + 723)
    random_order = rng.sample(range(len(fixtures)), len(fixtures))
    tie_break = {index: rng.random() for index in range(len(fixtures))}
    previews = {index: fixture.preview() for index, fixture in enumerate(fixtures)}
    started = process_time_ns()
    learner = NeuralMotion(seed) if condition == "neural_random" else CausalKernel(tuple(previews.values()))
    learning_time = process_time_ns() - started
    selection_time, intervention_time = 0, 0
    curve = [score(learner, held, 0)]
    history = []

    for step in range(1, budget + 1):
        started = process_time_ns()
        index = learner.select(previews, tie_break) if condition == "causal_active" else random_order[step - 1]
        selection_time += process_time_ns() - started
        previews.pop(index)
        started = process_time_ns()
        experience = fixtures[index].execute()
        intervention_time += process_time_ns() - started
        started = process_time_ns()
        learner.observe(experience)
        learning_time += process_time_ns() - started
        curve.append(score(learner, held, step))
        history.append({"fixture": index, "event": asdict(experience)})

    final = {
        "seed": seed, "condition": condition, "discovery_step": discovery_step(curve), "curve": curve,
        "interventions": history, "candidate_previews": len(fixtures),
    }
    if isinstance(learner, CausalKernel):
        final.update(program=learner.program.describe(learner.offsets), programs_considered=len(learner.programs),
                     program_updates=len(learner.programs) * budget, planner_predictions=learner.planner_predictions)
    else:
        final.update(optimizer_steps=budget, examples_processed=sum(min(8, step) for step in range(1, budget + 1)))
    if measure_time:
        final["cpu_ms"] = {
            "selection": selection_time / 1e6, "intervention": intervention_time / 1e6, "learning": learning_time / 1e6,
            "total": (selection_time + intervention_time + learning_time) / 1e6,
        }
    return final


def run(seeds: list[int], budget: int = DEFAULT_BUDGET, *, measure_time: bool = False) -> dict:
    trials = []
    for seed in seeds:
        fixtures = fixture_pool(seed, evaluation=False)
        if not 1 <= budget <= len(fixtures):
            raise ValueError(f"Budget must be between 1 and {len(fixtures)} interventions.")
        trials.extend(trial(seed, condition, fixtures, budget, measure_time) for condition in CONDITIONS)
    summary = []
    for condition in CONDITIONS:
        group = [result for result in trials if result["condition"] == condition]
        discovery = [result["discovery_step"] for result in group if result["discovery_step"] is not None]
        row = {
            "condition": condition, "mean_final_position_accuracy": mean(r["curve"][-1]["position_accuracy"] for r in group),
            "mean_final_brier": mean(r["curve"][-1]["brier"] for r in group),
            "discoveries": len(discovery), "mean_discovery_step": mean(discovery) if discovery else None,
        }
        if measure_time:
            row["mean_total_cpu_ms"] = mean(result["cpu_ms"]["total"] for result in group)
        summary.append(row)
    return {
        "experiment": "001 — Discover the Rule", "model_version": __version__, "seeds": seeds, "budget": budget,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine()},
        "protocol": {
            "held_out": len(held_events(101)), "source_pool_per_tile": TRAIN_PER_TILE,
            "predicate_cost": PREDICATE_COST, "contradiction_cost": CONTRADICTION_COST,
            "discovery_threshold": DISCOVERY_THRESHOLD, "consecutive_checkpoints": DISCOVERY_STEPS,
        },
        "summary": summary, "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 42, 55])
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--measure-time", action="store_true", help="record CPU time excluding held-out evaluation")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed <= MAX_SEED for seed in args.seeds):
        parser.error(f"seeds must be distinct and between 0 and {MAX_SEED}")
    torch.set_num_threads(1)
    try:
        report = run(args.seeds, args.budget, measure_time=args.measure_time)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
