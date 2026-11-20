"""Compare active exploration and replay with a seeded, equal-step baseline."""

import argparse
import json
from statistics import mean

import torch

from .experiment import Experiment
from .store import MAX_SEED
from .world import HELD_OUT

CONDITIONS = (("active", True), ("random", True), ("active", False))


def run(seeds: list[int], steps: int, interval: int) -> dict:
    points = sorted({0, steps, *range(interval, steps, interval)})
    trials = []
    for strategy, replay in CONDITIONS:
        for seed in seeds:
            experiment = Experiment(seed, strategy, replay)
            history = [experiment.initial]
            for step in range(1, steps + 1):
                record = experiment.advance()
                if step in points:
                    history.append(record["metrics"])
            trials.append({
                "seed": seed, "strategy": strategy, "replay": replay, "questions": experiment.questions,
                "rejected": experiment.rejected, "history": history,
            })

    summary = []
    for strategy, replay in CONDITIONS:
        group = [trial for trial in trials if trial["strategy"] == strategy and trial["replay"] == replay]
        summary.append({
            "strategy": strategy, "replay": replay,
            "mean_questions": mean(trial["questions"] for trial in group),
            "held_out_curve": [
                {"step": step, "mean_accuracy": mean(trial["history"][index]["accuracy"] for trial in group)}
                for index, step in enumerate(points)
            ],
            "mean_final_seen_accuracy": mean(trial["history"][-1]["seen_accuracy"] for trial in group),
            "mean_final_prediction_error": mean_or_none(
                [trial["history"][-1]["prediction_error"] for trial in group]
            ),
        })
    return {"steps": steps, "held_out": HELD_OUT, "seeds": seeds, "summary": summary, "trials": trials}


def mean_or_none(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 42, 55])
    args = parser.parse_args()
    if args.steps < 1 or args.interval < 1:
        parser.error("steps and interval must be positive")
    if any(not 0 <= seed <= MAX_SEED for seed in args.seeds):
        parser.error(f"seeds must be between 0 and {MAX_SEED}")
    torch.set_num_threads(1)
    print(json.dumps(run(args.seeds, args.steps, args.interval), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
