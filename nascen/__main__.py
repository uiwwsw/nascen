"""Command-line interface for running a local learning experiment."""

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any

import torch

from .experiment import STRATEGIES
from .store import MAX_BATCH, MAX_SEED, InvalidRequest, Store


def show(state: dict[str, Any]) -> None:
    metrics = state["metrics"]
    print(f'Run {state["run_id"]} · seed {state["seed"]} · {state["strategy"]} exploration'
          f' · replay {"on" if state["replay"] else "off"}')
    print(f'{state["steps"]} experiences · {state["questions"]} mentor questions'
          f' · {state["rejected"]} protected-tile proposals blocked')
    print(f'Held-out accuracy {metrics["accuracy"]:.1%} · seen-combination accuracy {metrics["seen_accuracy"]:.1%}')
    print(f'Push prediction error {number(metrics["prediction_error"])} · mean reward {number(metrics["mean_reward"])}')

    latest = state["latest"]
    if latest is not None:
        action = describe_action(latest["action"])
        obj = latest["object"]
        estimate = latest["prediction"]
        forecast = f"{estimate:.0%} chance of movement" if isinstance(estimate, float) else appearance(estimate)
        outcome = describe_outcome(latest)
        print(f'\nLatest · #{latest["step"]} · {action} ({appearance(obj)})')
        print(f'  Prediction: {forecast} · outcome: {outcome} · reward: {latest["reward"]:+.3f}')
        if latest["approval"] is not None:
            print(f'  Mentor feedback: {latest["approval"]:+d}')

    world = state["world"]
    print(f'\nScene {world["scene"]} · objects available to teach')
    for obj in world["objects"]:
        cell = f'{chr(ord("A") + obj["x"])}{world["height"] - obj["y"]}'
        print(f'  {obj["id"]}. {appearance(obj):16} {cell} · agent guesses {appearance(obj["prediction"])}')


def appearance(obj: dict) -> str:
    return f'{obj["color"]} {obj["shape"]}'


def describe_action(action: str) -> str:
    return {"inspect": "inspect", "ask": "ask the mentor"}.get(action, f"push {action}")


def describe_outcome(episode: dict) -> str:
    if episode["label"] is not None:
        return episode["label"]
    if episode["moved"] is not None:
        return "object moved" if episode["moved"] else "object stayed"
    return "observed"


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(".nascen/lab.sqlite3"), help="SQLite file for this lab")
    parser.add_argument("--json", action="store_true", help="print run state as JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show the current run, metrics and objects")
    train = commands.add_parser("train", help="advance the current run")
    train.add_argument("--steps", type=int, default=50, choices=range(1, MAX_BATCH + 1), metavar="1–50")
    teach = commands.add_parser("teach", help="label an object from the current scene")
    teach.add_argument("object", type=int, help="object number shown by status")
    feedback = commands.add_parser("feedback", help="rate the latest experience once")
    feedback.add_argument("approval", type=int, choices=(-1, 1))
    reset = commands.add_parser("reset", help="start a fresh run and keep the previous run in SQLite")
    reset.add_argument("--seed", type=int, default=42, help=f"random seed in 0–{MAX_SEED}")
    reset.add_argument("--strategy", default="active", choices=STRATEGIES)
    reset.add_argument("--no-replay", action="store_true", help="train only on the newest experience")
    return parser, parser.parse_args()


def main() -> None:
    parser, args = arguments()
    torch.set_num_threads(1)
    try:
        store = Store(args.db)
        state = store.get()
        run_id = state["run_id"]
        match args.command:
            case "train":
                state = store.step(run_id, args.steps)
            case "teach":
                state = store.step(run_id, 1, teach_id=args.object, scene=state["world"]["scene"])
            case "feedback":
                state = store.feedback(run_id, state["steps"], args.approval)
            case "reset":
                state = store.start(run_id, args.seed, args.strategy, not args.no_replay)
    except (InvalidRequest, OSError, sqlite3.Error) as error:
        parser.exit(1, f"nascen: {error}\n")

    if args.json:
        print(json.dumps(state, indent=2, allow_nan=False))
    else:
        show(state)


if __name__ == "__main__":
    main()
