"""Prepare, train, evaluate or benchmark Nascen's transition learner."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..teacher import DEFAULT_PORT
from .benchmark import BATCH, SEEDS, STEPS, run
from .dataset import BUDGETS, Question, digest, load_partition, prepare, read_json, write_json
from .encoder import MultilingualE5, encode_partition
from .evaluation import encode_questions, exam_texts, grade, predict, route_statistics
from .teacher import collect, load_reviewed
from .trainer import (
    MODES, EncodedEpisodes, LossWeights, episode_texts, fit, lesson_texts, load_checkpoint, save_checkpoint,
    theory_targets,
)

DEFAULT_DATA = Path(".nascen/theory-language/data")
DEFAULT_ENCODER = Path(".nascen/teacher-assets/multilingual-e5-small")
DEFAULT_LESSONS = Path("docs/experiment-007-theory-lessons.json")
DEFAULT_REVIEW = Path("docs/experiment-007-theory-review.json")
DEFAULT_CHECKPOINT = Path(".nascen/theory-language/principle.pt")


def add_training_options(parser, *, needs_data=True):
    if needs_data:
        parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--lessons", type=Path, default=DEFAULT_LESSONS)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    for field in ("next", "contrast", "principle", "consistency"):
        parser.add_argument(f"--w-{field}", type=float, default=getattr(LossWeights(), field))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare")
    data.add_argument("--output", type=Path, default=DEFAULT_DATA)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--size", type=int, default=max(BUDGETS))
    data.add_argument("--with-theory", action="store_true", help="include reviewed principle prose in the training manifest")
    data.add_argument("--lessons", type=Path, default=DEFAULT_LESSONS)
    data.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    teacher = commands.add_parser("collect")
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--teacher", required=True)
    teacher.add_argument("--port", type=int, default=DEFAULT_PORT)
    teacher.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train")
    add_training_options(train)
    train.add_argument("--mode", choices=MODES, required=True)
    train.add_argument("--budget", type=int, default=max(BUDGETS))
    train.add_argument("--seed", type=int, default=SEEDS[0])
    train.add_argument("--checkpoint", type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    evaluate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluate.add_argument("--output", type=Path)
    bench = commands.add_parser("benchmark")
    add_training_options(bench)
    bench.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    bench.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    bench.add_argument("--checkpoint-dir", type=Path)
    bench.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("predict")
    infer.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    infer.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    infer.add_argument("--before", required=True)
    infer.add_argument("--action", action="append", required=True)
    infer.add_argument("--option", action="append", nargs="+", required=True)
    return parser, parser.parse_args()


def parse_weights(args):
    return LossWeights(**{field: getattr(args, f"w_{field}") for field in ("next", "contrast", "principle", "consistency")})


def train_command(args, encoder):
    episodes = load_partition(args.data, "train")
    if not 1 <= args.budget <= len(episodes):
        raise ValueError("The episode budget is outside the prepared training partition.")
    episodes = episodes[:args.budget]
    manifest = read_json(args.data / "manifest.json")
    vectors, encode_cost = encode_partition(encoder, episode_texts(episodes))
    targets = None
    weights = parse_weights(args)
    metadata = {
        "mode": args.mode, "budget": args.budget, "seed": args.seed, "dataset": manifest,
        "steps": args.steps, "batch_size": args.batch_size, "loss_weights": asdict(weights),
    }
    if args.mode != "semantic":
        collection = read_json(args.lessons)
        review = read_json(args.review)
        if manifest["seed"] != collection["teacher"]["seed"]:
            raise ValueError("Use the same training seed as the teacher collection.")
        lessons = load_reviewed(collection, review)
        teacher_vectors, _ = encode_partition(encoder, lesson_texts(lessons))
        targets, stats = theory_targets(episodes, lessons, teacher_vectors, args.mode, args.seed)
        metadata |= {"teacher": digest(collection), "review": digest(review), "principle_targets": stats}
    data = EncodedEpisodes.build(episodes, vectors)
    model, stats = fit(data, encoder.width, args.seed, args.steps, args.batch_size, weights, targets)
    path = args.checkpoint or DEFAULT_CHECKPOINT.with_name(f"{args.mode}.pt")
    save_checkpoint(path, model, encoder, metadata)
    return {"checkpoint": str(path), "metadata": metadata, "encoding": encode_cost, "training": stats}


def main():
    parser, args = arguments()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        if args.command == "prepare":
            lessons = None
            if args.with_theory:
                collection = read_json(args.lessons)
                if args.seed != collection["teacher"]["seed"]:
                    raise ValueError("Use the same training seed as the teacher collection.")
                lessons = load_reviewed(collection, read_json(args.review))
            result = prepare(args.output, args.seed, size=args.size, lessons=lessons)
        elif args.command == "collect":
            raw = collect(args.output, args.teacher, args.port, args.seed)
            result = {"collection": str(args.output), "principles": len(raw["lessons"]), "sha256": digest(raw)}
        else:
            encoder = MultilingualE5(args.encoder)
            if args.command == "train":
                result = train_command(args, encoder)
            elif args.command == "benchmark":
                if any(budget < 1 for budget in args.budgets):
                    raise ValueError("All budgets must be positive.")
                if len(set(args.budgets)) != len(args.budgets) or len(set(args.seeds)) != len(args.seeds):
                    raise ValueError("Budgets and seeds must be distinct.")
                report = run(
                    args.data, encoder, read_json(args.lessons), read_json(args.review), parse_weights(args),
                    budgets=args.budgets, seeds=args.seeds, steps=args.steps, batch_size=args.batch_size,
                    checkpoint_dir=args.checkpoint_dir,
                )
                write_json(args.output, report)
                result = {"report": str(args.output), "summary": report["summary"]}
            elif args.command == "evaluate":
                model, metadata = load_checkpoint(args.checkpoint, encoder)
                questions = load_partition(args.data, "exam")
                vectors, _ = encode_partition(encoder, exam_texts(questions))
                choices, routes, _ = predict(model, encode_questions(questions, vectors))
                metrics = grade(load_partition(args.data, "answers"), choices)
                result = {"student": metadata, "metrics": metrics, "routing": route_statistics(routes),
                          "predictions": choices}
                if args.output:
                    write_json(args.output, result)
            else:
                if len(args.action) != len(args.option) or len({len(options) for options in args.option}) != 1:
                    raise ValueError("Provide an --option group of the same size for each --action.")
                if any(len(set(options)) != len(options) or len(options) < 2 for options in args.option):
                    raise ValueError("Each action needs at least two distinct candidate outcomes.")
                model, _ = load_checkpoint(args.checkpoint, encoder)
                question = Question("query", args.before, args.action, args.option)
                vectors, _ = encode_partition(encoder, exam_texts([question]))
                choices, _, _ = predict(model, encode_questions([question], vectors))
                result = {"indices": choices["query"], "outcomes": [
                    options[i] for options, i in zip(args.option, choices["query"], strict=True)
                ]}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Nascen theory language: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
