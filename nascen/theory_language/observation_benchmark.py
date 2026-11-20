"""Verify the replacement learner on Experiment 009 and separately held Korean wording."""

import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch

from .dataset import digest, write_json
from .encoder import MultilingualE5, encode_partition
from .evaluation import encode_questions, exam_texts
from .grounded_data import held_out_exam, training_windows
from .grounded_rollout import fit, score
from .observation_encoding import encode_model_questions, encode_observations
from .observation_exam import independent_exam
from .observation_model import predict_observations
from .observation_training import encode_training, fit_observations
from .revision import DEFAULT_ENCODER, predict_revision
from .trainer import EncodedEpisodes, episode_texts, save_checkpoint

PHASES = ("original", "initial", "action", "choices", "all")


def evaluation_metrics(guesses, keys, phases=False):
    if phases:
        return {
            phase: score(tuple(key for key in keys if key.id.startswith(phase + "/")), guesses)[0]
            for phase in PHASES
        }
    return score(keys, guesses)[0]


def evaluate_observations(model, encoder):
    questions, keys = independent_exam()
    encoded, cost = encode_model_questions(encoder, questions, clauses=True)
    guesses = predict_observations(model, encoded)
    return {"metrics": evaluation_metrics(guesses, keys, phases=True), "encoding": cost,
            "predictions": guesses}


def run(encoder, windows=300, seeds=(11, 22, 42), steps=480, checkpoint=None):
    original, original_keys = held_out_exam()
    held, held_keys = independent_exam()
    test_vectors, exam_cost = encode_partition(encoder, exam_texts((*original, *held)))
    observed_vectors, observed_cost = encode_observations(encoder, exam_texts((*original, *held)))
    suites = (("009", original, original_keys), ("unseen", held, held_keys))
    prepared = {name: (encode_questions(questions, test_vectors), encode_questions(questions, observed_vectors), keys)
                for name, questions, keys in suites}

    old_training = tuple(row for window in training_windows(windows) for row in window)
    old_vectors, old_cost = encode_partition(encoder, episode_texts(old_training))
    old_data = EncodedEpisodes.build(old_training, old_vectors)
    new_data = encode_training(encoder, windows)
    trials = []
    for seed in seeds:
        original_model = fit(old_data, encoder.width, "single", seed, steps, 16)
        improved, stats = fit_observations(new_data, encoder.width, seed, steps, 48)
        for method, model in (("009-single", original_model), ("observation", improved)):
            metrics, predictions = {}, {}
            for suite, (plain, segmented, keys) in prepared.items():
                guesses = predict_observations(model, segmented) if method == "observation" else predict_revision(model, plain)[0]
                metrics[suite] = evaluation_metrics(guesses, keys, phases=suite == "unseen")
                predictions[suite] = digest(guesses)
            trials.append({"model": method, "seed": seed, "metrics": metrics, "predictions_sha256": predictions,
                           "parameters": sum(value.numel() for value in model.parameters())})
        if checkpoint is not None and seed == seeds[0]:
            save_checkpoint(checkpoint, improved, encoder, {
                "mode": "observation", "seed": seed, "steps": steps, "batch_size": 48,
                "data": new_data.provenance, "training": stats,
            })
    summary = {}
    for name in ("009-single", "observation"):
        group = [row["metrics"] for row in trials if row["model"] == name]
        summary[name] = {
            "009": {key: mean(row["009"][key] for row in group) for key in group[0]["009"]},
            "unseen": {phase: {key: mean(row["unseen"][phase][key] for row in group)
                               for key in group[0]["unseen"][phase]} for phase in PHASES},
        }
    return {
        "experiment": "010 · Observation-grounded transition model",
        "seeds": seeds, "steps": steps, "observed_transitions": len(old_training), "transitions_per_update": 48,
        "old_training_sha256": digest([asdict(row) for row in old_training]),
        "observation_training": new_data.provenance,
        "exam_sha256": {name: digest([asdict(q) for q in questions])
                        for name, questions in (("009", original), ("unseen", held))},
        "answer_sha256": {name: digest([asdict(k) for k in keys])
                          for name, keys in (("009", original_keys), ("unseen", held_keys))},
        "encoder": encoder.identity, "encoding_cost": {"009-exam": exam_cost, "observation-exam": observed_cost,
                                                      "009-training": old_cost},
        "summary": summary, "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--windows", type=int, default=300)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 22, 42))
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive steps and distinct seeds.")
    torch.set_num_threads(1)
    try:
        report = run(MultilingualE5(args.encoder), args.windows, args.seeds, args.steps, args.checkpoint)
        write_json(args.output, report)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Nascen observation benchmark: {error}\n")
    for name, result in report["summary"].items():
        held = result["unseen"]["all"]
        print(f"{name} · unseen words action pair {held['one_step_pair']:.1%} · full rollout {held['order_exact']:.1%}")


if __name__ == "__main__":
    main()
