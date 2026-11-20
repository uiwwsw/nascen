"""Matched-budget comparisons and experiment provenance."""

from dataclasses import asdict
import json
import platform
from statistics import mean
from time import process_time

import torch

from .dataset import BUDGETS, digest, load_partition, read_json
from .encoder import encode_partition
from .evaluation import (
    encode_questions, exam_texts, grade, predict, predict_copy, predict_rules, route_statistics,
)
from .rule_control import RULE_EFFECTS, RuleControl, reference_texts
from .teacher import cost, load_reviewed
from .trainer import EncodedEpisodes, episode_texts, fit, lesson_texts, save_checkpoint, theory_targets

SEEDS = (11, 22, 42)
STEPS = 240
BATCH = 32
ARM = {"semantic": "A", "principle": "B", "shuffled": "C", "noisy": "B-noisy"}


def bytes_used(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def run(data, encoder, collection, review, weights, *, budgets=BUDGETS, seeds=SEEDS, steps=STEPS,
        batch_size=BATCH, checkpoint_dir=None):
    episodes = load_partition(data, "train")
    questions = load_partition(data, "exam")
    manifest = read_json(data / "manifest.json")
    if len(episodes) < max(budgets):
        raise ValueError("The prepared training partition is smaller than the requested budget.")
    if manifest["seed"] != collection["teacher"]["seed"]:
        raise ValueError("Use training data with the same seed as the teacher collection.")
    lessons = load_reviewed(collection, review)
    train_vectors, training_cost = encode_partition(encoder, episode_texts(episodes[:max(budgets)]))
    theory_vectors, theory_cost = encode_partition(encoder, lesson_texts(lessons))
    exam_vectors, exam_cost = encode_partition(encoder, exam_texts(questions))
    rule_vectors, rule_cost = encode_partition(encoder, reference_texts())
    exam_inputs = encode_questions(questions, exam_vectors)

    # Predictions are fully computed before loading the answer manifest.
    trials = []
    rule_control = RuleControl(rule_vectors, lessons)
    start = process_time()
    stored_predictions = predict_rules(rule_control, exam_inputs)
    trials.append({"arm": "D", "budget": None, "seed": None, "inference_cpu_seconds": process_time() - start,
                   "predictions": stored_predictions})
    trials.append({"arm": "E5-copy", "budget": None, "seed": None, "predictions": predict_copy(exam_inputs)})
    for budget in budgets:
        cohort = episodes[:budget]
        encoded = EncodedEpisodes.build(cohort, train_vectors)
        modes = ("semantic", "principle", "shuffled", "noisy") if budget == 300 else ("semantic", "principle", "shuffled")
        for mode in modes:
            for seed in seeds:
                targets, theory_stats = theory_targets(cohort, lessons, theory_vectors, mode, seed)
                model, learning = fit(encoded, encoder.width, seed, steps, batch_size, weights, targets)
                output, routes, inference_seconds = predict(model, exam_inputs)
                uniform, _, _ = predict(model, exam_inputs, uniform=True)
                trials.append({
                    "arm": ARM[mode], "budget": budget, "seed": seed, "training": learning,
                    "theory": theory_stats, "routing": route_statistics(routes),
                    "inference_cpu_seconds": inference_seconds, "predictions": output, "uniform": uniform,
                })
                if checkpoint_dir is not None:
                    metadata = {
                        "mode": mode, "budget": budget, "seed": seed, "dataset": manifest,
                        "steps": steps, "batch_size": batch_size, "loss_weights": asdict(weights),
                    }
                    if targets is not None:
                        metadata |= {"teacher": digest(collection), "review": digest(review), "principle_targets": theory_stats}
                    save_checkpoint(checkpoint_dir / f"{mode}-{budget}-{seed}.pt", model, encoder, metadata)

    answers = load_partition(data, "answers")
    for row in trials:
        row["metrics"] = grade(answers, row["predictions"])
        if "uniform" in row:
            row["uniform_routing_accuracy"] = grade(answers, row.pop("uniform"))["accuracy"]

    grouped = {}
    for key, question in zip(answers, questions, strict=True):
        if key.skill in {"paraphrase", "negation", "subject", "time", "condition"}:
            grouped.setdefault((key.skill, key.group), []).append((key, question))
    geometry = {}
    for (skill, _), ((left_key, left), (right_key, right)) in grouped.items():
        rows = geometry.setdefault(skill, {"action": [], "correct_outcome": []})
        rows["action"].append(float(exam_vectors[left.actions[0]] @ exam_vectors[right.actions[0]]))
        rows["correct_outcome"].append(float(
            exam_vectors[left.options[0][left_key.steps[0]]] @ exam_vectors[right.options[0][right_key.steps[0]]]
        ))

    summary = []
    for arm, budget in dict.fromkeys((row["arm"], row["budget"]) for row in trials):
        rows = [row for row in trials if (row["arm"], row["budget"]) == (arm, budget)]
        group = [row["metrics"] for row in rows]
        summary.append({
            "arm": arm, "budget": budget, "accuracy": mean(row["accuracy"] for row in group),
            "seed_accuracies": [row["accuracy"] for row in group],
            "by_skill": {skill: mean(row["by_skill"][skill]["accuracy"] for row in group) for skill in group[0]["by_skill"]},
            "both_correct": {skill: mean(row["both_correct"][skill] for row in group) for skill in group[0]["both_correct"]},
            **{key: mean(row[key] for row in group)
               for key in ("composition_intermediate", "composition_final", "composition_all_steps")},
            "mean_effective_slots": mean(row["routing"]["effective_slots"] for row in rows)
            if "routing" in rows[0] else None,
            "uniform_routing_accuracy": mean(row["uniform_routing_accuracy"] for row in rows)
            if "uniform_routing_accuracy" in rows[0] else None,
        })

    examples = []
    largest = max(budgets)
    for skill in ("paraphrase", "negation", "subject", "time", "condition", "transfer", "composition", "outside_knowledge"):
        for answer, question in zip(answers, questions, strict=True):
            if answer.skill != skill:
                continue
            chosen = {row["arm"]: row["predictions"][question.id] for row in trials
                      if row["seed"] == seeds[0] and row["budget"] == largest}
            if any(tuple(value) != tuple(answer.steps) for value in chosen.values()):
                examples.append(asdict(question) | {"skill": skill, "expected": answer.steps, "predictions": chosen})
                break

    return {
        "experiment": "007 · LTM latent transition engine", "schema": 1, "manifest": manifest,
        "budgets": budgets, "seeds": seeds, "steps": steps, "batch_size": batch_size, "loss_weights": asdict(weights),
        "questions": len(questions), "in_domain_questions": sum(a.skill != "outside_knowledge" for a in answers),
        "composition_questions": sum(a.skill == "composition" for a in answers),
        "training_payload_bytes": {
            budget: bytes_used([{key: value for key, value in asdict(row).items()
                                 if key not in {"lesson", "source", "principle", "conditions", "principle_source"}}
                                for row in episodes[:budget]])
            for budget in budgets
        },
        "reviewed_teacher_bytes": bytes_used({name: lesson.texts() for name, lesson in lessons.items()}),
        "encoder": encoder.identity, "encoding_cost": {
            "training": training_cost, "theory_and_corruption": theory_cost, "exam": exam_cost, "rule_prototypes": rule_cost,
        },
        "encoder_pair_cosines": {skill: {name: mean(values) for name, values in row.items()}
                                 for skill, row in geometry.items()},
        "rule_program_sha256": digest(RULE_EFFECTS),
        "teacher": collection["teacher"], "teacher_prompt_version": collection["prompt_version"],
        "teacher_prompt_sha256": collection["prompt_sha256"], "teacher_cost": cost(collection),
        "lesson_sha256": digest(collection), "review_sha256": digest(review),
        "teacher_audit": {
            "reviewed": len(review["principles"]), "accepted": len(lessons), "rows": review["principles"],
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine(),
                        "threads": torch.get_num_threads()},
        "summary": summary, "trials": trials, "error_examples": examples,
        "answer_key": [asdict(row) for row in answers],
    }
