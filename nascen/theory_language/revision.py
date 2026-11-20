"""Audit the original choices and run the balanced transition follow-up."""

import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import process_time

import torch

from .dataset import digest, exam, read_json, training_pool, write_json
from .encoder import MultilingualE5, encode_partition
from .evaluation import encode_questions, exam_texts, grade, predict_copy, route_statistics
from .revision_model import VARIANTS, fit_revision
from .teacher import load_reviewed
from .trainer import EncodedEpisodes, episode_texts, lesson_texts, theory_targets
from .world import ACTIONS, EXAM_ITEMS, STATES, TRAIN_ITEMS, describe_state

FRESH_ITEMS = {
    "physical": ("나무 쐐기", "황동 추", "도자기 접시", "황색 손수레", "접이식 상자", "대리석 판"),
    "social": ("훈련 지도", "회전 표지판", "방수 시계", "신호용 깃발", "접이식 들것", "공구 가방"),
}
CONFIRMATION_ITEMS = {
    "physical": ("납작한 주석판", "소나무 기둥", "세라믹 공", "알루미늄 테두리", "청동 추", "직육면체 모형"),
    "social": ("훈련용 나침반", "비상용 담요", "측정용 줄자", "접이식 우비", "신호 봉", "현장 기록판"),
}
CONFIRMATION_CHAINS = (
    ("physical", (True, True, False), ("block", "unblock", "push")),
    ("physical", (False, False, False), ("push", "unblock", "advance")),
    ("physical", (True, True, False), ("block", "advance")),
    ("social", (False, False, False), ("urgent_before", "owner_grants", "use_in")),
    ("social", (False, False, False), ("peer_grants", "use_in")),
    ("social", (False, False, False), ("urgent_now", "urgent_before", "use_in")),
    ("social", (True, False, False), ("peer_grants", "use_in", "use_out")),
)
DEFAULT_ENCODER = Path(".nascen/teacher-assets/multilingual-e5-small")
CONTRASTS = {"negation", "condition", "subject", "time"}


def candidate_only(encoded):
    return {
        identifier: values for ids, _, _, options in encoded
        for identifier, values in zip(ids, (options @ options.transpose(-1, -2)).sum(-1).argmax(-1).tolist(), strict=True)
    }


def audit_original(encoder):
    questions, answers = exam()
    scope = [(q, a) for q, a in zip(questions, answers, strict=True) if a.skill != "outside_knowledge"]
    questions = tuple(q for q, _ in scope)
    vectors, _ = encode_partition(encoder, exam_texts(questions))
    medoid = candidate_only(encode_questions(questions, vectors))
    states = {describe_state(world, item, state, style): state
              for world in ACTIONS for item in EXAM_ITEMS[world] for state in STATES for style in range(3)}
    text_correct, exact_correct, count = 0, 0, 0
    for question, key in scope:
        for opts, expected, predicted in zip(question.options, key.steps, medoid[key.id], strict=True):
            candidate_states = [states[option] for option in opts]
            distances = [
                sum(sum(a != b for a, b in zip(left, right, strict=True)) for right in candidate_states)
                for left in candidate_states
            ]
            exact_correct += distances.index(min(distances)) == expected
            text_correct += predicted == expected
            count += 1
    return {"steps": count, "exact_choice_geometry_correct": exact_correct, "e5_candidate_only_correct": text_correct}


@torch.no_grad()
def predict_revision(model, encoded, *, uniform=False):
    predictions = {}
    routes = []
    for ids, before, actions, candidates in encoded:
        predicted, _, _, routing, prior, gate = model(before, actions, uniform=uniform)
        choices = model.outcome_scores(predicted, candidates, prior, gate).argmax(-1).tolist()
        predictions.update(zip(ids, choices, strict=True))
        routes.append(routing.flatten(0, 1))
    return predictions, torch.cat(routes)


def score(questions, answers, outputs):
    result = grade(answers, outputs)
    changed, unchanged = [], []
    paired = {}
    for question, key in zip(questions, answers, strict=True):
        if key.skill == "outside_knowledge":
            continue
        prior = question.before
        for opts, expected, predicted in zip(question.options, key.steps, outputs[key.id], strict=True):
            target = opts[expected]
            (changed if target != prior else unchanged).append(expected == predicted)
            prior = target
        if key.skill in CONTRASTS:
            paired.setdefault(key.group, []).append(outputs[key.id][-1] == key.steps[-1])
    return {
        "accuracy": result["accuracy"],
        "outside_knowledge_accuracy": result["by_skill"]["outside_knowledge"]["accuracy"],
        "changed_accuracy": mean(changed), "unchanged_accuracy": mean(unchanged),
        "changed_steps": len(changed), "unchanged_steps": len(unchanged),
        "counterfactual_both_correct": mean(all(pair) for pair in paired.values()),
        "composition_all_steps": result["composition_all_steps"],
        "transfer": result["by_skill"]["transfer"]["accuracy"],
    }


def run(encoder, collection, review, budgets, seeds, steps, batch_size, phase):
    items = FRESH_ITEMS if phase == "diagnosis" else CONFIRMATION_ITEMS
    variants = ("A", "B", "C", "D", "E") if phase == "diagnosis" else ("A", "B", "F")
    for world in items:
        if set(items[world]) & (set(TRAIN_ITEMS[world]) | set(EXAM_ITEMS[world])):
            raise ValueError("The revised exam must use new objects.")
    audit = audit_original(encoder)
    # Materialize separate raw exam inputs, then encode them independently of the training partition.
    questions, keys = exam(full_options=True, items=items,
                           chains=CONFIRMATION_CHAINS if phase == "confirmation" else None)
    test_vectors, test_cost = encode_partition(encoder, exam_texts(questions))
    encoded = encode_questions(questions, test_vectors)
    episodes = training_pool(full_options=True)[:max(budgets)]
    train_vectors, train_cost = encode_partition(encoder, episode_texts(episodes))
    lessons = load_reviewed(collection, review)
    theory_vectors, theory_cost = encode_partition(encoder, lesson_texts(lessons) if "E" in variants else ())
    trials = []
    for name, outputs in (("candidate-only", candidate_only(encoded)), ("E5-copy", predict_copy(encoded))):
        trials.append({"arm": name, "budget": None, "seed": None, "predictions": outputs,
                       "metrics": score(questions, keys, outputs)})
    for budget in budgets:
        cohort = episodes[:budget]
        data = EncodedEpisodes.build(cohort, train_vectors)
        for variant in variants:
            for seed in seeds:
                targets = theory_targets(cohort, lessons, theory_vectors, "principle", seed)[0] if variant == "E" else None
                start = process_time()
                model = fit_revision(data, encoder.width, variant, seed, steps, batch_size, targets)
                seconds = process_time() - start
                outputs, routes = predict_revision(model, encoded)
                uniform, _ = predict_revision(model, encoded, uniform=True)
                trials.append({
                    "arm": variant, "budget": budget, "seed": seed, "cpu_seconds": seconds,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "metrics": score(questions, keys, outputs), "predictions": outputs,
                    "routing": route_statistics(routes),
                    "uniform_routing_accuracy": grade(keys, uniform)["accuracy"],
                })
    summary = []
    for arm, budget in dict.fromkeys((row["arm"], row["budget"]) for row in trials):
        group = [row for row in trials if row["arm"] == arm and row["budget"] == budget]
        summary.append({
            "arm": arm, "budget": budget,
            **{key: mean(r["metrics"][key] for r in group) for key in group[0]["metrics"]},
            "seed_accuracies": [row["metrics"]["accuracy"] for row in group],
            "effective_slots": mean(row["routing"]["effective_slots"] for row in group) if "routing" in group[0] else None,
            "uniform_routing_accuracy": mean(row["uniform_routing_accuracy"] for row in group)
            if "routing" in group[0] else None,
        })
    return {
        "experiment": "008 · Balanced outcome choices and action-sensitive transitions",
        "phase": phase,
        "original_choice_audit": audit,
        "budgets": budgets, "seeds": seeds, "steps": steps, "batch_size": batch_size,
        "train_sha256": digest([asdict(row) for row in episodes]),
        "exam_sha256": digest([asdict(row) for row in questions]),
        "answer_sha256": digest([asdict(row) for row in keys]),
        "lesson_sha256": digest(collection), "review_sha256": digest(review),
        "encoder": encoder.identity,
        "encoding_cost": {"training": train_cost, "exam": test_cost, "theory": theory_cost},
        "variants": {key: {"projected_outcomes": VARIANTS[key][0], "balanced_routing": VARIANTS[key][1],
                           "teacher": key == "E", "learned_change_gate": key == "F"} for key in variants},
        "answer_key": [asdict(row) for row in keys], "summary": summary, "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("diagnosis", "confirmation"), default="diagnosis")
    parser.add_argument("--budgets", type=int, nargs="+", default=(100, 300, 1000))
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 22, 42))
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lessons", type=Path, default=Path("docs/experiment-007-theory-lessons.json"))
    parser.add_argument("--review", type=Path, default=Path("docs/experiment-007-theory-review.json"))
    args = parser.parse_args()
    if not all(1 <= budget <= len(training_pool()) for budget in args.budgets) or args.steps < 1 or args.batch_size < 1:
        parser.error("Choose valid positive training budgets, step and batch counts.")
    if len(set(args.budgets)) != len(args.budgets) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Budgets and seeds must be distinct.")
    torch.set_num_threads(1)
    report = run(MultilingualE5(args.encoder), read_json(args.lessons), read_json(args.review),
                 args.budgets, args.seeds, args.steps, args.batch_size, args.phase)
    write_json(args.output, report)
    for row in report["summary"]:
        print(f"{row['arm']} {row['budget']} · {row['accuracy']:.1%}", flush=True)


if __name__ == "__main__":
    main()
