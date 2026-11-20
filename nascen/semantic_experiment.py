"""Run Experiment 007, comparing four approaches on the same Korean exam."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from statistics import mean
from tempfile import NamedTemporaryFile
from time import perf_counter, process_time

import torch

from .concept_model import ConceptModel, continuation_loss, relation_loss
from .principle_learner import PrincipleTargets, principle_texts
from .semantic_dataset import PRINCIPLES, RELATION_PAIRS, RELATIONS, budget_scenes, digest, splits, write_json
from .semantic_reasoning import RULE_PROGRAM, context_vectors, rule_choice, student_scores
from .semantic_teacher import load_lessons
from .sentence_encoder import MultilingualE5, SentenceVectors, SurfaceTokens

SEEDS = (11, 22, 42)
BUDGETS = (1, 2, 3, 4)
SKILLS = ("paraphrase", "exception", "transfer", "composition")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def input_bytes(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def scene_texts(scenes):
    for row in scenes:
        yield from row.context
        yield from row.options


def training_tensors(scenes, pairs, vectors):
    contexts = torch.stack([context_vectors(row.context, vectors) for row in scenes])
    options = torch.stack([context_vectors(row.options, vectors) for row in scenes])
    labels = torch.tensor([row.answer for row in scenes])
    relations = None
    if pairs:
        relations = (
            torch.stack([vectors[pair.first] for pair in pairs]),
            torch.stack([vectors[pair.second] for pair in pairs]),
            torch.tensor([RELATIONS.index(pair.reference) for pair in pairs]),
        )
    return contexts, options, labels, relations


def fit(inputs, width, seed, epochs, principles=None):
    contexts, options, labels, relations = inputs
    student = ConceptModel(width, seed)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.005)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = continuation_loss(student(contexts), options, labels)
        if relations is not None:
            loss = loss + 0.25 * relation_loss(student, *relations)
        if principles is not None:
            loss = loss + 0.25 * principles.loss(student, contexts)
        loss.backward()
        optimizer.step()
    return student.eval()


def encode_exam(exam, vectors):
    # This is the complete inference input. The answer key stays with the grader.
    return [(row.context, context_vectors(row.context, vectors), context_vectors(row.options, vectors)) for row in exam]


def judge(exam, predictions):
    results = [choice == row.answer for row, choice in zip(exam, predictions, strict=True)]
    scores = {
        "accuracy": mean(results),
        "coverage": mean(choice is not None for choice in predictions),
    }
    for skill in SKILLS:
        scores[skill] = mean(right for row, right in zip(exam, results, strict=True) if row.skill == skill)
    for topic in ("physical", "social"):
        scores[f"composition_{topic}"] = mean(right for row, right in zip(exam, results, strict=True)
                                               if row.skill == "composition" and row.topic == topic)
    for skill in ("paraphrase", "exception"):
        pairs = {}
        for row, right in zip(exam, results, strict=True):
            if row.skill == skill:
                pairs.setdefault(row.pair, []).append(right)
        scores[f"{skill}_both_correct"] = mean(all(pair) for pair in pairs.values())
    failures = [
        {"id": row.id, "predicted": choice}
        for row, choice, right in zip(exam, predictions, results, strict=True) if not right
    ]
    return scores, failures


def encode_inputs(encoder, train, exam, pairs, lessons):
    start = process_time()
    vectors = SentenceVectors(encoder, [
        *scene_texts(train), *(text for pair in pairs for text in (pair.first, pair.second)), *principle_texts(lessons),
    ])
    train_seconds = process_time() - start
    start = process_time()
    exam_vectors = SentenceVectors(encoder, scene_texts(exam))
    exam_inputs = encode_exam(exam, exam_vectors)
    return vectors, exam_inputs, {
        "cached_train_cpu_seconds": train_seconds,
        "cached_exam_cpu_seconds": process_time() - start,
        "unique_training_sentences": len(vectors.vectors), "unique_exam_sentences": len(exam_vectors.vectors),
    }


def supply_cost(data: dict, keys) -> dict:
    batches = [attempt for key in keys for attempt in data["cost"][key]]
    tokens = {}
    for field in ("prompt_tokens", "completion_tokens"):
        values = [entry["usage"].get(field) for entry in batches]
        tokens[field] = sum(values) if all(isinstance(value, int) for value in values) else None
    return {"requests": len(batches), "wall_seconds": sum(entry["wall_seconds"] for entry in batches), **tokens}


def run(teacher: dict, review: dict, encoder, *, budgets=BUDGETS, seeds=SEEDS, epochs=100,
        checkpoint: Path | None = None) -> dict:
    lessons, audit = load_lessons(teacher)
    if review.get("lessons_sha256") != digest(teacher):
        raise ValueError("The independent principle review must match this teacher collection.")
    reviewed = review["principles"]
    if reviewed.keys() != lessons.keys() or any(type(row.get("accepted")) is not bool or not row.get("reason")
                                               for row in reviewed.values()):
        raise ValueError("Review every teacher principle and record a reason.")
    accepted = {key: lesson for key, lesson in lessons.items() if reviewed[key]["accepted"]}
    if not accepted:
        raise ValueError("No teacher principles survived independent review.")
    # Compare against independent references and discard disagreements.
    pairs = tuple(pair for pair in RELATION_PAIRS if teacher["relations"][pair.id] == pair.reference)
    train, exam = splits()
    surface = SurfaceTokens(encoder.width)
    surface_vectors, surface_exam, surface_cost = encode_inputs(surface, train, exam, pairs, {})
    concept_vectors, concept_exam, concept_cost = encode_inputs(encoder, train, exam, pairs, accepted)

    start = process_time()
    rule_predictions = [rule_choice(context, options, accepted, concept_vectors) for context, _, options in concept_exam]
    rule_seconds = process_time() - start
    rule_scores, rule_failures = judge(exam, rule_predictions)
    relation_keys = tuple(key for key in teacher["cost"] if key.startswith("relations-"))
    principle_keys = tuple(PRINCIPLES)
    relation_bytes = input_bytes([asdict(pair) for pair in pairs])
    lesson_bytes = input_bytes({key: asdict(lesson) for key, lesson in accepted.items()})
    trials = [{
        "arm": "D", "budget_groups_per_rule": None, "seed": None, "training_scenes": 0,
        "principle_bytes": lesson_bytes,
        "training_payload_bytes": lesson_bytes + input_bytes({key: RULE_PROGRAM[key] for key in accepted}),
        "trained_parameters": 0, "train_cpu_seconds": 0, "exam_cpu_seconds": rule_seconds,
        "metrics": rule_scores, "failures": rule_failures,
    }]
    for budget in budgets:
        scenes = budget_scenes(train, budget)
        base_bytes = input_bytes([{"context": row.context, "options": row.options, "answer": row.answer} for row in scenes])
        association_bytes = input_bytes([row.principle if row.principle in accepted else None for row in scenes])
        surface_inputs = training_tensors(scenes, pairs, surface_vectors)
        concept_inputs = training_tensors(scenes, pairs, concept_vectors)
        auxiliary = PrincipleTargets(accepted, scenes, concept_vectors)
        for arm, inputs, exam_inputs, principles in (
            ("A", surface_inputs, surface_exam, None),
            ("B", concept_inputs, concept_exam, None),
            ("C", concept_inputs, concept_exam, auxiliary),
        ):
            extra_bytes = lesson_bytes + association_bytes if principles is not None else 0
            for seed in seeds:
                start = process_time()
                student = fit(inputs, encoder.width, seed, epochs, principles)
                train_seconds = process_time() - start
                if checkpoint is not None and arm == "C" and budget == max(budgets) and seed == seeds[0]:
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    with NamedTemporaryFile(dir=checkpoint.parent, delete=False) as stream:
                        temporary = Path(stream.name)
                        try:
                            torch.save({
                                "schema": 1, "seed": seed, "encoder": encoder.identity, "weights": student.state_dict(),
                            }, stream)
                            stream.close()
                            temporary.replace(checkpoint)
                        finally:
                            temporary.unlink(missing_ok=True)
                start = process_time()
                predictions = [
                    int(student_scores(student, context.unsqueeze(0), options.unsqueeze(0)).argmax())
                    for _, context, options in exam_inputs
                ]
                exam_seconds = process_time() - start
                scores, failures = judge(exam, predictions)
                trials.append({
                    "arm": arm, "budget_groups_per_rule": budget, "seed": seed, "training_scenes": len(scenes),
                    "principle_bytes": extra_bytes,
                    "training_payload_bytes": base_bytes + relation_bytes + extra_bytes,
                    "trained_parameters": sum(p.numel() for p in student.parameters())
                    - (sum(p.numel() for p in student.relation.parameters()) if not pairs else 0),
                    "train_cpu_seconds": train_seconds, "exam_cpu_seconds": exam_seconds,
                    "metrics": scores, "failures": failures,
                })

    summary = []
    for arm, budget in dict.fromkeys((trial["arm"], trial["budget_groups_per_rule"]) for trial in trials):
        group = [trial for trial in trials if trial["arm"] == arm and trial["budget_groups_per_rule"] == budget]
        first = group[0]
        summary.append({key: first[key] for key in (
            "arm", "budget_groups_per_rule", "training_scenes", "training_payload_bytes", "trained_parameters",
        )} | {
            "metrics": {key: mean(row["metrics"][key] for row in group) for key in first["metrics"]},
            "seed_accuracies": [row["metrics"]["accuracy"] for row in group],
            "mean_train_cpu_seconds": mean(row["train_cpu_seconds"] for row in group),
        })
    return {
        "experiment": "007 — Korean sentence semantics and principles", "schema": 1,
        "dataset": teacher["dataset"], "training_sha256": teacher["training_sha256"],
        "exam_sha256": digest([asdict(row) for row in exam]),
        "lessons_sha256": digest(teacher), "principle_review_sha256": digest(review),
        "exam_questions": len(exam), "exam_groups": len({row.group for row in exam}),
        "chance_accuracy": mean(1 / len(row.options) for row in exam),
        "exam_inputs": "raw Korean context and unlabelled candidate continuations",
        "student_inference_dependencies": "student weights and its frozen sentence encoder",
        "epochs": epochs, "seeds": seeds,
        "teacher": teacher["teacher"], "teacher_prompt": teacher["prompt_version"],
        "teacher_prompt_sha256": teacher["prompt_sha256"],
        "principle_references_sha256": teacher["principle_references_sha256"],
        "rule_program_sha256": digest(RULE_PROGRAM),
        "teacher_cost": {
            "A": supply_cost(teacher, relation_keys), "B": supply_cost(teacher, relation_keys),
            "C": supply_cost(teacher, (*relation_keys, *principle_keys)), "D": supply_cost(teacher, principle_keys),
        },
        "encoder": {
            "A": surface.identity | surface_cost, "B/C/D": encoder.identity | concept_cost,
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine()},
        "relation_audit": audit, "principle_review": reviewed,
        "summary": summary, "trials": trials,
        "answer_key": [
            {"id": row.id, "topic": row.topic, "skill": row.skill, "context": row.context,
             "options": row.options, "expected": row.answer}
            for row in exam
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lessons", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True, help="local multilingual-e5-small snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="save arm C at the largest budget and first seed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--budgets", type=int, choices=BUDGETS, nargs="+", default=BUDGETS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.budgets)) != len(args.budgets):
        parser.error("--seeds and --budgets must not contain duplicates")
    paths = [args.lessons, args.review, args.output, *([args.checkpoint] if args.checkpoint else [])]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("lesson, review, report and checkpoint paths must be distinct")
    torch.set_num_threads(1)
    try:
        start = process_time()
        encoder = MultilingualE5(args.encoder)
        load_seconds = process_time() - start
        result = run(read_json(args.lessons), read_json(args.review), encoder,
                     budgets=args.budgets, seeds=args.seeds, epochs=args.epochs, checkpoint=args.checkpoint)
        result["encoder"]["B/C/D"]["load_cpu_seconds"] = load_seconds
        write_json(args.output, result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Experiment 007 stopped: {error}\n")
    for row in result["summary"]:
        accuracy = row["metrics"]["accuracy"]
        print(f'{row["arm"]} · {row["training_scenes"]} scenes · {accuracy:.1%} accuracy')


if __name__ == "__main__":
    main()
