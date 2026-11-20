"""Learn and combine independently verified principles in Experiment 006."""

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import platform
from statistics import mean, median
from time import perf_counter, perf_counter_ns, process_time

import torch

from .composition_split import CASE_BUDGET, DEPTHS, PAIRS_PER_DEPTH, coverage_first_problems, examination, training_problems
from .composition_student import EPOCHS, TransitionStudent, decode_tables, execute, fit, predict
from .composition_teacher import DATASET, verify
from .composition_world import SCALES, STATES, Principle, Problem, annotations_size, digest, library, trace
from .distillation import SEEDS, write_json

Predictor = Callable[[list[Problem]], list[tuple[int, ...]]]
CASE_ARMS = {"examples": training_problems, "examples_coverage_first": coverage_first_problems}
COMPARISONS = (*CASE_ARMS, "learned_principles", "stored_principles")


def encode(problem: Problem) -> list:
    return [problem.state, list(problem.rules)]


def scores(problems: list[Problem], truth: list[tuple[int, ...]], predicted: list[tuple[int, ...]]) -> dict:
    answer = [expected[-1] == actual[-1] for expected, actual in zip(truth, predicted, strict=True)]
    history = [expected == actual for expected, actual in zip(truth, predicted, strict=True)]
    by_depth = {}
    for depth in DEPTHS:
        indices = [i for i, problem in enumerate(problems) if len(problem.rules) == depth]
        paired = list(zip(indices[::2], indices[1::2], strict=True))
        reordered = [(i, j) for i, j in paired if set(problems[i].rules) == set(problems[j].rules)]
        by_depth[str(depth)] = {
            "questions": len(indices), "decisions_correct": sum(answer[i] for i in indices),
            "traces_correct": sum(history[i] for i in indices),
            "changing_pairs_correct": sum(answer[i] and answer[j] for i, j in paired),
            "reordered_pairs": len(reordered), "reordered_pairs_correct": sum(answer[i] and answer[j] for i, j in reordered),
        }
    gold_states = [row[-1] for row in truth]
    return {
        "decisions_correct": sum(answer), "traces_correct": sum(history),
        "changing_pairs_correct": sum(a and b for a, b in zip(answer[::2], answer[1::2], strict=True)),
        "by_depth": by_depth,
        "majority_answer_count": max(gold_states.count(state) for state in range(len(STATES))),
        "predictions_sha256": digest(predicted),
    }


def grade(rules: list[Principle], problems: list[Problem], infer: Predictor, measure: bool) -> dict:
    outputs = infer(problems)
    result = scores(problems, [trace(rules, problem) for problem in problems], outputs)
    if measure:
        batch_us = []
        for _ in range(9):
            started = perf_counter_ns()
            infer(problems)
            batch_us.append((perf_counter_ns() - started) / 1000)
        single_us = []
        for problem in problems[::max(1, len(problems) // 64)]:
            started = perf_counter_ns()
            infer([problem])
            single_us.append((perf_counter_ns() - started) / 1000)
        result["latency"] = {"batch_us_per_question": median(batch_us) / len(problems),
                             "single_question_us": median(single_us)}
    return result


def case_lessons(
    rules: list[Principle], seed: int, generate: Callable[[int, int], list[Problem]],
) -> tuple[list[Problem], torch.Tensor, dict]:
    wall, cpu = perf_counter(), process_time()
    train = generate(len(rules), seed)
    answers = [trace(rules, problem)[-1] for problem in train]
    serialized = [[*encode(problem), answer] for problem, answer in zip(train, answers, strict=True)]
    return train, torch.tensor(answers), {
        "case_count": len(train), "annotation_bytes": annotations_size(serialized),
        "atomic_cases": sum(len(problem.rules) == 1 for problem in train),
        "rules_seen": len({index for problem in train for index in problem.rules}),
        "sha256": digest(serialized),
        "preparation_wall_seconds": perf_counter() - wall, "preparation_cpu_seconds": process_time() - cpu,
    }


def run_student(seed: int, size: int, problems: list[Problem], rules: list[Principle],
                labels: torch.Tensor, training: list[Problem] | None, measure: bool) -> tuple[dict, Predictor]:
    wall, cpu = perf_counter(), process_time()
    model = TransitionStudent(size, seed)
    fit(model, labels, training)
    cost = {"fit_wall_seconds": perf_counter() - wall, "fit_cpu_seconds": process_time() - cpu}

    def infer(questions: list[Problem]) -> list[tuple[int, ...]]:
        return predict(model, questions)

    result = grade(rules, problems, infer, measure)
    result.update({"seed": seed, "parameters": sum(parameter.numel() for parameter in model.parameters())})
    if measure:
        result.update(cost)
    return result, infer


def run_scale(rules: list[Principle], tables: list[str], audit: dict, seeds: tuple[int, ...], pairs: int,
              measure: bool, *, anchors: list[Problem] | None = None) -> dict:
    size = len(rules)
    exam = [problem for pair in examination(rules, size, pairs) for problem in pair]
    wall, cpu = perf_counter(), process_time()
    stored = decode_tables(tables)
    stored_wall, stored_cpu = perf_counter() - wall, process_time() - cpu
    wall, cpu = perf_counter(), process_time()
    compiled = torch.tensor(stored)
    prepare_wall, prepare_cpu = perf_counter() - wall, process_time() - cpu
    results = {name: [] for name in COMPARISONS}

    def infer_stored(questions: list[Problem]) -> list[tuple[int, ...]]:
        return execute(stored, questions)

    stored_result = grade(rules, exam, infer_stored, measure)
    if anchors is not None:
        stored_result["anchors"] = grade(rules, anchors, infer_stored, measure)
    results["stored_principles"].append(stored_result)

    for seed in seeds:
        for arm, generate in CASE_ARMS.items():
            training, labels, cost = case_lessons(rules, seed, generate)
            row, infer = run_student(seed, size, exam, rules, labels, training, measure)
            row["case_supervision"] = cost if measure else {
                key: value for key, value in cost.items() if not key.startswith("preparation_")
            }
            if anchors is not None:
                row["anchors"] = grade(rules, anchors, infer, measure)
            results[arm].append(row)
        row, infer = run_student(seed, size, exam, rules, compiled, None, measure)
        if anchors is not None:
            row["anchors"] = grade(rules, anchors, infer, measure)
        results["learned_principles"].append(row)

    report = {
        "principles": size, "exam_questions": len(exam), "pairs_per_depth": pairs,
        "exam_sha256": digest([encode(problem) for problem in exam]),
        "principle_annotation_bytes": annotations_size(tables),
        "atomic_transition_labels": size * len(STATES),
        "teacher": audit if measure else {key: value for key, value in audit.items() if key != "verification_wall_seconds"},
        "results": results,
    }
    if measure:
        report["verified_setup_wall_seconds"] = prepare_wall
        report["verified_setup_cpu_seconds"] = prepare_cpu
        report["stored_index_wall_seconds"] = stored_wall
        report["stored_index_cpu_seconds"] = stored_cpu
        teacher_overhead = audit["teacher_wall_seconds"] + audit["verification_wall_seconds"] + stored_wall
        report["median_ready_wall_seconds"] = {
            **{
                arm: median(row["case_supervision"]["preparation_wall_seconds"] + row["fit_wall_seconds"]
                            for row in results[arm])
                for arm in CASE_ARMS
            },
            "learned_principles": teacher_overhead + prepare_wall
                                  + median(row["fit_wall_seconds"] for row in results["learned_principles"]),
            "stored_principles": teacher_overhead,
        }
    return report


def verified_cohort(record: dict, world: list[Principle], full_audit: dict,
                    size: int = 100) -> tuple[list[Principle], list[str], dict]:
    rejected = set(full_audit["rejected_ids"])
    candidates = sum(len(batch["programs"]) for batch in record["batches"])
    chosen = [index for index in range(candidates) if index not in rejected][:size]
    if len(chosen) != size:
        raise ValueError(f"The record needs at least {size} accepted model-authored principles.")
    end = 0
    for batch in record["batches"]:
        end += len(batch["programs"])
        if end > chosen[-1]:
            break
    tables, audit = verify(record, world, end)
    audit.update({
        "source_ids": chosen, "proposals_paid_for": len(tables), "accepted_in_paid_prefix": audit["accepted"],
        "used_teacher_principles": size, "simulator_fills": 0,
    })
    return [world[index] for index in chosen], [tables[index] for index in chosen], audit


def run(record: dict, sizes: tuple[int, ...] = SCALES, seeds: tuple[int, ...] = SEEDS,
        *, pairs: int = PAIRS_PER_DEPTH, measure: bool = False, include_verified: bool = True) -> dict:
    world = library()
    anchors = [problem for pair in examination(world, SCALES[0], pairs) for problem in pair]
    if measure:
        # Warm one-time optimizer initialization before timing either learned model.
        torch.optim.Adam([torch.zeros(1, requires_grad=True)])
    results = {}
    for size in sizes:
        tables, audit = verify(record, world, size)
        results[str(size)] = run_scale(world[:size], tables, audit, seeds, pairs, measure,
                                      anchors=anchors if size != SCALES[0] else None)
    report = {
        "dataset": DATASET, "environment": {"platform": platform.platform(), "torch": str(torch.__version__)},
        "teacher_model": record["model"], "teaching_sha256": digest(record), "seeds": list(seeds),
        "optimizer_steps": EPOCHS, "case_budget": CASE_BUDGET, "scales": results,
    }
    if include_verified:
        count = sum(len(batch["programs"]) for batch in record["batches"])
        _, audit = verify(record, world, count)
        selected, taught, cost = verified_cohort(record, world, audit)
        report["verified_only_100"] = run_scale(selected, taught, cost, seeds, pairs, measure)
    return report


def print_summary(report: dict) -> None:
    groups = report["scales"].copy()
    if "verified_only_100" in report:
        groups["verified_only_100"] = report["verified_only_100"]
    for size, group in groups.items():
        print(f"{size} principles · {group['exam_questions']} exams")
        for arm, rows in group["results"].items():
            print(f"  {arm:22s} {mean(row['decisions_correct'] for row in rows):6.1f}/{group['exam_questions']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teaching", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", choices=SCALES, default=SCALES)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--pairs-per-depth", type=int, choices=range(1, PAIRS_PER_DEPTH + 1),
                        metavar="COUNT", default=PAIRS_PER_DEPTH)
    parser.add_argument("--measure-time", action="store_true")
    parser.add_argument("--prefixes-only", action="store_true", help="skip the separate 100-rule verified-only cohort")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        record = json.loads(args.teaching.read_text())
        result = run(record, tuple(args.sizes), tuple(args.seed) if args.seed else SEEDS,
                     pairs=args.pairs_per_depth, measure=args.measure_time, include_verified=not args.prefixes_only)
        if args.output:
            write_json(args.output, result)
        print_summary(result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Composition experiment stopped: {error}\n")


if __name__ == "__main__":
    main()
