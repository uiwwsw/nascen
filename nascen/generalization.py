"""Experiment 003 — test unfamiliar wording and held-out condition combinations offline."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import platform
from statistics import mean

import torch

from . import __version__
from .consent import DECISIONS, PRINCIPLES, SETTINGS, URGENCY_TEXT, Case, Question, compile_question, reference, split
from .distillation import EPOCHS, SEEDS, features, fit, predict, write_json
from .teacher import Lesson, read_lessons

ARM = "scratch_verified_principles"


@dataclass(frozen=True)
class Condition:
    alternative: bool
    limited: bool

    def matches(self, case: Case) -> bool:
        return case.owner == "unavailable" and (case.alternative, case.limited) == (self.alternative, self.limited)

    def probes(self) -> list[Case]:
        return [Case(place, "unavailable", urgent, self.alternative, self.limited)
                for place, urgent in product(SETTINGS, (False, True))]


CONDITIONS = tuple(Condition(alternative, limited) for alternative, limited in product((False, True), repeat=2))


def new_urgency_wording(question: Question) -> str:
    old = URGENCY_TEXT[question.case.urgent][0]
    replacement = (
        "Someone could suffer serious harm right now"
        if question.case.urgent else "Nobody would be in immediate danger if they waited"
    )
    return question.text.replace(old, replacement, 1)


def narrative(case: Case) -> str:
    owner = {
        "agreed": "the owner said they could use the item",
        "refused": "the owner turned down the request",
        "reachable": "they could ask the owner, who has yet to say yes or no",
        "unavailable": "there is no time to reach the owner",
    }[case.owner]
    danger = "Waiting could result in serious harm right away" if case.urgent else "It is safe to wait"
    substitute = ("They can use a permitted substitute that works just as well" if case.alternative
                  else "No permitted substitute can get the same result")
    scope = ("They would use it only for what is immediately needed" if case.limited
             else "They want to use it for additional purposes, too")
    return f"A player at the {case.setting} wants to borrow an item, and {owner}. {danger}. {substitute}. {scope}. What should they do?"


def compile_inputs(texts: Sequence[str]) -> list[Case | None]:
    inputs = []
    for text in texts:
        try:
            inputs.append(compile_question(text))
        except ValueError:
            inputs.append(None)
    return inputs


def verified_students(rows: list[tuple[Question, Lesson]], seeds: tuple[int, ...]) -> tuple[list[dict], list[Lesson]]:
    verified = [(question, lesson) for question, lesson in rows
                if reference(question.case) == (lesson.decision, lesson.principle)]
    if not verified:
        raise ValueError("A fold has no teacher lessons that agree with the exercise rubric.")
    inputs = features([compile_question(question.text) for question, _ in verified])
    labels = [lesson for _, lesson in verified]
    decisions = torch.tensor([DECISIONS.index(lesson.decision) for lesson in labels])
    principles = torch.tensor([tuple(PRINCIPLES).index(lesson.principle) for lesson in labels])
    return [fit(ARM, seed, inputs, decisions, principles) for seed in seeds], labels


def grade_cases(run: dict, expected: list[Case], observed: list[Case | None]) -> dict:
    if len(expected) != len(observed):
        raise ValueError("Missing inputs for some evaluation cases.")
    valid = [(index, case) for index, case in enumerate(observed) if case is not None]
    predictions = dict(zip(
        (index for index, _ in valid), predict(run, features([case for _, case in valid])) if valid else (), strict=True,
    ))
    gold = list(map(reference, expected))
    return {
        "seed": run["seed"], "questions": len(expected), "answered": len(valid),
        "fact_accuracy": mean(left == right for left, right in zip(expected, observed, strict=True)),
        "decision_accuracy": mean(predictions.get(index, (None, None))[0] == answer
                                  for index, (answer, _) in enumerate(gold)),
        "joint_accuracy": mean(predictions.get(index) == label for index, label in enumerate(gold)),
        "predictions": [
            {"decision": predictions.get(index, (None, None))[0], "principle": predictions.get(index, (None, None))[1]}
            for index in range(len(expected))
        ],
    }


def case_record(case: Case) -> dict:
    decision, principle = reference(case)
    return asdict(case) | {"expected_decision": decision, "expected_principle": principle}


def surface_trials(rows: list[tuple[Question, Lesson]], split_seed: int, seeds: tuple[int, ...]) -> dict:
    students, lessons = verified_students(rows, seeds)
    _, paired = split(split_seed)
    questions = [left for left, _ in paired]
    expected = [question.case for question in questions]
    variants = {
        "supported_text": [question.text for question in questions],
        "new_urgency_wording": [new_urgency_wording(question) for question in questions],
        "new_narrative": [narrative(case) for case in expected],
    }
    trials = {}
    for variant, texts in variants.items():
        compiled = compile_inputs(texts)
        trials[variant] = {
            "texts": texts, "trials": [grade_cases(student, expected, compiled) for student in students],
        }
    trials["structured_diagnostic"] = {
        "trials": [grade_cases(student, expected, expected) for student in students],
    }
    return {
        "questions": len(expected), "training_used": [lesson.id for lesson in lessons],
        "cases": [case_record(case) for case in expected], "arms": trials,
        "summary": [
            {"input": variant, **aggregate(arm["trials"], ("answered", "fact_accuracy", "decision_accuracy", "joint_accuracy"))}
            for variant, arm in trials.items()
        ],
    }


def aggregate(trials: list[dict], metrics: tuple[str, ...]) -> dict:
    return {name: mean(trial[name] for trial in trials) for name in metrics}


def holdout(
    rows: list[tuple[Question, Lesson]], condition: Condition, seeds: tuple[int, ...],
) -> dict:
    inputs = [(question, lesson) for question, lesson in rows if not condition.matches(question.case)]
    students, lessons = verified_students(inputs, seeds)
    cases = condition.probes()
    answers = [reference(case)[0] for case in cases]
    expected_reasons = {reference(case)[1] for case in cases}
    training_reasons = {lesson.principle for lesson in lessons}
    trials = []
    for student in students:
        result = grade_cases(student, cases, cases)
        choices = [row["decision"] for row in result["predictions"]]
        result["pair_accuracy"] = mean(
            choices[index:index + 2] == answers[index:index + 2] for index in range(0, len(answers), 2)
        )
        result["decision_change_rate"] = mean(choices[index] != choices[index + 1] for index in range(0, len(answers), 2))
        trials.append(result)
    return {
        "condition": asdict(condition), "excluded_lessons": [q.id for q, _ in rows if condition.matches(q.case)],
        "training_used": [lesson.id for lesson in lessons], "cases": [case_record(case) for case in cases],
        "expected_principles_without_examples": sorted(expected_reasons - training_reasons),
        "trials": trials,
        "summary": aggregate(trials, ("decision_accuracy", "joint_accuracy", "pair_accuracy", "decision_change_rate")),
    }


def run(lesson_file: Path, seeds: tuple[int, ...] = SEEDS) -> dict:
    metadata, rows = read_lessons(lesson_file)
    folds = [holdout(rows, condition, seeds) for condition in CONDITIONS]
    conditional_summary = aggregate(
        [fold["summary"] for fold in folds], ("decision_accuracy", "joint_accuracy", "pair_accuracy", "decision_change_rate"),
    )
    return {
        "experiment": "003 — Generalization by Wording and Condition",
        "nascen_version": __version__, "dataset": metadata["dataset"], "teacher": metadata["teacher"],
        "lesson_sha256": sha256(lesson_file.read_bytes()).hexdigest(),
        "split_seed": metadata["seed"], "student_seeds": seeds, "epochs": EPOCHS,
        "teacher_available_to_exam": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine()},
        "surface": surface_trials(rows, metadata["seed"], seeds),
        "conditional_summary": conditional_summary | {"pairs_per_seed": sum(len(fold["cases"]) // 2 for fold in folds)},
        "conditional": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lessons", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed <= 2**31 - 1 for seed in args.seeds):
        parser.error("Use distinct student seeds between 0 and 2147483647.")
    torch.set_num_threads(1)
    try:
        report = run(args.lessons, tuple(args.seeds))
        if args.output:
            write_json(args.output, report)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Generalization experiment stopped: {error}\n")
    print(json.dumps({
        "surface": report["surface"]["summary"],
        "conditional_summary": report["conditional_summary"],
        "conditional": [fold["condition"] | fold["summary"] for fold in report["conditional"]],
    }, indent=2))


if __name__ == "__main__":
    main()
