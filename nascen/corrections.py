"""Experiment 004 — learn from mistakes, then evaluate on separate cases offline."""

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
from random import Random
from statistics import mean

import torch

from . import __version__
from .consent import SETTINGS, Case, Question, reference, render, split
from .distillation import EPOCHS, SEEDS, features, predict, write_json
from .generalization import case_record, grade_cases, verified_students
from .teacher import DEFAULT_PORT, TEACHING_RULES, Lesson, read_lessons, request_lessons

EXPERIMENT = "004 — Learning from Mistakes"
DEFAULT_BUDGET = 8
STRATEGIES = ("mistakes", "random")
PRACTICE_SETTINGS = SETTINGS[:3]
EXAM_SETTINGS = SETTINGS[3:]
HELD_CONDITIONS = (True, True)
FEEDBACK_PROMPT = TEACHING_RULES + """

You get one practice case with structured facts and the student's previous decision and principle.
Check the rules in order using those facts. Owner status is one of agreed, refused, reachable, and
unavailable. Urgent means imminent serious harm, alternative means an equally effective authorized
alternative exists, and limited means the proposed use is confined to what is necessary.
If the student was wrong, explain the decisive condition that changes the correct choice.
If the student was right, explain the decisive condition that supports the choice.
The explanation should be one sentence. Fill both the correct decision and correct principle.
"""
Answer = tuple[str, str]
Training = list[tuple[Question, Lesson]]
Teacher = Callable[[Question, Answer, int], Lesson]


def practice_pool() -> list[Question]:
    cases = [Case(place, "unavailable", urgent, alternative, limited)
             for place, urgent, alternative, limited in product(PRACTICE_SETTINGS, (False, True), (False, True), (False, True))
             if (alternative, limited) != HELD_CONDITIONS]
    return [Question(f"practice-{index:02d}", case, render(case)) for index, case in enumerate(cases)]


def exam_cases() -> list[Case]:
    return [Case(place, "unavailable", urgent, alternative, limited)
            for place, alternative, limited, urgent in product(EXAM_SETTINGS, (False, True), (False, True), (False, True))]


def practice_kind(case: Case) -> tuple[bool, bool, bool]:
    return case.urgent, case.alternative, case.limited


def local_feedback(question: Question, answer: Answer, seed: int, port: int) -> Lesson:
    return request_lessons(
        [question.id], FEEDBACK_PROMPT,
        [{"id": question.id, "facts": asdict(question.case),
          "student": {"decision": answer[0], "principle": answer[1]}}],
        port, seed,
    )[0]


def fit_student(rows: Training, seed: int) -> dict:
    return verified_students(rows, (seed,))[0][0]


def choose_question(
    strategy: str, student: dict, remaining: list[Question], visited: Counter,
) -> tuple[Question | None, Answer | None, bool, int]:
    candidates = remaining if strategy == "mistakes" else remaining[:1]
    if not candidates:
        return None, None, False, 0
    answers = predict(student, features([question.case for question in candidates]))
    graded = [(question, answer, answer != reference(question.case))
              for question, answer in zip(candidates, answers, strict=True)]
    eligible = [item for item in graded if item[2]] if strategy == "mistakes" else graded
    if not eligible:
        return None, None, False, len(graded)
    question, answer, wrong = min(eligible, key=lambda item: visited[practice_kind(item[0].case)])
    return question, answer, wrong, len(graded)


def trajectory(
    baseline: Training, seed: int, strategy: str, budget: int, recorded: list[dict],
    teacher: Teacher | None = None, persist: Callable[[], None] | None = None,
) -> dict:
    training = list(baseline)
    student = fit_student(training, seed)
    checkpoints = [student]
    remaining = practice_pool()
    Random(seed + 104).shuffle(remaining)
    visited: Counter = Counter()
    grader_checks = 0
    accepted = 0

    for index in range(budget):
        question, answer, wrong, checks = choose_question(strategy, student, remaining, visited)
        grader_checks += checks
        if question is None:
            break
        if index < len(recorded):
            lesson = Lesson(**recorded[index]["lesson"])
        elif teacher:
            lesson = teacher(question, answer, seed)
        else:
            raise ValueError("The correction trace ended before the practice budget was completed.")
        if lesson.id != question.id:
            raise ValueError("A teacher reply does not belong to the selected practice case.")
        verified = (lesson.decision, lesson.principle) == reference(question.case)
        turn = {
            "round": index + 1, "question": question.id,
            "prior": {"decision": answer[0], "principle": answer[1]}, "grader_found_error": wrong,
            "selection_grader_checks": checks, "lesson": asdict(lesson), "accepted": verified,
        }
        if index < len(recorded):
            if recorded[index] != turn:
                raise ValueError("The trace does not reproduce the recorded practice decision.")
        else:
            recorded.append(turn)
            if persist:
                persist()
        if verified:
            training.append((question, lesson))
            student = fit_student(training, seed)
            accepted += 1
        checkpoints.append(student)
        visited[practice_kind(question.case)] += 1
        remaining.remove(question)

    steps = len(checkpoints) - 1
    if len(recorded) != steps:
        raise ValueError("The trace contains extra rounds after the run stopped.")
    return {
        "steps": steps, "accepted": accepted,
        "mistakes_queried": sum(turn["grader_found_error"] for turn in recorded),
        "selection_grader_checks": grader_checks, "checkpoints": checkpoints,
    }


def training_data(lesson_path: Path) -> tuple[dict, Training]:
    metadata, lessons = read_lessons(lesson_path)
    baseline = [(question, lesson) for question, lesson in lessons
                if question.case.owner != "unavailable" and reference(question.case) == (lesson.decision, lesson.principle)]
    if not baseline:
        raise ValueError("No verified lessons in the initial training group.")
    return metadata, baseline


def make_manifest(lesson_path: Path, metadata: dict, teacher: str, budget: int, seeds: tuple[int, ...]) -> dict:
    return {
        "experiment": EXPERIMENT, "nascen_version": __version__, "dataset": metadata["dataset"],
        "original_lessons_sha256": sha256(lesson_path.read_bytes()).hexdigest(),
        "teacher": teacher, "prompt_sha256": sha256(FEEDBACK_PROMPT.encode()).hexdigest(),
        "budget": budget, "seeds": list(seeds), "epochs_per_fit": EPOCHS,
        "practice_cases": [{"id": q.id, "facts": asdict(q.case)} for q in practice_pool()],
    }


def collect(
    lesson_path: Path, output: Path, teacher_name: str, teacher: Teacher,
    budget: int = DEFAULT_BUDGET, seeds: tuple[int, ...] = SEEDS,
) -> dict:
    metadata, baseline = training_data(lesson_path)
    manifest = make_manifest(lesson_path, metadata, teacher_name, budget, seeds)
    if output.exists():
        trace = json.loads(output.read_text())
        if {key: value for key, value in trace.items() if key != "turns"} != manifest:
            raise ValueError("The existing trace uses a different teacher, dataset, or protocol.")
    else:
        trace = manifest | {"turns": {str(seed): {name: [] for name in STRATEGIES} for seed in seeds}}

    def checkpoint() -> None:
        write_json(output, trace)

    for seed in seeds:
        arms = trace["turns"][str(seed)]
        active = trajectory(baseline, seed, "mistakes", budget, arms["mistakes"], teacher, checkpoint)
        trajectory(baseline, seed, "random", active["steps"], arms["random"], teacher, checkpoint)
    checkpoint()
    return trace


def measure(student: dict, cases: list[Case]) -> dict:
    grade = grade_cases(student, cases, cases)
    predictions = [row["decision"] for row in grade["predictions"]]
    expected = [reference(case)[0] for case in cases]
    return {
        "decision": grade["decision_accuracy"], "joint": grade["joint_accuracy"],
        "both_decisions": mean(predictions[i:i + 2] == expected[i:i + 2] for i in range(0, len(cases), 2)),
    }


def evaluate(lesson_path: Path, trace: dict) -> dict:
    metadata, baseline = training_data(lesson_path)
    seeds = tuple(trace["seeds"])
    manifest = make_manifest(lesson_path, metadata, trace["teacher"], trace["budget"], seeds)
    if {key: value for key, value in trace.items() if key != "turns"} != manifest:
        raise ValueError("The correction trace does not match the current experiment protocol.")
    cases = exam_cases()
    new_conditions = [case for case in cases if (case.alternative, case.limited) == HELD_CONDITIONS]
    practiced_conditions = [case for case in cases if (case.alternative, case.limited) != HELD_CONDITIONS]
    _, earlier = split(metadata["seed"])
    retention = [left.case for left, _ in earlier if left.case.owner != "unavailable"]
    runs = []
    for seed in seeds:
        arms = trace["turns"][str(seed)]
        budget = trace["budget"]
        for strategy in STRATEGIES:
            turns = arms[strategy]
            replay = trajectory(baseline, seed, strategy, budget, list(turns))
            budget = replay["steps"]
            curve = []
            for step, student in enumerate(replay.pop("checkpoints")):
                curve.append({
                    "step": step, "all": measure(student, cases),
                    "new_place": measure(student, practiced_conditions),
                    "unpracticed_conditions": measure(student, new_conditions),
                    "retention_decision": grade_cases(student, retention, retention)["decision_accuracy"],
                })
            runs.append({"seed": seed, "strategy": strategy, **replay, "curve": curve})
    summary = []
    for strategy in STRATEGIES:
        group = [trial for trial in runs if trial["strategy"] == strategy]
        summary.append({
            "strategy": strategy,
            **{key: mean(trial[key] for trial in group)
               for key in ("steps", "accepted", "mistakes_queried", "selection_grader_checks")},
            "baseline_decision": mean(trial["curve"][0]["all"]["decision"] for trial in group),
            "final_decision": mean(trial["curve"][-1]["all"]["decision"] for trial in group),
            "final_joint": mean(trial["curve"][-1]["all"]["joint"] for trial in group),
            "final_both_decisions": mean(trial["curve"][-1]["all"]["both_decisions"] for trial in group),
            "final_new_place_decision": mean(trial["curve"][-1]["new_place"]["decision"] for trial in group),
            "final_unpracticed_conditions_decision": mean(
                trial["curve"][-1]["unpracticed_conditions"]["decision"] for trial in group
            ),
            "final_unpracticed_both_decisions": mean(
                trial["curve"][-1]["unpracticed_conditions"]["both_decisions"] for trial in group
            ),
            "baseline_retention_decision": mean(trial["curve"][0]["retention_decision"] for trial in group),
            "final_retention_decision": mean(trial["curve"][-1]["retention_decision"] for trial in group),
        })
    return {
        "experiment": EXPERIMENT, "teacher": trace["teacher"], "nascen_version": __version__,
        "baseline_lessons": [lesson.id for _, lesson in baseline], "teacher_available_to_exam": False,
        "source_lessons_sha256": manifest["original_lessons_sha256"],
        "trace_sha256": sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine()},
        "exam_cases": [case_record(case) for case in cases], "retention_count": len(retention),
        "summary": summary, "trials": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    practice = commands.add_parser("teach", help="collect local teacher responses for the practice phase")
    practice.add_argument("--teacher", required=True, help="record the local model and revision")
    practice.add_argument("--port", type=int, default=DEFAULT_PORT)
    practice.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    practice.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    exam = commands.add_parser("exam", help="replay the correction trace and score sealed cases offline")
    exam.add_argument("--output", type=Path)
    for command in (practice, exam):
        command.add_argument("--lessons", type=Path, required=True)
        command.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.command == "teach":
            if not 1 <= args.port <= 65535 or not 1 <= args.budget <= len(practice_pool()):
                raise ValueError("Provide a valid local port and a budget within the practice pool.")
            if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed <= 2**31 - 1 for seed in args.seeds):
                raise ValueError("Use distinct seeds between 0 and 2147483647.")
            trace = collect(args.lessons, args.trace, args.teacher,
                            lambda question, answer, seed: local_feedback(question, answer, seed, args.port),
                            args.budget, tuple(args.seeds))
            print(json.dumps({seed: {name: len(turns) for name, turns in arms.items()}
                              for seed, arms in trace["turns"].items()}, indent=2))
        else:
            trace = json.loads(args.trace.read_text())
            report = evaluate(args.lessons, trace)
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report["summary"], indent=2))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Correction experiment stopped: {error}\n")


if __name__ == "__main__":
    main()
