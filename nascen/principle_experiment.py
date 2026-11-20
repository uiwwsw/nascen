"""Collect, distill, and test a principle in the fictional lending exercise."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
from random import Random
import re
from statistics import mean
from time import process_time_ns

import torch

from .consent import DECISIONS, OWNER_STATES, PRINCIPLES, SETTINGS, Case, reference
from .distillation import EPOCHS, SEEDS, features, fit, predict, write_json
from .principles import (
    ARM, DATASET, collect, constraints, execute, read_packet, reference_packet, teaching_rules, train_principle,
)
from .teacher import DEFAULT_PORT, DEFAULT_SEED

TRAIN_SETTINGS, EXAM_SETTINGS = SETTINGS[:3], SETTINGS[3:]
HELD_FACTS = {
    ("agreed", True, True, False), ("refused", True, True, False),
    ("reachable", False, False, True), ("reachable", True, True, True),
    ("unavailable", False, True, True), ("unavailable", True, True, True),
}
OWNER_WORDS = dict(zip(OWNER_STATES, (
    "the owner said yes", "the owner said no", "the owner is available to ask", "the owner cannot answer in time",
), strict=True))
FACT_WORDS = (
    OWNER_WORDS,
    {False: "there is time to act without serious harm", True: "serious harm will occur without immediate action"},
    {False: "no authorized substitute works as well", True: "an authorized substitute works as well"},
    {False: "the use exceeds what is needed", True: "the use covers only what is needed"},
)
FACT_CODES = [{phrase: value for value, phrase in group.items()} for group in FACT_WORDS]
QUESTION = "May the player borrow the item"


def facts(case: Case) -> tuple:
    return case.owner, case.urgent, case.alternative, case.limited


def render_transfer(case: Case) -> str:
    owner, urgency, alternative, scope = (words[value] for words, value in zip(FACT_WORDS, facts(case), strict=True))
    return f"At the {case.setting}, {owner}. {scope.capitalize()}; {alternative}; {urgency}. {QUESTION}?"


def compile_transfer(question: str) -> Case:
    """Read the controlled vocabulary in any order with one statement per field."""
    prefix, rest = question.strip().casefold().split(",", 1)
    match = re.fullmatch(r"at the (\w+)", prefix)
    if match is None or match[1] not in SETTINGS:
        raise ValueError("Unknown lending setting.")
    clauses = [part.strip() for part in re.split(r"[.;?]", rest) if part.strip()]
    if clauses.count(QUESTION.casefold()) != 1:
        raise ValueError("Expected one lending question.")
    clauses.remove(QUESTION.casefold())
    if len(clauses) != len(FACT_CODES):
        raise ValueError("Expected exactly four fact statements.")
    values = []
    for lexicon in FACT_CODES:
        matches = [lexicon[clause] for clause in clauses if clause in lexicon]
        if len(matches) != 1:
            raise ValueError("Each fact needs a single supported statement.")
        values.append(matches[0])
    return Case(match[1], *values)


def cases(settings: tuple[str, ...]) -> list[Case]:
    return [Case(*values) for values in product(settings, OWNER_STATES, (False, True), (False, True), (False, True))]


def training_pool() -> list[Case]:
    return [case for case in cases(TRAIN_SETTINGS) if facts(case) not in HELD_FACTS]


def annotation_bytes(value: dict | list) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def revision_material(packet: dict) -> dict:
    rules = teaching_rules(packet["teaching"])
    by_reason = {}
    for case in training_pool():
        expected = reference(case)
        if execute(rules, case) != expected:
            by_reason.setdefault(expected[1], asdict(case) | {"decision": expected[0], "principle": expected[1]})
    if not by_reason:
        raise ValueError("The initial packet had no mismatches in the practice pool.")
    return {"first_attempt": packet["teaching"], "counterexamples": list(by_reason.values())}


def train(packet: dict, seeds: tuple[int, ...] = SEEDS, *, measure_time: bool = False) -> dict:
    started = process_time_ns()
    regions = constraints(teaching_rules(packet["teaching"]))
    compilation_ms = (process_time_ns() - started) / 1e6
    pool = training_pool()
    runs = []
    for seed in seeds:
        examples = pool.copy()
        Random(seed).shuffle(examples)
        started = process_time_ns()
        learned = train_principle(regions, seed)
        if measure_time:
            learned["fit_cpu_ms"] = (process_time_ns() - started) / 1e6
        runs.append(learned | {"annotation_bytes": annotation_bytes(packet["teaching"])})

        for budget, rationale in ((8, False), (24, False), (len(pool), False), (len(pool), True)):
            selected = examples[:budget]
            annotated = [reference(case) for case in selected]
            labels = torch.tensor([
                (DECISIONS.index(answer), tuple(PRINCIPLES).index(reason)) for answer, reason in annotated
            ])
            annotations = [
                asdict(case) | {"decision": answer} | ({"principle": reason} if rationale else {})
                for case, (answer, reason) in zip(selected, annotated, strict=True)
            ]
            inputs = features(selected)
            started = process_time_ns()
            run = fit("scratch_principles" if rationale else "scratch_answers", seed, inputs, *labels.T)
            if measure_time:
                run["fit_cpu_ms"] = (process_time_ns() - started) / 1e6
            runs.append(run | {
                "arm": f"examples_{budget}" + ("_with_reasons" if rationale else ""),
                "annotation_bytes": annotation_bytes(annotations),
                "training_pool_indices": [pool.index(case) for case in selected],
                "distinct_training_facts": len(set(map(facts, selected))),
            })
    bundle = {
        "dataset": DATASET, "epoch_count": EPOCHS,
        "teaching_sha256": sha256(json.dumps(packet["teaching"], sort_keys=True).encode()).hexdigest(),
        "students": runs,
    }
    if measure_time:
        bundle["compilation_cpu_ms"] = compilation_ms
    return bundle


def grade(questions: list[Case], outcomes: list[tuple[str | None, str | None]]) -> dict:
    gold = list(map(reference, questions))
    decisions = [expected[0] == given[0] for expected, given in zip(gold, outcomes, strict=True)]
    joint = [expected == given for expected, given in zip(gold, outcomes, strict=True)]
    held = [i for i, case in enumerate(questions) if facts(case) in HELD_FACTS]
    index = {case: i for i, case in enumerate(questions)}
    changed = []
    for i, case in enumerate(questions):
        if case.urgent:
            continue
        partner = index[Case(case.setting, case.owner, True, case.alternative, case.limited)]
        if gold[i][0] != gold[partner][0]:
            changed.append((i, partner))
    return {
        "answered": sum(answer is not None for answer, _ in outcomes),
        "correct_decisions": sum(decisions), "correct_joint": sum(joint) if any(p for _, p in outcomes) else None,
        "held_decisions": sum(decisions[i] for i in held),
        "held_joint": sum(joint[i] for i in held) if any(p for _, p in outcomes) else None,
        "urgency_pairs_correct": sum(decisions[i] and decisions[j] for i, j in changed),
        "joint_by_principle": {
            reason: sum(correct and expected[1] == reason for correct, expected in zip(joint, gold, strict=True))
            for reason in PRINCIPLES
        },
    }


def exam(bundle: dict, packet: dict | None = None) -> dict:
    if bundle["dataset"] != DATASET:
        raise ValueError("Unsupported principle checkpoint.")
    questions = cases(EXAM_SETTINGS)
    texts = list(map(render_transfer, questions))
    parsed = list(map(compile_transfer, texts))
    inputs = features(questions + parsed)
    runs = []
    for checkpoint in bundle["students"]:
        outcomes = predict(checkpoint, inputs)
        runs.append({key: value for key, value in checkpoint.items() if key != "weights"} | {
            "facts": grade(questions, outcomes[:len(questions)]), "text": grade(questions, outcomes[len(questions):]),
            "text_predictions": [f"{answer}/{reason or '-'}" for answer, reason in outcomes[len(questions):]],
        })
    result = {
        "dataset": DATASET, "environment": {"platform": platform.platform(), "torch": str(torch.__version__)},
        "training_pool": [asdict(case) for case in training_pool()],
        "epoch_count": bundle["epoch_count"],
        "questions": [
            asdict(case) | {"text": sentence, "gold": reference(case), "held_facts": facts(case) in HELD_FACTS}
            for case, sentence in zip(questions, texts, strict=True)
        ],
        "parsed_correctly": sum(case == recovered for case, recovered in zip(questions, parsed, strict=True)),
        "runs": runs,
    }
    means = {}
    for arm in dict.fromkeys(run["arm"] for run in runs):
        group = [run for run in runs if run["arm"] == arm]
        means[arm] = {
            key: mean(run["text"][key] for run in group) if group[0]["text"][key] is not None else None
            for key in ("correct_decisions", "correct_joint", "held_decisions", "held_joint", "urgency_pairs_correct")
        }
    result["mean_text_counts"] = means
    if packet is not None:
        digest = sha256(json.dumps(packet["teaching"], sort_keys=True).encode()).hexdigest()
        if digest != bundle["teaching_sha256"]:
            raise ValueError("Compare with the same rule packet used to train the student.")
        rules = teaching_rules(packet["teaching"])
        text_outcomes = [execute(rules, case) for case in parsed]
        rule_predictions = [f"{answer or '-'}/{reason or '-'}" for answer, reason in text_outcomes]
        for run in runs:
            if run["arm"] == ARM:
                run["text_rule_agreement"] = sum(
                    left == right for left, right in zip(run["text_predictions"], rule_predictions, strict=True)
                )
        result["stored_rule"] = {
            "facts": grade(questions, [execute(rules, case) for case in questions]), "text": grade(questions, text_outcomes),
            "text_predictions": rule_predictions,
            "teacher_requests": packet["requests"], "annotation_bytes": annotation_bytes(packet["teaching"]),
            "teacher_usage": packet["usage"], "teacher_seconds": packet["teacher_seconds"],
            "source": packet["model"], "clauses": len(rules), "effective_regions": len(constraints(rules)),
        }
    if "compilation_cpu_ms" in bundle:
        result["compilation_cpu_ms"] = bundle["compilation_cpu_ms"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    teaching = commands.add_parser("teach", help="collect one principle from a loopback teacher")
    teaching.add_argument("--output", type=Path, required=True)
    teaching.add_argument("--teacher", required=True, help="record the local model and revision")
    teaching.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=DEFAULT_PORT)
    teaching.add_argument("--seed", type=int, default=DEFAULT_SEED)
    teaching.add_argument("--revise", type=Path, help="include the initial packet and training-only feedback")
    fixture = commands.add_parser("reference", help="save the hand-transcribed exercise code as a clean control")
    fixture.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("train", help="fit students offline and save weights")
    training.add_argument("--rules", type=Path, required=True)
    training.add_argument("--checkpoint", type=Path, required=True)
    training.add_argument("--measure-time", action="store_true")
    testing = commands.add_parser("exam", help="test held-out cases without contacting the teacher")
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--rules", type=Path, help="include the stored-rule interpreter as a separate control")
    testing.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.command == "teach":
            revision = revision_material(read_packet(args.revise)) if args.revise else None
            packet = collect(args.output, args.teacher, args.port, args.seed, revision=revision)
            print(f"Collected one principle with {len(packet['teaching']['rules'])} ordered clauses.")
        elif args.command == "reference":
            write_json(args.output, reference_packet())
        elif args.command == "train":
            bundle = train(read_packet(args.rules), measure_time=args.measure_time)
            write_json(args.checkpoint, bundle)
            print(f"Saved {len(bundle['students'])} independent checkpoints.")
        else:
            bundle = json.loads(args.checkpoint.read_text())
            result = exam(bundle, read_packet(args.rules) if args.rules else None)
            if args.output:
                write_json(args.output, result)
            counts = result["mean_text_counts"].copy()
            if "stored_rule" in result:
                counts["stored_rule"] = result["stored_rule"]["text"]
            print(json.dumps(counts, indent=2))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Principle experiment stopped: {error}\n")


if __name__ == "__main__":
    main()
