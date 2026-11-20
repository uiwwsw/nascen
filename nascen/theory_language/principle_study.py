"""Learn a principle from Korean descriptions and test it on distinct condition combinations."""

import argparse
from dataclasses import asdict
from itertools import chain
import json
from pathlib import Path
import platform
from statistics import mean
from tempfile import NamedTemporaryFile
from time import process_time

import torch

from .dataset import digest, read_json, write_json
from .encoder import MultilingualE5, encode_partition
from .principle_data import (
    BUDGETS, DOMAINS, LAMP_DOMAIN, OUTCOMES, STATES, TRACK_OUTCOMES, Lesson, active, held_rules, observed_facts,
    outcome, reference, split_states, study_manifest, training_rules,
)
from .principle_model import PrincipleLearner, fit_curriculum, predict, study
from .principle_teacher import collect, load_lessons, revise
from .principle_update import Evidence, evidence_vectors, read_evidence, validate_facts, verified_study
from .revision import DEFAULT_ENCODER

LESSONS = Path("docs/experiment-011-lessons.json")
REVIEW = Path("docs/experiment-011-review.json")
SEEDS = (11, 22, 42)
ARMS = ("experience", "relevant", "irrelevant")
TARGET_BALANCED_ACCURACY = 0.80


def prepare_curriculum(encoder, domain=LAMP_DOMAIN):
    rows = []
    for rule_index, rule in enumerate(training_rules(domain)):
        indexes = rule.structure
        for state_index, state in enumerate(STATES):
            for style in (0, 1):
                phrases, order = observed_facts(state, style, f"train-{rule_index}-{state_index}-{style}", domain=domain)
                lesson = reference(rule, 1 - style, domain=domain).phrases()
                rows.append((phrases, lesson, outcome(rule, state), tuple(order.index(i) for i in indexes),
                             tuple(state[i] == value for i, value in zip(indexes, rule.values, strict=True))))
    vectors, cost = encode_partition(encoder, (text for row in rows for text in (*row[0], *row[1])))
    facts = torch.stack([torch.stack([vectors[text] for text in row[0]]) for row in rows])
    lessons = torch.stack([torch.stack([vectors[text] for text in row[1]]) for row in rows])
    labels = torch.tensor([row[2] for row in rows], dtype=torch.float32)
    attention = torch.tensor([row[3] for row in rows])
    truth = torch.tensor([row[4] for row in rows], dtype=torch.float32)
    return (facts, lessons, labels, attention, truth), {"examples": len(rows), "sha256": digest(rows), "encoding": cost}


def lesson_vector(encoder, lesson):
    if lesson is None:
        return None
    sentences, _ = encode_partition(encoder, lesson.phrases())
    return torch.stack([sentences[text] for text in lesson.phrases()])


def save(path, model, encoder, *, adapter=None, metadata, domain=LAMP_DOMAIN, evidence=()):
    bundle = {"schema": "principle-study-v1", "encoder": encoder.identity, "width": model.width,
              "weights": model.state_dict(), "adapter": adapter, "metadata": metadata}
    if domain != LAMP_DOMAIN:
        bundle["domain"] = domain
    if evidence:
        bundle["verified_evidence"] = [{"key": row.key, "facts": row.facts, "lit": row.lit} for row in evidence]
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(bundle, stream)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_bundle(path, encoder):
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    if (bundle["schema"] != "principle-study-v1" or bundle["encoder"] != encoder.identity
            or bundle["width"] != encoder.width or bundle.get("domain", LAMP_DOMAIN) not in DOMAINS):
        raise ValueError("The principle checkpoint and sentence encoder must match.")
    model = PrincipleLearner(bundle["width"], 0)
    model.load_state_dict(bundle["weights"])
    return model.eval().requires_grad_(False), bundle


def load(path, encoder):
    model, bundle = load_bundle(path, encoder)
    return model, bundle["adapter"]


def balanced(guesses, truth):
    return mean(mean(guess == value for guess, gold in zip(guesses, truth, strict=True) if gold == value)
                for value in (False, True))


def benchmark(encoder, packet, review, *, seeds=SEEDS, steps=480, checkpoint=None):
    curriculum, training_cost = prepare_curriculum(encoder)
    lessons = load_lessons(packet, review)
    tasks = held_rules()
    support_states, queries = split_states()
    sent_support = [observed_facts(state, 0, f"support-{i}")[0] for i, state in enumerate(support_states)]
    sent_exams = [observed_facts(state, 1, f"query-{i}")[0] for i, state in enumerate(queries)]
    support = evidence_vectors(encoder, sent_support)
    exam = evidence_vectors(encoder, sent_exams)
    teacher_vectors, lessons_cost = encode_partition(encoder, (text for lesson in lessons.values() for text in lesson.phrases()))
    adapters = {key: torch.stack([teacher_vectors[text] for text in lesson.phrases()]) for key, lesson in lessons.items()}
    # Rotate within the same connective so the irrelevant arm has the same shape and style.
    irrelevant = {}
    for both in (False, True):
        group = [key for key, rule in tasks.items() if rule.both == both]
        irrelevant.update(zip(group, group[1:] + group[:1], strict=True))
    trials = []
    for seed in seeds:
        start = process_time()
        model = fit_curriculum(curriculum, encoder.width, seed, steps=steps)
        training_cpu = process_time() - start
        for name, rule in tasks.items():
            labels = torch.tensor([outcome(rule, state) for state in support_states], dtype=torch.float32)
            truth = [outcome(rule, state) for state in queries]
            veto = [active(rule, state) and state[rule.exception] == rule.values[2] for state in queries]
            before = (predict(model, model.write(None), exam) >= 0.5).tolist()
            for budget in BUDGETS:
                seen = {tuple(state[i] for i in rule.structure) for state in support_states[:budget]}
                novel = [tuple(state[i] for i in rule.structure) not in seen for state in queries]
                for arm, phrase in (("experience", None), ("relevant", adapters[name]),
                                    ("irrelevant", adapters[irrelevant[name]])):
                    adapter = study(model, phrase, support[:budget], labels[:budget])
                    probabilities = predict(model, adapter, exam)
                    guesses = (probabilities >= 0.5).tolist()
                    trials.append({
                        "seed": seed, "lesson": name, "budget": budget, "arm": arm, "curriculum_cpu_seconds": training_cpu,
                        "balanced_accuracy": balanced(guesses, truth),
                        "accuracy": mean(g == t for g, t in zip(guesses, truth, strict=True)),
                        "veto_accuracy": mean(not g for g, exceptional in zip(guesses, veto, strict=True) if exceptional),
                        "novel_relevant_correct": sum(g == t for g, t, unknown in zip(guesses, truth, novel, strict=True)
                                                      if unknown),
                        "novel_relevant_total": sum(novel),
                        "changed_from_no_study": sum(a != b for a, b in zip(guesses, before, strict=True)),
                        "predictions": guesses, "probabilities": probabilities.tolist(),
                    })
                    if checkpoint is not None and seed == seeds[0] and name == next(iter(tasks)) and budget == 2 and arm == "relevant":
                        save(checkpoint, model, encoder, adapter=adapter,
                             metadata={"seed": seed, "observations": budget, "source": "reviewed_teacher"})
    summary = []
    effects = []
    for budget in BUDGETS:
        predictions = {(row["seed"], row["lesson"], row["arm"]): row["predictions"]
                       for row in trials if row["budget"] == budget}
        corrected, regressed = 0, 0
        for seed in seeds:
            for key, rule in tasks.items():
                before, after = predictions[seed, key, "experience"], predictions[seed, key, "relevant"]
                truth = [outcome(rule, state) for state in queries]
                corrected += sum(a != target and b == target for a, b, target in zip(before, after, truth, strict=True))
                regressed += sum(a == target and b != target for a, b, target in zip(before, after, truth, strict=True))
        effects.append({"budget": budget, "corrected": corrected, "regressed": regressed,
                        "compared_decisions": len(seeds) * len(tasks) * len(queries)})
        for arm in ARMS:
            rows = [row for row in trials if row["arm"] == arm and row["budget"] == budget]
            novel_count = sum(row["novel_relevant_total"] for row in rows)
            summary.append({"budget": budget, "arm": arm, **{
                metric: mean(row[metric] for row in rows)
                for metric in ("balanced_accuracy", "accuracy", "veto_accuracy", "changed_from_no_study")
            }, "novel_relevant_accuracy": sum(row["novel_relevant_correct"] for row in rows) / novel_count
            if novel_count else None, "novel_relevant_total": novel_count, "seed_balanced_accuracy": [
                mean(row["balanced_accuracy"] for row in rows if row["seed"] == seed) for seed in seeds
            ]})
    costs = [*packet["cost"].values(), *(row["cost"] for row in packet.get("revisions", ()))]
    return {
        "experiment": "011 · Learning grounded principles from Korean explanations",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine(),
                        "threads": torch.get_num_threads()},
        "manifest": study_manifest(), "encoder": encoder.identity, "curriculum": training_cost,
        "seeds": seeds, "curriculum_steps": steps, "study_steps": 40, "budgets": BUDGETS,
        "teacher": packet["teacher"], "lessons_sha256": digest(packet), "review_sha256": digest(review),
        "teacher_cost": {"requests": len(costs), "wall_seconds": sum(row["wall_seconds"] for row in costs),
                         **{key: sum(row["usage"][key] for row in costs)
                            if all(key in row["usage"] for row in costs) else None
                            for key in ("prompt_tokens", "completion_tokens")}},
        "lesson_encoding": lessons_cost, "irrelevant_pairing": irrelevant,
        "support_sha256": digest(sent_support), "query_sha256": digest(sent_exams),
        "query_facts": sent_exams, "answers": {key: [outcome(rule, state) for state in queries] for key, rule in tasks.items()},
        "balanced_accuracy_target": TARGET_BALANCED_ACCURACY,
        "first_tested_budget_at_target": {
            arm: next((row["budget"] for row in summary
                       if row["arm"] == arm and row["balanced_accuracy"] >= TARGET_BALANCED_ACCURACY), None)
            for arm in ARMS
        },
        "effect_of_relevant_study_vs_experience": effects, "summary": summary, "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    teaching = commands.add_parser("collect")
    teaching.add_argument("--output", type=Path, required=True)
    teaching.add_argument("--teacher", required=True)
    teaching.add_argument("--port", type=int, default=8766)
    teaching.add_argument("--retry", choices=tuple(held_rules()))
    teaching.add_argument("--feedback")
    train = commands.add_parser("train")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, default=11)
    train.add_argument("--steps", type=int, default=480)
    train.add_argument("--domain", choices=DOMAINS, default=LAMP_DOMAIN)
    learn = commands.add_parser("study")
    learn.add_argument("--base", type=Path, required=True)
    learn.add_argument("--lesson", type=Path)
    learn.add_argument("--observations", type=Path)
    learn.add_argument("--verify", type=Path, help="independent, labeled observations to preserve during an update")
    learn.add_argument("--output", type=Path, required=True)
    inference = commands.add_parser("predict")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--facts", nargs=4, required=True)
    experiment = commands.add_parser("benchmark")
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--lessons", type=Path, default=LESSONS)
    experiment.add_argument("--review", type=Path, default=REVIEW)
    experiment.add_argument("--checkpoint", type=Path)
    experiment.add_argument("--steps", type=int, default=480)
    experiment.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    for command in (train, learn, inference, experiment):
        command.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.command == "collect":
            if bool(args.retry) != bool(args.feedback):
                raise ValueError("Specify both --retry and --feedback for a revised lesson.")
            packet = (revise(args.output, args.teacher, args.retry, args.feedback, args.port)
                      if args.retry else collect(args.output, args.teacher, args.port))
            result = {"lessons": len(packet["lessons"])}
        else:
            encoder = MultilingualE5(args.encoder)
            if args.command in ("train", "benchmark") and args.steps < 1:
                raise ValueError("Choose a positive number of curriculum steps.")
            if args.command == "train":
                data, provenance = prepare_curriculum(encoder, args.domain)
                model = fit_curriculum(data, encoder.width, args.seed, args.steps)
                save(args.output, model, encoder, metadata={"seed": args.seed, "curriculum": provenance},
                     domain=args.domain)
                result = {"checkpoint": str(args.output), "domain": args.domain, "curriculum": provenance}
            elif args.command == "study":
                input_paths = (args.base, args.lesson, args.observations, args.verify)
                if any(path is not None and args.output.resolve() == path.resolve() for path in input_paths):
                    raise ValueError("Write the adapted student to a separate file.")
                model, bundle = load_bundle(args.base, encoder)
                lesson = Lesson(**read_json(args.lesson)) if args.lesson else None
                observations = read_evidence(args.observations, encoder)
                verified = verified_study(
                    model, lesson_vector(encoder, lesson), observations, previous=bundle["adapter"],
                    recorded=tuple(Evidence(**row) for row in bundle.get("verified_evidence", ())),
                    verification=read_evidence(args.verify, encoder),
                )
                result = {"checkpoint": str(args.output), "observations": len(observations),
                          "status": verified.status, "conflicts": verified.conflicts, "reason": verified.reason,
                          "verified_observations": len(verified.evidence)}
                prior_keys = {row["key"] for row in bundle.get("verified_evidence", ())}
                observation_keys = {row.key for row in observations}
                metadata = result | {
                    "observations_added": sum(row.key in observation_keys - prior_keys for row in verified.evidence),
                    "new_lesson": lesson is not None and verified.status != "rejected",
                    "continued_study": bundle["adapter"] is not None and lesson is None,
                }
                save(args.output, model, encoder, adapter=verified.adapter, metadata=metadata,
                     domain=bundle.get("domain", LAMP_DOMAIN), evidence=verified.evidence)
            elif args.command == "predict":
                model, bundle = load_bundle(args.checkpoint, encoder)
                adapter = bundle["adapter"]
                if adapter is None:
                    raise ValueError("Load a checkpoint produced by study.")
                validate_facts(args.facts)
                probability = float(predict(model, adapter, evidence_vectors(encoder, [args.facts]))[0])
                outcomes = OUTCOMES if bundle.get("domain", LAMP_DOMAIN) == LAMP_DOMAIN else TRACK_OUTCOMES
                result = {"outcome": outcomes[probability >= 0.5], "score": probability}
            else:
                if len(set(args.seeds)) != len(args.seeds):
                    raise ValueError("Use distinct seeds.")
                report = benchmark(encoder, read_json(args.lessons), read_json(args.review),
                                   seeds=args.seeds, steps=args.steps, checkpoint=args.checkpoint)
                write_json(args.output, report)
                result = {"report": str(args.output), "summary": report["summary"]}
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Nascen principle study: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
