"""Train and examine independent students on the controlled lending task."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import platform
from statistics import mean

import torch
from torch import nn
from torch.nn.functional import cross_entropy, normalize

from . import __version__
from .consent import DECISIONS, PRINCIPLES, Case, compile_question, reference, render, split
from .teacher import DATASET_VERSION, read_lessons

BUNDLE_VERSION = 1
ARMS = ("scratch_answers", "scratch_principles", "scratch_verified_principles", "minilm_principles")
SEEDS = (11, 22, 42)
HIDDEN_SIZE = 24
EPOCHS = 180
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt")


class Student(nn.Module):
    def __init__(self, width: int, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.Sequential(nn.Linear(width, HIDDEN_SIZE), nn.Tanh())
            self.decision = nn.Linear(HIDDEN_SIZE, len(DECISIONS))
            self.principle = nn.Linear(HIDDEN_SIZE, len(PRINCIPLES))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.hidden(inputs)
        return self.decision(features), self.principle(features)


class MiniLM:
    def __init__(self, path: Path):
        if not path.is_dir():
            raise ValueError("Provide a local MiniLM snapshot directory.")
        files = {}
        for name in MINILM_FILES:
            with (path / name).open("rb") as source:
                files[name] = file_digest(source, "sha256").hexdigest()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ValueError("Install the optional requirements-teacher.txt dependencies.") from error
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.encoder = AutoModel.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).eval()
        self.identity = {
            "model": MINILM_ID, "files_sha256": files, "parameters": sum(p.numel() for p in self.encoder.parameters()),
        }

    @torch.no_grad()
    def features(self, cases: list[Case]) -> torch.Tensor:
        unique = list(dict.fromkeys(cases))
        vectors = []
        for start in range(0, len(unique), 16):
            texts = [render(case) for case in unique[start:start + 16]]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            tokens = self.encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            vectors.extend(normalize((tokens * mask).sum(1) / mask.sum(1).clamp(min=1), dim=1))
        index = dict(zip(unique, vectors, strict=True))
        return torch.stack([index[case] for case in cases])


def features(cases: list[Case], encoder: MiniLM | None = None) -> torch.Tensor:
    if encoder is not None:
        return encoder.features(cases)
    return torch.tensor([case.features() for case in cases])


def fit(arm: str, seed: int, inputs: torch.Tensor, decisions: torch.Tensor, principles: torch.Tensor) -> dict:
    width = inputs.shape[-1]
    student = Student(width, seed)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
    with_principles = arm != "scratch_answers"
    counts = torch.bincount(principles, minlength=len(PRINCIPLES)).float()
    reason_weights = counts.clamp(min=1).rsqrt()

    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        answer, reason = student(inputs)
        loss = cross_entropy(answer, decisions)
        if with_principles:
            loss = loss + cross_entropy(reason, principles, weight=reason_weights)
        loss.backward()
        optimizer.step()

    return student_checkpoint(student, arm, seed, len(inputs), with_principles)


def student_checkpoint(student: Student, arm: str, seed: int, examples: int, with_principles: bool) -> dict:
    return {
        "arm": arm, "seed": seed, "width": student.hidden[0].in_features, "trains_principle": with_principles,
        "training_examples": examples,
        "trainable_parameters": sum(weight.numel() for weight in student.parameters()),
        "weights": {key: weight.tolist() for key, weight in student.state_dict().items()},
    }


def train(lesson_file: Path, encoder: MiniLM | None = None, seeds: tuple[int, ...] = SEEDS) -> dict:
    metadata, collected = read_lessons(lesson_file)
    training = [(question, lesson) for question, lesson in collected if lesson.consistent]
    if not training:
        raise ValueError("No internally consistent teacher examples are available for training.")
    cases = [compile_question(question.text) for question, _ in training]
    decisions = torch.tensor([DECISIONS.index(lesson.decision) for _, lesson in training])
    principles = torch.tensor([tuple(PRINCIPLES).index(lesson.principle) for _, lesson in training])
    verified = [index for index, (case, (_, lesson)) in enumerate(zip(cases, training, strict=True))
                if reference(case) == (lesson.decision, lesson.principle)]
    scratch = features(cases)
    conditions = {
        "scratch_answers": (scratch, decisions, principles),
        "scratch_principles": (scratch, decisions, principles),
    }
    # This control reuses only the teacher labels that match the stipulated training rubric.
    if verified:
        conditions["scratch_verified_principles"] = (scratch[verified], decisions[verified], principles[verified])
    if encoder is not None:
        conditions["minilm_principles"] = (features(cases, encoder), decisions, principles)

    students = [fit(arm, seed, *data) for arm, data in conditions.items() for seed in seeds]
    audited = [(reference(question.case), (lesson.decision, lesson.principle)) for question, lesson in collected]
    return {
        "schema": BUNDLE_VERSION, "nascen_version": __version__, "dataset": metadata["dataset"],
        "split_seed": metadata["seed"], "teacher": metadata["teacher"], "epoch_count": EPOCHS,
        "lesson_sha256": sha256(lesson_file.read_bytes()).hexdigest(),
        "teacher_audit": {
            "received": len(collected), "used": len(training), "verified": len(verified),
            "decision_accuracy": mean(gold[0] == predicted[0] for gold, predicted in audited),
            "joint_accuracy": mean(gold == predicted for gold, predicted in audited),
        },
        "pretrained_encoder": encoder.identity if encoder is not None else None,
        "students": students,
    }


def load_bundle(path: Path) -> dict:
    bundle = json.loads(path.read_text())
    if bundle["schema"] != BUNDLE_VERSION or bundle["dataset"] != DATASET_VERSION:
        raise ValueError("Unsupported student checkpoint.")
    return bundle


def validate_encoder(bundle: dict, encoder: MiniLM | None) -> None:
    expected = bundle["pretrained_encoder"]
    if expected is not None and (encoder is None or encoder.identity != expected):
        raise ValueError("Use the same local MiniLM snapshot used to train the student.")


@torch.no_grad()
def predict(run: dict, inputs: torch.Tensor) -> list[tuple[str, str | None]]:
    model = Student(run["width"], run["seed"])
    model.load_state_dict({key: torch.tensor(value) for key, value in run["weights"].items()})
    decision, principle = model(inputs)
    answers = [DECISIONS[index] for index in decision.argmax(dim=1).tolist()]
    reasons = [tuple(PRINCIPLES)[index] for index in principle.argmax(dim=1).tolist()]
    if not run["trains_principle"]:
        reasons = [None] * len(answers)
    return list(zip(answers, reasons, strict=True))


def grade(rows: list[dict], reasoning: bool) -> dict:
    originals = rows[::2]
    contrasts = list(zip(originals[::2], originals[1::2], strict=True))
    changed = [(left, right) for left, right in contrasts if left["gold_decision"] != right["gold_decision"]]
    urgent = [row for row in rows if row["gold_principle"] == "urgent_exception"]
    answers = [row["decision"] == row["gold_decision"] for row in rows]
    principles = [row["principle"] == row["gold_principle"] for row in rows]

    return {
        "decision_accuracy": mean(answers),
        "principle_accuracy": mean(principles) if reasoning else None,
        "joint_accuracy": mean(answer and reason for answer, reason in zip(answers, principles, strict=True)) if reasoning else None,
        "rationale_consistency": mean(PRINCIPLES[row["principle"]][0] == row["decision"] for row in rows) if reasoning else None,
        "paraphrase_consistency": mean(
            (left["decision"], left["principle"]) == (right["decision"], right["principle"])
            for left, right in zip(rows[::2], rows[1::2], strict=True)
        ),
        "counterfactual_accuracy": mean(all(row["decision"] == row["gold_decision"] for row in pair) for pair in changed),
        "counterfactual_pairs": len(changed),
        "urgent_exception_accuracy": mean(row["decision"] == row["gold_decision"] for row in urgent),
        "urgent_exception_questions": len(urgent),
    }


def examine(bundle: dict, encoder: MiniLM | None = None) -> dict:
    validate_encoder(bundle, encoder)
    _, paired = split(bundle["split_seed"])
    questions = [question for pair in paired for question in pair]
    cases = [compile_question(question.text) for question in questions]
    scratch = features(cases)
    pretrained = features(cases, encoder) if encoder is not None else None
    trials = []
    for run in bundle["students"]:
        inputs = pretrained if run["arm"] == "minilm_principles" else scratch
        predictions = predict(run, inputs)
        answers = []
        for question, case, (decision, principle) in zip(questions, cases, predictions, strict=True):
            gold_decision, gold_principle = reference(case)
            answers.append({
                "id": question.id, "question": question.text, "decision": decision, "principle": principle,
                "gold_decision": gold_decision, "gold_principle": gold_principle,
            })
        trials.append({key: run[key] for key in ("arm", "seed", "trainable_parameters", "training_examples")} | {
            "metrics": grade(answers, run["trains_principle"]),
            "predictions": [{key: row[key] for key in ("id", "decision", "principle")} for row in answers],
        })

    groups = dict.fromkeys(trial["arm"] for trial in trials)
    summary = []
    for arm in groups:
        group = [trial["metrics"] for trial in trials if trial["arm"] == arm]
        summary.append({"arm": arm, **{
            key: mean(metrics[key] for metrics in group) if value is not None else None
            for key, value in group[0].items()
        }})

    return {key: value for key, value in bundle.items() if key != "students"} | {
        "experiment": "002 — Independent Student", "questions": len(questions), "distinct_cases": len(paired),
        "teacher_available_to_exam": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine()},
        "summary": summary, "trials": trials,
        "answer_key": [
            {"id": question.id, "question": question.text, "decision": answer, "principle": reason}
            for question, (answer, reason) in zip(questions, map(reference, cases), strict=True)
        ],
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train students using a completed local lesson file")
    training.add_argument("--lessons", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    exam = commands.add_parser("exam", help="grade a saved student without any teacher calls")
    exam.add_argument("--student", type=Path, required=True)
    exam.add_argument("--output", type=Path)
    answer = commands.add_parser("answer", help="answer one question using a saved student")
    answer.add_argument("--student", type=Path, required=True)
    answer.add_argument("--arm", choices=ARMS, default="scratch_principles")
    answer.add_argument("--seed", type=int, default=42)
    answer.add_argument("--question", required=True)
    for command in (training, exam, answer):
        command.add_argument("--minilm", type=Path, help="path to a local frozen MiniLM snapshot")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        encoder = MiniLM(args.minilm) if args.minilm else None
        if args.command == "train":
            if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed <= 2**31 - 1 for seed in args.seeds):
                raise ValueError("Student seeds must be distinct integers between 0 and 2147483647.")
            bundle = train(args.lessons, encoder, tuple(args.seeds))
            write_json(args.output, bundle)
            print(f"Saved {len(bundle['students'])} student checkpoints to {args.output}.")
        else:
            bundle = load_bundle(args.student)
            if args.command == "exam":
                report = examine(bundle, encoder)
                if args.output:
                    write_json(args.output, report)
                print(json.dumps(report["summary"], indent=2))
            else:
                run = next((r for r in bundle["students"] if (r["arm"], r["seed"]) == (args.arm, args.seed)), None)
                if run is None:
                    raise ValueError("The saved bundle lacks that student arm and seed.")
                if args.arm == "minilm_principles":
                    validate_encoder(bundle, encoder)
                decision, principle = predict(
                    run, features([compile_question(args.question)], encoder if args.arm == "minilm_principles" else None),
                )[0]
                print(json.dumps({
                    "decision": decision, "principle": principle,
                    "explanation": PRINCIPLES[principle][1] if principle is not None else None,
                }, indent=2))
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Distillation stopped: {error}\n")


if __name__ == "__main__":
    main()
