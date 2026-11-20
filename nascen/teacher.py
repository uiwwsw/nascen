"""Collect training-only lessons from a local, licensed teacher model."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .consent import DECISIONS, PRINCIPLES, Question, reference, split

DATASET_VERSION = "lending-v1"
DEFAULT_SEED = 42
DEFAULT_PORT = 8766
BATCH_SIZE = 5

LENDING_CODE = """You are a teacher for a fictional lending exercise. Use this stipulated exercise code.
Apply the first matching rule:
1. If the owner explicitly consented, use allow and permission.
2. If the owner explicitly refused, use decline and refusal.
3. If the owner can be asked, or there is no imminent serious harm, use ask and ask_owner.
4. Otherwise, if an equally effective authorized alternative exists, use decline and safer_option.
5. Otherwise, if the proposed use goes beyond the immediate need, use decline and over_scope.
6. Otherwise, use allow and urgent_exception."""
TEACHING_RULES = LENDING_CODE + """
For each training question, give a decision, its principle, and a one-sentence explanation that cites
the decisive facts. Return only the JSON object. Preserve every supplied ID exactly once."""


@dataclass(frozen=True)
class Lesson:
    id: str
    decision: str
    principle: str
    explanation: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or self.decision not in DECISIONS or self.principle not in PRINCIPLES:
            raise ValueError("Invalid lesson ID, decision, or principle.")
        if not isinstance(self.explanation, str) or not 8 <= len(self.explanation.strip()) <= 400:
            raise ValueError("A lesson needs a short explanation.")

    @property
    def consistent(self) -> bool:
        return PRINCIPLES[self.principle][0] == self.decision


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def lesson_schema(ids: list[str]) -> dict:
    item = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "decision", "principle", "explanation"],
        "properties": {
            "id": {"type": "string", "enum": ids},
            "decision": {"type": "string", "enum": list(DECISIONS)},
            "principle": {"type": "string", "enum": list(PRINCIPLES)},
            "explanation": {"type": "string", "minLength": 8, "maxLength": 400},
        },
    }
    return {
        "type": "object", "additionalProperties": False, "required": ["lessons"],
        "properties": {"lessons": {"type": "array", "minItems": len(ids), "maxItems": len(ids), "items": item}},
    }


def local_lessons(questions: list[Question], port: int, seed: int) -> list[Lesson]:
    return request_lessons(
        [question.id for question in questions],
        TEACHING_RULES,
        [{"id": question.id, "question": question.text} for question in questions],
        port, seed,
    )


def request_lessons(ids: list[str], instructions: str, examples: list[dict], port: int, seed: int) -> list[Lesson]:
    answer, _ = request_json(instructions, examples, lesson_schema(ids), port, seed, max_tokens=850)
    lessons = [Lesson(**row) for row in answer["lessons"]]
    if len(lessons) != len(ids) or {lesson.id for lesson in lessons} != set(ids):
        raise ValueError("The teacher response omitted or repeated a training ID.")
    by_id = {lesson.id: lesson for lesson in lessons}
    return [by_id[identifier] for identifier in ids]


def request_json(instructions: str, material: list | dict, schema: dict, port: int, seed: int, *,
                 max_tokens: int) -> tuple[dict, dict]:
    request = {
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(material)},
        ],
        "response_format": {"type": "json_schema", "schema": schema},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0, "max_tokens": max_tokens, "seed": seed,
    }
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(request).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    ), timeout=180) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("The teacher did not finish its response.")
    return json.loads(choice["message"]["content"]), payload.get("usage", {})


def fixture_lessons(questions: list[Question]) -> list[Lesson]:
    """Deterministic rubric-derived fixture for exercising the pipeline in CI."""
    lessons = []
    for question in questions:
        answer, reason = reference(question.case)
        lessons.append(Lesson(question.id, answer, reason, PRINCIPLES[reason][1]))
    return lessons


def read_lessons(path: Path, *, complete: bool = True) -> tuple[dict, list[tuple[Question, Lesson]]]:
    result = json.loads(path.read_text())
    if result["dataset"] != DATASET_VERSION:
        raise ValueError("Unknown lesson dataset version.")
    questions, _ = split(result["seed"])
    if result["questions_sha256"] != questions_digest(questions):
        raise ValueError("The lesson dataset question manifest has changed.")
    rows = result["lessons"]
    if len(rows) > len(questions) or (complete and len(rows) != len(questions)):
        raise ValueError("Lesson collection is incomplete or exceeds its budget.")
    collected = []
    for question, row in zip(questions, rows):
        stored = dict(row)
        if stored.pop("question") != question.text or stored["id"] != question.id:
            raise ValueError("The lesson file does not match the training-only manifest.")
        collected.append((question, Lesson(**stored)))
    return result, collected


def questions_digest(questions: list[Question]) -> str:
    return hashlib.sha256(json.dumps([(q.id, q.text) for q in questions]).encode()).hexdigest()


def collect(output: Path, seed: int, port: int, teacher: str, fixture: bool) -> dict:
    questions, _ = split(seed)
    provenance = {"kind": "rubric_fixture" if fixture else "local_model", "model": teacher}
    metadata = {
        "dataset": DATASET_VERSION, "seed": seed, "teacher": provenance,
        "questions_sha256": questions_digest(questions), "batch_size": BATCH_SIZE,
        "prompt_sha256": hashlib.sha256(TEACHING_RULES.encode()).hexdigest(),
    }
    if output.exists():
        prior, _ = read_lessons(output, complete=False)
        if any(prior[key] != value for key, value in metadata.items()):
            raise ValueError("The existing lesson file was collected with different settings.")
        result = prior
    else:
        result = metadata | {"lessons": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    for start in range(len(result["lessons"]), len(questions), BATCH_SIZE):
        batch = questions[start:start + BATCH_SIZE]
        lessons = fixture_lessons(batch) if fixture else local_lessons(batch, port, seed)
        result["lessons"].extend(
            asdict(lesson) | {"question": question.text} for question, lesson in zip(batch, lessons, strict=True)
        )
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=DEFAULT_PORT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teacher", help="record the local model name and revision")
    source.add_argument("--fixture", action="store_true", help="generate labelled fixtures for pipeline checks")
    args = parser.parse_args()
    try:
        result = collect(args.output, args.seed, args.port, args.teacher or "lending-rubric", args.fixture)
    except (OSError, URLError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Teacher collection stopped: {error}\n")
    print(f"Collected {len(result['lessons'])} lessons from {result['teacher']['model']}.")


if __name__ == "__main__":
    main()
