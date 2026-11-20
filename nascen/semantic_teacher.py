"""Annotate only the training partition with a local teacher and audit its output."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

from .semantic_dataset import (
    PRINCIPLES, RELATION_PAIRS, RELATIONS, VERSION, budget_scenes, digest, splits, write_json,
)
from .teacher import DEFAULT_PORT, request_json

PROMPT_VERSION = "ko-semantic-v5"
RELATION_PROMPT = """두 한국어 문장 사이의 관계를 분류하고 지정된 JSON만 반환하라.
equivalent는 뜻이 같다, related는 소재만 관련된다, contradiction은 동시에 참일 수 없다,
entailment는 첫 문장이 둘째 문장을 논리적으로 포함한다, cause는 원인과 실제 결과,
condition은 일어나기 위한 전제, exception은 일반 규칙의 예외를 뜻한다.
시제와 부정, 행위자, 허락 범위를 구분하라. ID를 각각 한 번만 사용하라."""
PRINCIPLE_PROMPT = """Rewrite the supplied Korean reference rule into five short Korean sentences. Treat the reference as authoritative and use the two situations only to disambiguate the branches. Use generic actors and objects. For 'when' and 'otherwise_when', directly contrast the applicable conditions. Don't copy a common premise into both fields. For 'then' and 'otherwise_then', state what happens in each branch, preserving negation. Write the explanation as a single complete Korean sentence covering both branches. Use natural Korean plain style and the requested JSON fields.
Example for a different situation: {"when":"스위치가 켜져 있다.","then":"조명이 들어온다.","otherwise_when":"스위치가 꺼져 있다.","otherwise_then":"조명이 들어오지 않는다.","explanation":"스위치를 켜야 조명이 들어오며, 끈 상태에서는 조명이 들어오지 않는다."}"""
DISTINCT_BRANCHES = """
반드시 두 사례의 두 번째 맥락 문장을 비교하라. 두 사례에 공통인 조건은 구분 기준이 아니다.
when과 otherwise_when은 무엇이 달라지는지 직접 밝혀야 한다. then과 otherwise_then도 달라야 한다."""


@dataclass(frozen=True)
class PrincipleLesson:
    when: str
    then: str
    otherwise_when: str
    otherwise_then: str
    explanation: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not isinstance(value, str) or not 5 <= len(value.strip()) <= 200 or "\n" in value:
                raise ValueError("Each principle field must be a short sentence.")
        if self.when.strip() == self.otherwise_when.strip() or self.then.strip() == self.otherwise_then.strip():
            raise ValueError("A principle must distinguish its two branches.")


def _object_schema(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def _classify(pairs, port: int, seed: int):
    schema = _object_schema({"pairs": {
        "type": "array", "minItems": len(pairs), "maxItems": len(pairs),
        "items": _object_schema({
            "id": {"type": "string", "enum": [pair.id for pair in pairs]},
            "relation": {"type": "string", "enum": list(RELATIONS)},
        }),
    }})
    payload = [{"id": pair.id, "first": pair.first, "second": pair.second} for pair in pairs]
    raw, usage = request_json(RELATION_PROMPT, payload, schema, port, seed, max_tokens=640, ensure_ascii=False)
    rows = raw["pairs"]
    answers = {row["id"]: row["relation"] for row in rows}
    if len(answers) != len(rows) or answers.keys() != {pair.id for pair in pairs} or not set(answers.values()) <= set(RELATIONS):
        raise ValueError("Invalid teacher relation set.")
    return answers, usage


def _extract(rows, reference: str, port: int, seed: int):
    schema = _object_schema({name: {"type": "string"} for name in PrincipleLesson.__dataclass_fields__})
    payload = {
        "reference_rule": reference, "positive_example": rows[0].context, "contrast_example": rows[1].context,
    }
    attempts = []
    for prompt in (PRINCIPLE_PROMPT, PRINCIPLE_PROMPT + DISTINCT_BRANCHES):
        start = perf_counter()
        raw, usage = request_json(prompt, payload, schema, port, seed, max_tokens=480, ensure_ascii=False)
        attempts.append({"wall_seconds": perf_counter() - start, "usage": usage})
        try:
            return PrincipleLesson(**raw), attempts
        except (TypeError, ValueError) as error:
            attempts[-1]["validation_failure"] = str(error)
    return None, attempts


def lesson_manifest() -> dict:
    train, _ = splits()
    return {
        "dataset": VERSION, "prompt_version": PROMPT_VERSION,
        "prompt_sha256": digest([RELATION_PROMPT, PRINCIPLE_PROMPT, DISTINCT_BRANCHES]),
        "training_sha256": digest([asdict(row) for row in train] + [asdict(pair) for pair in RELATION_PAIRS]),
        "principle_references_sha256": digest(PRINCIPLES),
    }


def collect(output: Path, model: str, port: int, seed: int) -> dict:
    train, _ = splits()
    manifest = lesson_manifest() | {"teacher": {"model": model, "seed": seed, "kind": "local_model"}}
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if any(result.get(key) != value for key, value in manifest.items()):
            raise ValueError("The existing collection belongs to another training manifest or teacher.")
    else:
        result = manifest | {"relations": {}, "principles": {}, "cost": {}}
    for start in range(0, len(RELATION_PAIRS), 7):
        batch = RELATION_PAIRS[start:start + 7]
        if all(pair.id in result["relations"] for pair in batch):
            continue
        clock = perf_counter()
        answers, usage = _classify(batch, port, seed)
        result["relations"].update(answers)
        result["cost"][f"relations-{start}"] = [{"wall_seconds": perf_counter() - clock, "usage": usage}]
        write_json(output, result)

    examples = budget_scenes(train, 1)
    for key in PRINCIPLES:
        if key in result["principles"]:
            continue
        lesson, attempts = _extract([row for row in examples if row.principle == key], PRINCIPLES[key], port, seed)
        result["cost"].setdefault(key, []).extend(attempts)
        if lesson is None:
            write_json(output, result)
            raise ValueError(f"{key}: invalid principle after {len(attempts)} attempts: {attempts[-1]['validation_failure']}")
        result["principles"][key] = asdict(lesson)
        write_json(output, result)
    return result


def load_lessons(data: dict) -> tuple[dict[str, PrincipleLesson], dict]:
    if any(data.get(key) != value for key, value in lesson_manifest().items()):
        raise ValueError("The teacher collection does not match the current training manifest.")
    labels = data["relations"]
    if labels.keys() != {pair.id for pair in RELATION_PAIRS} or not set(labels.values()) <= set(RELATIONS):
        raise ValueError("The relation collection is incomplete or invalid.")
    if data["principles"].keys() != PRINCIPLES.keys():
        raise ValueError("The principle collection is incomplete or invalid.")
    expected_batches = {*PRINCIPLES, *(f"relations-{start}" for start in range(0, len(RELATION_PAIRS), 7))}
    if data["cost"].keys() != expected_batches:
        raise ValueError("The teacher cost log is incomplete.")
    lessons = {key: PrincipleLesson(**row) for key, row in data["principles"].items()}
    rows = [{"id": pair.id, "reference": pair.reference, "teacher": labels[pair.id]} for pair in RELATION_PAIRS]
    errors = sum(row["reference"] != row["teacher"] for row in rows)
    return lessons, {"reviewed": len(rows), "errors": errors, "error_rate": errors / len(rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher", required=True, help="local model name and revision or weight digest")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = collect(args.output, args.teacher, args.port, args.seed)
        _, audit = load_lessons(result)
    except (OSError, TypeError, KeyError, ValueError) as error:
        parser.exit(1, f"Semantic teacher collection stopped: {error}\n")
    print(f"{len(result['principles'])} principles, {audit['errors']}/{audit['reviewed']} relation disagreements.")


if __name__ == "__main__":
    main()
