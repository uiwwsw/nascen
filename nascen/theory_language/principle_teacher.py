"""Collect Korean principle explanations without sending observations or tests to the teacher."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from ..teacher import DEFAULT_PORT, request_json
from .dataset import digest, read_json, write_json
from .principle_data import Lesson, held_rules, reference, study_manifest

PROMPT = """Rewrite the five Korean fields of this simulated lamp rule in natural Korean.
Preserve exactly the polarity, each condition, and whether one or both conditions must hold.
The exception overrides the other conditions. The counterexample must illustrate that override.
Keep the same fields, mention no new state, and return only the JSON object.
Make each field a natural Korean sentence. Preserve terms such as 전원, 덮개, 코드, 정비."""


def collect(output: Path, teacher: str, port=DEFAULT_PORT, seed=42):
    manifest = study_manifest() | {"prompt_sha256": digest(PROMPT), "teacher": {"model": teacher, "seed": seed}}
    if output.exists():
        data = read_json(output)
        if any(data.get(key) != value for key, value in manifest.items()):
            raise ValueError("Teacher file does not match this curriculum or prompt.")
    else:
        data = manifest | {"lessons": {}, "cost": {}}
    for key, rule in held_rules().items():
        if key in data["lessons"]:
            continue
        data["lessons"][key], data["cost"][key] = _request(asdict(reference(rule)), PROMPT, port, seed)
        write_json(output, data)
    return data


def _request(material, prompt, port, seed):
    fields = tuple(Lesson.__dataclass_fields__)
    schema = {"type": "object", "required": list(fields), "additionalProperties": False,
              "properties": {name: {"type": "string"} for name in fields}}
    start = perf_counter()
    raw, usage = request_json(prompt, material, schema, port, seed, max_tokens=440, ensure_ascii=False)
    return asdict(Lesson(**raw)), {"wall_seconds": perf_counter() - start, "usage": usage}


def revise(output, teacher, key, feedback, port=DEFAULT_PORT):
    data = collect(output, teacher, port)
    previous = data["lessons"][key]
    prompt = PROMPT + "\nThe prior answer has a verified error. Apply the review using the reference as authoritative."
    material = {"reference": asdict(reference(held_rules()[key])), "previous": previous, "review": feedback}
    lesson, cost = _request(material, prompt, port, data["teacher"]["seed"])
    data.setdefault("revisions", []).append({"lesson": key, "previous": previous, "cost": data["cost"][key],
                                           "feedback": feedback, "prompt_sha256": digest(prompt)})
    data["lessons"][key], data["cost"][key] = lesson, cost
    write_json(output, data)
    return data


def load_lessons(packet, review):
    if any(packet.get(key) != value for key, value in (study_manifest() | {"prompt_sha256": digest(PROMPT)}).items()):
        raise ValueError("Teacher file does not match this curriculum or prompt.")
    keys = held_rules().keys()
    if packet["lessons"].keys() != keys or packet["cost"].keys() != keys or review["lessons"].keys() != keys:
        raise ValueError("Review and cost records are required for all held lessons.")
    if review["collection_sha256"] != digest(packet):
        raise ValueError("Review must match the exact teacher collection.")
    outcomes = {key: entry for key, entry in review["lessons"].items()
                if entry["accepted"] is True and entry["reason"]}
    if outcomes.keys() != keys:
        raise ValueError("Every held lesson needs an accepted review before the matched comparison.")
    return {name: Lesson(**row) for name, row in packet["lessons"].items()}
