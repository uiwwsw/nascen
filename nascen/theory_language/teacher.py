"""Collect reviewable Korean principle descriptions from training experiences only."""

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter

from ..teacher import DEFAULT_PORT, request_json
from .dataset import VERSION, digest, read_json, training_pool, write_json
from .world import REFERENCES

PROMPT_VERSION = "theory-teacher-ko-v2"
PROMPT = """Rewrite each field of the Korean reference packet as a concise Korean sentence.
The five named reference fields are authoritative. Preserve each original condition, state change, and exception.
Use the observations to check the context only; observed facts that are not explicitly in the reference packet
are never conditions. Do not add or reverse conditions. Use generic actors and objects.
Return only the five requested JSON fields, with one Korean string in each."""

REFERENCE_FIELDS = {
    "force": ("물체를 세게 민다.",
              "길이 열려 있으면 강한 밀기가 새 움직임을 만든다.",
              "길이 열려 있고 물체를 세게 민다.",
              "길이 닫혀 있거나 세게 밀지 않으면 새 움직임이 생기지 않으며 이전 상태를 유지한다."),
    "barrier": ("이동 경로에 장애물을 놓거나 치운다.",
                "장애물을 치우면 길이 열리고 놓으면 길이 막힌다.",
                "장애물 배치는 길의 상태와 물체의 움직임에 영향을 준다.",
                "장애물을 놓으면 이미 움직이던 물체는 멈춘다. 치우기만 하면 움직임은 그대로다."),
    "advance": ("실험 시간을 한 단계 진행한다.",
                "열린 길에서 움직이던 물체가 목표선까지 간다.",
                "목표선에 새로 도달하려면 길이 열려 있고 물체가 움직이고 있어야 한다.",
                "길이 막혔거나 물체가 멈춰 있으면 새 도착이 없다. 이전 도착 기록은 유지되고 물체는 멈춘다."),
    "permission": ("소유자 또는 관계없는 동료가 사용을 승인한다.",
                   "소유자의 승인만 물건의 사용 허락으로 기록된다.",
                   "승인한 사람이 물건의 소유자여야 한다.",
                   "소유자가 아닌 사람의 승인은 이전 허락 상태를 바꾸지 않는다."),
    "time": ("현재 또는 이미 끝난 비상상황을 보고한다.",
             "현재 진행 중인 비상상황을 긴급 상태로 기록한다.",
             "비상상황이 지금 발생해야 긴급 상태가 된다.",
             "이미 끝난 비상상황을 보고하면 지금의 긴급 상태를 해제한다."),
    "use": ("참가자가 특정 용도로 훈련 자원을 사용하려 한다.",
            "정해진 범위 안에서 소유자의 허락이나 현재 비상상황이 있으면 새 사용이 기록된다.",
            "허용 범위 안의 요청이어야 하며, 소유자 승인과 현재 비상상황 중 적어도 하나가 필요하다.",
            "범위 밖의 요청, 또는 승인과 비상상황이 모두 없는 요청으로는 새 사용이 기록되지 않는다."),
}


@dataclass(frozen=True)
class Lesson:
    event: str
    cause: str
    conditions: str
    exception: str
    principle: str

    def __post_init__(self):
        for text in asdict(self).values():
            if not isinstance(text, str) or not 5 <= len(text.strip()) <= 320:
                raise ValueError("Each teacher field must contain a short explanation.")

    def texts(self):
        return (self.event, self.cause, self.conditions, self.exception, self.principle)


def material(seed: int) -> dict:
    pool = training_pool(seed)
    return {
        family: {
            "reference": dict(zip(("event", "cause", "conditions", "exception", "principle"),
                                  (*REFERENCE_FIELDS[family], reference), strict=True)),
            "observations": [
                {"before": row.before, "action": row.action, "after": row.after}
                for row in [episode for episode in pool[:100] if episode.lesson == family][:2]
            ],
        }
        for family, reference in REFERENCES.items()
    }


def collect(output: Path, model: str, port: int = DEFAULT_PORT, seed: int = 42) -> dict:
    examples = material(seed)
    manifest = {
        "dataset": VERSION, "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(PROMPT),
        "training_material_sha256": digest(examples), "teacher": {"model": model, "seed": seed},
    }
    if output.exists():
        result = read_json(output)
        if any(result.get(key) != value for key, value in manifest.items()):
            raise ValueError("The collection belongs to a different teacher prompt or source.")
    else:
        result = manifest | {"lessons": {}, "cost": {}}
    fields = list(Lesson.__dataclass_fields__)
    schema = {
        "type": "object", "required": fields, "additionalProperties": False,
        "properties": {field: {"type": "string"} for field in fields},
    }
    for family, payload in examples.items():
        if family in result["lessons"]:
            continue
        start = perf_counter()
        raw, usage = request_json(PROMPT, payload, schema, port, seed, max_tokens=580, ensure_ascii=False)
        lesson = Lesson(**raw)
        result["lessons"][family] = asdict(lesson)
        result["cost"][family] = {"wall_seconds": perf_counter() - start, "usage": usage}
        write_json(output, result)
    return result


def load_reviewed(collection: dict, review: dict):
    if (collection["dataset"] != VERSION or collection["prompt_sha256"] != digest(PROMPT)
            or collection["training_material_sha256"] != digest(material(collection["teacher"]["seed"]))):
        raise ValueError("Unsupported teaching collection or training manifest.")
    if review["collection_sha256"] != digest(collection):
        raise ValueError("The review must match the teaching collection.")
    if any(collection[key].keys() != REFERENCES.keys() for key in ("lessons", "cost")):
        raise ValueError("The teaching collection is incomplete.")
    if review["principles"].keys() != REFERENCES.keys():
        raise ValueError("Review each of the teaching principles.")
    reviewed = {}
    for family, raw in collection["lessons"].items():
        judgment = review["principles"][family]
        if type(judgment["accepted"]) is not bool or not judgment["reason"]:
            raise ValueError("Each review needs an explicit verdict and a reason.")
        lesson = Lesson(**raw)
        if judgment["accepted"]:
            reviewed[family] = lesson
    if len(reviewed) < 2:
        raise ValueError("At least two reviewed principles are needed for a shuffled comparison.")
    return reviewed


def cost(collection: dict):
    requests = tuple(collection["cost"].values())
    return {
        "requests": len(requests), "wall_seconds": sum(row["wall_seconds"] for row in requests),
        **{key: sum(row["usage"][key] for row in requests) if all(key in row["usage"] for row in requests) else None
           for key in ("prompt_tokens", "completion_tokens")},
    }
