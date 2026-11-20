"""Compile a teacher's principle into constraints for an independent student."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn.functional import cross_entropy

from .consent import DECISIONS, OWNER_STATES, PRINCIPLES, SETTINGS, Case
from .distillation import EPOCHS, Student, student_checkpoint, write_json
from .teacher import LENDING_CODE, request_json

DATASET = "lending-principle-v1"
FIELD_VALUES = {
    "owner": OWNER_STATES,
    "urgent": ("no", "yes"),
    "alternative": ("no", "yes"),
    "limited": ("no", "yes"),
}
FULL_REGION = tuple((1 << len(values)) - 1 for values in FIELD_VALUES.values())
ARM = "learned_principle"

INSTRUCTIONS = LENDING_CODE + """
Teach the policy and its exceptions in one short explanation and an ordered list of guards.
This is a translation of the exercise code, not a request for worked examples.
The first guard whose four fields match wins. Use "any" for a field that the guard does
not test. Each guard combines its fields with AND. Express an OR with two guards in order.
The last guard is a fallback with "any" in all four fields.
urgent means imminent serious harm, alternative means an equally effective authorized
alternative exists, and limited means the use is limited to the immediate need."""
TEACH_REQUEST = {
    "request": "Return one principle with its ordered exceptions. Preserve the exercise's precedence.",
    "owner_values": {
        "agreed": "explicit consent", "refused": "explicit refusal",
        "reachable": "no answer yet and can be asked", "unavailable": "cannot answer in time",
    },
}


@dataclass(frozen=True)
class Rule:
    region: tuple[int, ...]
    decision: str
    principle: str

    @property
    def outcome(self) -> tuple[str, str]:
        return self.decision, self.principle


def rule_schema() -> dict:
    guard = {
        name: {"type": "string", "enum": ["any", *values]} for name, values in FIELD_VALUES.items()
    }
    clause = {
        "type": "object", "additionalProperties": False, "required": ["when", "decision", "principle"],
        "properties": {
            "when": {"type": "object", "additionalProperties": False, "required": list(guard), "properties": guard},
            "decision": {"type": "string", "enum": list(DECISIONS)},
            "principle": {"type": "string", "enum": list(PRINCIPLES)},
        },
    }
    return {
        "type": "object", "additionalProperties": False, "required": ["explanation", "rules"],
        "properties": {
            "explanation": {"type": "string", "minLength": 8, "maxLength": 700},
            "rules": {"type": "array", "minItems": 1, "maxItems": 12, "items": clause},
        },
    }


def teaching_rules(teaching: dict) -> list[Rule]:
    explanation = teaching["explanation"]
    clauses = teaching["rules"]
    if not isinstance(explanation, str) or not 8 <= len(explanation.strip()) <= 700 or not 1 <= len(clauses) <= 12:
        raise ValueError("A teaching needs a short explanation and between 1 and 12 rules.")
    rules = []
    for clause in clauses:
        fields = clause["when"]
        if set(fields) != set(FIELD_VALUES):
            raise ValueError("Each guard must specify all four semantic fields.")
        region = tuple(
            full if fields[name] == "any" else 1 << values.index(fields[name])
            for (name, values), full in zip(FIELD_VALUES.items(), FULL_REGION, strict=True)
        )
        decision, principle = clause["decision"], clause["principle"]
        if decision not in DECISIONS or principle not in PRINCIPLES:
            raise ValueError("Unknown decision or principle label.")
        rules.append(Rule(region, decision, principle))
    return rules


def collect(output: Path, model: str, port: int, seed: int, *,
            revision: dict | None = None) -> dict:
    started = perf_counter()
    schema = rule_schema()
    instructions = INSTRUCTIONS
    material = TEACH_REQUEST
    if revision is not None:
        instructions += """
The first attempt failed on the supplied practice counterexamples. Repair the original
ordered guards. An unconditional owner=unavailable clause shadows every rule after it.
Use the first matching guard to check every counterexample before submitting the repair.
Return a new complete rule list."""
        material = TEACH_REQUEST | revision
    reply, usage = request_json(instructions, material, schema, port, seed, max_tokens=1600)
    teaching_rules(reply)
    packet = {
        "dataset": DATASET, "model": model, "seed": seed,
        "requests": 1, "usage": usage, "teacher_seconds": perf_counter() - started,
        "prompt_sha256": sha256(json.dumps([instructions, material, schema]).encode()).hexdigest(),
        "teaching": reply,
    }
    if revision is not None:
        packet["revision"] = revision
    write_json(output, packet)
    return packet


def read_packet(path: Path) -> dict:
    packet = json.loads(path.read_text())
    if packet["dataset"] != DATASET:
        raise ValueError("Unsupported principle packet.")
    return packet


def reference_packet() -> dict:
    """Hand-transcribed exercise code for isolating the effect of teacher errors."""
    guards = [
        ({"owner": "agreed"}, "allow", "permission"),
        ({"owner": "refused"}, "decline", "refusal"),
        ({"owner": "reachable"}, "ask", "ask_owner"),
        ({"urgent": "no"}, "ask", "ask_owner"),
        ({"alternative": "yes"}, "decline", "safer_option"),
        ({"limited": "no"}, "decline", "over_scope"),
        ({}, "allow", "urgent_exception"),
    ]
    teaching = {
        "explanation": "Respect the owner's decision. If the owner cannot be asked and serious harm is imminent, "
                       "use the item only as needed and only when no equally effective authorized alternative exists.",
        "rules": [
            {"when": {field: guard.get(field, "any") for field in FIELD_VALUES},
             "decision": decision, "principle": principle}
            for guard, decision, principle in guards
        ],
    }
    return {
        "dataset": DATASET, "model": "hand-transcribed exercise code",
        "requests": 0, "usage": {}, "teacher_seconds": 0, "teaching": teaching,
    }


def case_region(case: Case) -> tuple[int, ...]:
    return (1 << OWNER_STATES.index(case.owner), 1 << case.urgent, 1 << case.alternative, 1 << case.limited)


def execute(rules: list[Rule], case: Case) -> tuple[str | None, str | None]:
    facts = case_region(case)
    return next((rule.outcome for rule in rules
                 if all(a & b for a, b in zip(rule.region, facts, strict=True))), (None, None))


def subtract(region: tuple[int, ...], matched: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Remove a contained rectangle by disjointly splitting its unmatched dimensions."""
    remainder = []
    for index, (allowed, taken) in enumerate(zip(region, matched, strict=True)):
        outside = allowed & ~taken
        if outside:
            remainder.append((*matched[:index], outside, *region[index + 1:]))
    return remainder


def constraints(rules: list[Rule]) -> list[Rule]:
    """Resolve first-match precedence into disjoint, partially specified fact regions."""
    uncovered = [FULL_REGION]
    regions = []
    for rule in rules:
        remainder = []
        for region in uncovered:
            matched = tuple(a & b for a, b in zip(region, rule.region, strict=True))
            if all(matched):
                regions.append(Rule(matched, *rule.outcome))
                remainder.extend(subtract(region, matched))
            else:
                remainder.append(region)
        uncovered = remainder
    return regions


def allowed_values(regions: list[Rule]) -> list[torch.Tensor]:
    # A region includes every setting. Owner is one-hot and the other fields are binary.
    return [
        torch.ones(len(regions), len(SETTINGS), dtype=torch.bool),
        *(torch.tensor([
            [bool(rule.region[index] & (1 << value)) for value in range(len(values))]
            for rule in regions
        ]) for index, values in enumerate(FIELD_VALUES.values())),
    ]


def hidden_bounds(student: Student, allowed: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    layer = student.hidden[0]
    settings, owners, *flags = layer.weight.split([len(SETTINGS), len(OWNER_STATES), 1, 1, 1], dim=1)
    contributions = [settings, owners, *(torch.cat((torch.zeros_like(weight), weight), dim=1) for weight in flags)]
    low = layer.bias
    high = layer.bias
    for mask, weights in zip(allowed, contributions, strict=True):
        permitted = mask.unsqueeze(1)
        low = low + weights.masked_fill(~permitted, torch.inf).amin(dim=-1)
        high = high + weights.masked_fill(~permitted, -torch.inf).amax(dim=-1)
    return low.tanh(), high.tanh()


def guarded_loss(head: torch.nn.Linear, low: torch.Tensor, high: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = head.weight.clamp(min=0).T
    negative = head.weight.clamp(max=0).T
    lower = low @ positive + high @ negative + head.bias
    upper = high @ positive + low @ negative + head.bias
    column = target.unsqueeze(1)
    worst_logits = upper.scatter(1, column, lower.gather(1, column))
    return cross_entropy(worst_logits, target)


def train_principle(regions: list[Rule], seed: int) -> dict:
    student = Student(len(SETTINGS) + len(OWNER_STATES) + 3, seed)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
    masks = allowed_values(regions)
    decisions = torch.tensor([DECISIONS.index(rule.decision) for rule in regions])
    principles = torch.tensor([tuple(PRINCIPLES).index(rule.principle) for rule in regions])
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        low, high = hidden_bounds(student, masks)
        loss = guarded_loss(student.decision, low, high, decisions) + guarded_loss(student.principle, low, high, principles)
        loss.backward()
        optimizer.step()
    return student_checkpoint(student, ARM, seed, 0, True) | {"training_regions": len(regions)}
