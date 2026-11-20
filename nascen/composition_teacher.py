"""Collect and independently verify transition principles taught by a local model."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from .composition_world import SCALES, STATES, Principle, digest, library
from .distillation import write_json
from .teacher import DEFAULT_PORT, DEFAULT_SEED, request_json

DATASET = "switch-principles-v2"
BATCH_SIZE = 20
INSTRUCTIONS = """Translate the cycle cards of an eight-state machine into short executable principles.
Write the states of each cycle in order using lowercase letters and no spaces.
For a card with two or more cycles, separate the cycles with '/'.
For example "cycle B → E → A and cycle G → D" gives "bea/gd".
Leave unspecified states out. Do not calculate a destination table or add a repeated last letter.
Return one string per card in the same order inside the JSON object's programs array."""


def schema(count: int) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["programs"],
        "properties": {"programs": {"type": "array", "minItems": count, "maxItems": count,
                                    "items": {"type": "string", "pattern": "^[a-h]{2,8}(/[a-h]{2,8}){0,3}$"}}},
    }


def program_table(candidate: str) -> str:
    cycles = candidate.split("/")
    members = "".join(cycles)
    if any(len(cycle) < 2 for cycle in cycles) or set(members) - set(STATES) or len(set(members)) != len(members):
        raise ValueError("Cycles need distinct valid states.")
    destinations = dict(zip(STATES, STATES, strict=True))
    for cycle in cycles:
        destinations.update(zip(cycle, cycle[1:] + cycle[0], strict=True))
    return "".join(destinations.values())


def verified_table(rule: Principle, program: str) -> str | None:
    if not isinstance(program, str):
        return None
    try:
        table = program_table(program)
    except ValueError:
        return None
    if table == rule.table():
        return table
    return None


def teaching_prefix(record: dict, rules: list[Principle], size: int) -> list[dict]:
    if record.get("dataset") != DATASET:
        raise ValueError("Unsupported composition teaching dataset.")
    if record.get("manifest_sha256") != digest([rule.card() for rule in rules]):
        raise ValueError("The principle manifest changed.")
    batches = []
    consumed = 0
    for batch in record["batches"]:
        if consumed == size:
            break
        candidates = batch["programs"]
        if batch["start"] != consumed or not 0 < len(candidates) <= min(size - consumed, BATCH_SIZE):
            raise ValueError("The teacher record has a missing, overlapping, or oversized batch.")
        consumed += len(candidates)
        batches.append(batch)
    if consumed != size:
        raise ValueError(f"Only {consumed} of {size} required principles have teacher proposals.")
    return batches


def verify(record: dict, rules: list[Principle], size: int) -> tuple[list[str], dict]:
    started = perf_counter()
    batches = teaching_prefix(record, rules, size)
    proposals = [row for batch in batches for row in batch["programs"]]
    checked = [verified_table(rule, proposal) for rule, proposal in zip(rules[:size], proposals, strict=True)]
    # A rejected lesson has no influence on the models. Fill the same ID from the independent simulator.
    tables = [table if table is not None else rule.table() for rule, table in zip(rules[:size], checked, strict=True)]
    rejected = [index for index, table in enumerate(checked) if table is None]
    return tables, {
        "accepted": size - len(rejected), "simulator_fills": len(rejected), "rejected_ids": rejected,
        "reference_state_slots": size * len(STATES), "verification_wall_seconds": perf_counter() - started,
        "teacher_calls": len(batches),
        "teacher_wall_seconds": sum(batch["wall_seconds"] for batch in batches),
        "prompt_tokens": sum(batch["usage"]["prompt_tokens"] for batch in batches),
        "completion_tokens": sum(batch["usage"]["completion_tokens"] for batch in batches),
        "cached_prompt_tokens": sum(batch["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                    for batch in batches),
    }


def collect(output: Path, model: str, port: int, seed: int, size: int) -> dict:
    rules = library()
    metadata = {
        "dataset": DATASET, "model": model, "seed": seed, "batch_size": BATCH_SIZE,
        "manifest_sha256": digest([rule.card() for rule in rules]),
        "instructions_sha256": digest({"instructions": INSTRUCTIONS}),
    }
    if output.exists():
        record = json.loads(output.read_text())
        if any(record.get(key) != value for key, value in metadata.items()):
            raise ValueError("The existing collection used different settings.")
    else:
        record = metadata | {"batches": []}
    start = sum(len(batch["programs"]) for batch in record["batches"])
    teaching_prefix(record, rules, start)
    while start < size:
        boundary = next(scale for scale in SCALES if scale > start)
        end = min(start + BATCH_SIZE, boundary, size)
        prompt = {"cards": [rule.card() for rule in rules[start:end]], "required_count": end - start}
        started = perf_counter()
        reply, usage = request_json(INSTRUCTIONS, prompt, schema(end - start), port, seed, max_tokens=700)
        elapsed = perf_counter() - started
        rows = reply["programs"]
        if not isinstance(rows, list) or len(rows) != end - start:
            raise ValueError("The teacher omitted or added positions in the response.")
        record["batches"].append({"start": start, "programs": rows, "usage": usage, "wall_seconds": elapsed})
        write_json(output, record)
        start = end
        if start in SCALES or start % 100 == 0:
            _, audit = verify(record, rules, start)
            print(f"Collected {start}/{size}; verified {audit['accepted']}/{start}.", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", required=True, help="record the local teacher model and revision")
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--size", type=int, choices=SCALES, default=SCALES[-1])
    args = parser.parse_args()
    try:
        collect(args.output, args.teacher, args.port, args.seed, args.size)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Teaching stopped: {error}\n")


if __name__ == "__main__":
    main()
