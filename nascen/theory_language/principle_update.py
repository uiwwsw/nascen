"""Propose principle updates while keeping already verified predictions intact."""

from dataclasses import dataclass
from itertools import chain
from unicodedata import normalize

import torch

from .dataset import digest, read_json
from .encoder import encode_partition
from .principle_model import predict, study


@dataclass(frozen=True)
class Evidence:
    key: str
    facts: torch.Tensor
    lit: bool


@dataclass(frozen=True)
class Update:
    adapter: torch.Tensor
    evidence: tuple[Evidence, ...]
    status: str
    conflicts: int
    reason: str


def validate_facts(facts):
    if (not isinstance(facts, (tuple, list)) or len(facts) != 4
            or any(not isinstance(text, str) or not text.strip() for text in facts)):
        raise ValueError("Supply four nonempty Korean fact sentences.")


def evidence_vectors(encoder, rows):
    if not rows:
        return None
    vectors, _ = encode_partition(encoder, chain.from_iterable(rows))
    return torch.stack([torch.stack([vectors[text] for text in row]) for row in rows])


def read_evidence(path, encoder):
    if path is None:
        return ()
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("Observations must be a JSON list of fact and outcome records.")
    unique = {}
    for row in rows:
        validate_facts(row["facts"])
        if type(row["lit"]) is not bool:
            raise ValueError("Each observation needs a Boolean lit outcome.")
        facts = tuple(sorted(normalize("NFC", " ".join(text.split())) for text in row["facts"]))
        if facts in unique and unique[facts] != row["lit"]:
            raise ValueError("Conflicting outcomes were provided for the same facts.")
        unique[facts] = row["lit"]
    vectors = evidence_vectors(encoder, list(unique))
    return tuple(Evidence(digest(facts), vector, lit)
                 for (facts, lit), vector in zip(unique.items(), vectors if vectors is not None else (), strict=True))


def _combine(*collections):
    result = {}
    for row in chain.from_iterable(collections):
        if row.key in result and result[row.key].lit != row.lit:
            raise ValueError("New evidence contradicts the recorded outcome for the same facts.")
        result[row.key] = row
    return tuple(result.values())


def _contradictions(model, adapter, evidence):
    if not evidence:
        return ()
    classified = (predict(model, adapter, torch.stack([row.facts for row in evidence])) >= 0.5).tolist()
    return tuple(row for row, actual in zip(evidence, classified, strict=True) if actual != row.lit)


def verified_study(model, lesson, observations, *, previous=None, recorded=(), verification=()):
    observations = _combine(observations)
    guards = _combine(recorded, verification)
    evidence = _combine(guards, observations)
    baseline = study(model, None, None, None, previous=previous)
    active = study(model, lesson, None, None) if lesson is not None else baseline
    disagreements = _contradictions(model, active, observations)
    conflicts = len(disagreements)

    def finish(adapter, status, reason):
        wrong = {row.key for row in _contradictions(model, adapter, evidence)}
        return Update(adapter, tuple(row for row in evidence if row.key not in wrong), status, conflicts, reason)

    def reject(reason):
        if previous is None and lesson is not None and _contradictions(model, active, guards):
            raise ValueError(reason)
        retained = baseline if previous is not None else active
        return finish(retained, "rejected", reason)

    if not disagreements:
        if _contradictions(model, active, guards):
            return reject("The lesson or checkpoint fails a verified case.")
        status = "learned" if lesson is not None else "unchanged"
        return finish(active, status, "No conflicting observations required a weight update.")

    trained = {row.key for row in observations}
    if not any(row.key not in trained for row in guards):
        if _contradictions(model, active, recorded):
            return reject("The lesson or checkpoint fails a verified case.")
        return finish(active, "verification_needed", "Supply separate verified observations with --verify.")

    inputs = torch.stack([row.facts for row in observations])
    targets = torch.tensor([row.lit for row in observations], dtype=torch.float32)
    candidate = study(model, None, inputs, targets, previous=active)
    if _contradictions(model, candidate, evidence):
        return reject("The proposed update fails a new observation or changes a verified outcome.")
    return finish(candidate, "updated", "The proposal matches the new observations and all verified outcomes.")
