"""An independently specified finite world for composing state-change principles."""

from dataclasses import dataclass
from hashlib import sha256
import json
from random import Random

STATES = "abcdefgh"
SCALES = (10, 100, 1000)
WORLD_SEED = 613
CYCLE_LAYOUTS = (
    (2,), (3,), (4,), (5,), (6,), (7,), (8,),
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 3), (3, 4), (3, 5), (4, 4), (2, 2, 2), (2, 2, 2, 2),
)


@dataclass(frozen=True)
class Principle:
    cycles: tuple[tuple[int, ...], ...]

    def apply(self, state: int) -> int:
        # Simulate each cycle with elementary swaps. The interpreter under test uses tables.
        for pivot, *others in self.cycles:
            for other in others:
                if state == pivot:
                    state = other
                elif state == other:
                    state = pivot
        return state

    def table(self) -> str:
        return "".join(STATES[self.apply(state)] for state in range(len(STATES)))

    def card(self) -> str:
        return " and ".join("cycle " + " → ".join(STATES[state].upper() for state in cycle)
                            for cycle in self.cycles)


def library(size: int = SCALES[-1]) -> list[Principle]:
    if not 1 <= size <= SCALES[-1]:
        raise ValueError(f"Choose a library size between 1 and {SCALES[-1]}.")
    rng = Random(WORLD_SEED)
    rules = []
    seen = set()
    while len(rules) < size:
        values = rng.sample(range(len(STATES)), len(STATES))
        cycles = []
        start = 0
        for length in rng.choice(CYCLE_LAYOUTS):
            cycles.append(tuple(values[start:start + length]))
            start += length
        rule = Principle(tuple(cycles))
        table = rule.table()
        if table not in seen:
            seen.add(table)
            rules.append(rule)
    return rules


@dataclass(frozen=True)
class Problem:
    state: int
    rules: tuple[int, ...]


def trace(rules: list[Principle], problem: Problem) -> tuple[int, ...]:
    state = problem.state
    states = []
    for index in problem.rules:
        state = rules[index].apply(state)
        states.append(state)
    return tuple(states)


def digest(value: list | dict) -> str:
    return sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def annotations_size(value: list) -> int:
    return len(json.dumps(value, separators=(",", ":")).encode())
