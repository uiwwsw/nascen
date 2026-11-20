"""A controlled lamp simulator for teaching novel combinations of conditions and exceptions."""

from dataclasses import asdict, dataclass
from itertools import combinations, product
from random import Random

from .dataset import digest

VERSION = "lamp-principles-ko-v1"
STATES = tuple(product((False, True), repeat=4))
BUDGETS = (0, 2, 4, 8)
_FACTS = (
    (("전원 접점이 끊겨 있다.", "전원 접점이 연결되어 있다."),
     ("전원 연결이 끊어져 있다.", "전원이 연결된 상태다.")),
    (("보호 덮개가 열려 있다.", "보호 덮개가 닫혀 있다."),
     ("안전 덮개가 개방되어 있다.", "안전 덮개가 닫힌 상태다.")),
    (("확인 코드가 검증되지 않았다.", "확인 코드가 검증되었다."),
     ("확인 코드 검사를 통과하지 않았다.", "확인 코드 검사를 통과했다.")),
    (("정비 모드가 꺼져 있다.", "정비 모드가 켜져 있다."),
     ("장치가 정비 상태가 아니다.", "장치가 정비 상태다.")),
)
LOGIC = ("앞선 두 조건 가운데 하나 이상이 만족되어야 한다.", "앞선 두 조건이 모두 만족되어야 한다.")
HELD_STRUCTURES = ((0, 1, 2), (0, 2, 3), (1, 3, 0), (2, 3, 1))
OUTCOMES = ("시험용 표시등이 꺼진 상태로 남는다.", "시험용 표시등이 켜진다.")


@dataclass(frozen=True)
class Rule:
    conditions: tuple[int, int]
    exception: int
    values: tuple[bool, bool, bool]
    both: bool

    @property
    def structure(self):
        return (*self.conditions, self.exception)


@dataclass(frozen=True)
class Lesson:
    first: str
    second: str
    logic: str
    exception: str
    counterexample: str

    def __post_init__(self):
        if any(not isinstance(text, str) or not 4 <= len(text.strip()) <= 240 for text in asdict(self).values()):
            raise ValueError("All five lesson fields must be short, nonempty sentences.")

    def phrases(self):
        return self.first, self.second, self.exception, self.logic, self.counterexample


def active(rule, state):
    criteria = tuple(state[index] == value for index, value in zip(rule.conditions, rule.values[:2], strict=True))
    return (all(criteria) if rule.both else any(criteria))


def outcome(rule, state):
    return active(rule, state) and state[rule.exception] != rule.values[2]


def fact(index, value, style=0):
    return _FACTS[index][style][int(value)]


def reference(rule, style=0):
    first = fact(rule.conditions[0], rule.values[0], style)
    second = fact(rule.conditions[1], rule.values[1], style)
    exception = fact(rule.exception, rule.values[2], style)
    logic = LOGIC[rule.both]
    return Lesson(first, second, logic, exception,
                  f"앞선 조건을 만족해도 다음 상태에서는 표시등이 켜지지 않는다. {exception}")


def training_rules():
    return tuple(
        Rule(pair, exception, values, both)
        for pair in combinations(range(len(_FACTS)), 2)
        for exception in range(len(_FACTS)) if exception not in pair and (*pair, exception) not in HELD_STRUCTURES
        for values in product((False, True), repeat=3)
        for both in (False, True)
    )


def held_rules():
    sign_pairs = (
        ((True, False, True), (False, True, False)),
        ((False, True, True), (True, False, False)),
        ((True, True, False), (False, False, True)),
        ((False, False, True), (True, True, False)),
    )
    return {
        f"lesson-{index * 2 + both + 1:02d}": Rule((left, right), veto, values, bool(both))
        for index, ((left, right, veto), signs) in enumerate(zip(HELD_STRUCTURES, sign_pairs, strict=True))
        for both, values in enumerate(signs)
    }


def observed_facts(state, style, seed):
    order = list(range(len(state)))
    Random(seed).shuffle(order)
    return tuple(fact(index, state[index], style) for index in order), tuple(order)


def split_states():
    # Parity masks the unused fourth fact. Both halves contain every possible
    # setting of the three relevant facts, so both contain positive outcomes.
    support = [s for s in STATES if sum(s) % 2 == 0]
    query = tuple(s for s in STATES if sum(s) % 2 == 1)
    Random(42).shuffle(support)
    return tuple(support), query


def study_manifest():
    return {
        "dataset": VERSION,
        "training_rules_sha256": digest([asdict(rule) for rule in training_rules()]),
        "held_rules_sha256": digest({name: asdict(rule) for name, rule in held_rules().items()}),
    }
