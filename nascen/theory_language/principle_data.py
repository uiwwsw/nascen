"""A controlled lamp simulator for teaching novel combinations of conditions and exceptions."""

from dataclasses import asdict, dataclass
from itertools import combinations, product
from random import Random

from .dataset import digest

VERSION = "lamp-principles-ko-v1"
LAMP_DOMAIN = "lamp"
TRACK_DOMAIN = "track-motion"
DOMAINS = (LAMP_DOMAIN, TRACK_DOMAIN)
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
_TRACK_FACTS = (
    (("물체가 멈춰 있다.", "물체가 움직이고 있다."),
     ("실험 물체는 정지 상태다.", "실험 물체는 이동 중이다.")),
    (("이동 경로가 막혀 있다.", "이동 경로가 열려 있다."),
     ("앞길에 장애물이 있다.", "앞길에 장애물이 없다.")),
    (("아직 목표선에 도착한 적이 없다.", "이미 목표선에 도착한 적이 있다."),
     ("목표선을 지난 이력이 없다.", "목표선을 지난 이력이 있다.")),
    (("실험자가 물체를 세게 밀지 않았다.", "실험자가 물체를 세게 밀었다."),
     ("담당자가 물체에 미는 힘을 가하지 않았다.", "담당자가 물체를 힘껏 떠밀었다.")),
)
LOGIC = ("앞선 두 조건 가운데 하나 이상이 만족되어야 한다.", "앞선 두 조건이 모두 만족되어야 한다.")
HELD_STRUCTURES = ((0, 1, 2), (0, 2, 3), (1, 3, 0), (2, 3, 1))
TRACK_HELD_STRUCTURE = (1, 3, 2)
OUTCOMES = ("시험용 표시등이 꺼진 상태로 남는다.", "시험용 표시등이 켜진다.")
TRACK_OUTCOMES = ("물체가 멈춰 있다.", "물체가 움직인다.")


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


def fact(index, value, style=0, *, domain=LAMP_DOMAIN):
    vocabulary = {LAMP_DOMAIN: _FACTS, TRACK_DOMAIN: _TRACK_FACTS}[domain]
    return vocabulary[index][style][int(value)]


def reference(rule, style=0, *, domain=LAMP_DOMAIN):
    first = fact(rule.conditions[0], rule.values[0], style, domain=domain)
    second = fact(rule.conditions[1], rule.values[1], style, domain=domain)
    exception = fact(rule.exception, rule.values[2], style, domain=domain)
    logic = LOGIC[rule.both]
    if domain == TRACK_DOMAIN:
        return Lesson(first, second, logic, exception,
                      f"앞선 조건을 만족해도 다음 상태에서는 물체가 움직이지 않는다. {exception}")
    return Lesson(first, second, logic, exception,
                  f"앞선 조건을 만족해도 다음 상태에서는 표시등이 켜지지 않는다. {exception}")


def training_rules(domain=LAMP_DOMAIN):
    held = {LAMP_DOMAIN: HELD_STRUCTURES, TRACK_DOMAIN: (TRACK_HELD_STRUCTURE,)}[domain]
    return tuple(
        Rule(pair, exception, values, both)
        for pair in combinations(range(len(_FACTS)), 2)
        for exception in range(len(_FACTS)) if exception not in pair and (*pair, exception) not in held
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


def observed_facts(state, style, seed, *, domain=LAMP_DOMAIN):
    order = list(range(len(state)))
    Random(seed).shuffle(order)
    return tuple(fact(index, state[index], style, domain=domain) for index in order), tuple(order)


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
