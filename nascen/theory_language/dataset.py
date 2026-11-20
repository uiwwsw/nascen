"""Deterministic episode budgets and separate, structure-held-out exam manifests."""

from dataclasses import asdict, dataclass
from itertools import chain
import json
from pathlib import Path
from random import Random

from ..semantic_dataset import digest, write_json
from .world import ACTIONS, EXAM_ITEMS, FAMILY, STATES, TRAIN_ITEMS, describe_action, describe_state, transition

VERSION = "theory-ko-v1"
BUDGETS = (100, 300, 1000)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    before: str
    action: str
    after: str
    negatives: tuple[str, ...]
    equivalent_before: str
    equivalent_action: str
    lesson: str
    source: str = "simulated_observation"
    principle: str | None = None
    conditions: str | None = None
    principle_source: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    before: str
    actions: tuple[str, ...]
    options: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Answer:
    id: str
    steps: tuple[int, ...]
    skill: str
    group: str


def held_structure(world, state, action):
    return (STATES.index(state) + 3 * ACTIONS[world].index(action)) % len(STATES) == 0


def _alternatives(state):
    return tuple(tuple(not value if i == changed else value for i, value in enumerate(state))
                 for changed in range(3))


def training_pool(seed=42) -> tuple[Episode, ...]:
    rng = Random(seed)
    groups = {}
    for world, actions in ACTIONS.items():
        for action in actions:
            group = [
                (world, item, state, action)
                for item in TRAIN_ITEMS[world] for state in STATES if not held_structure(world, state, action)
            ]
            rng.shuffle(group)
            groups[action] = iter(group)
    # Every prefix includes examples of all action variants; smaller budgets are nested in larger ones.
    episodes = []
    for i, (world, item, state, action) in enumerate(
        chain.from_iterable(zip(*groups.values(), strict=True))
    ):
        style = rng.randrange(2)
        result = transition(state, action)
        negatives = tuple(describe_state(world, item, wrong, style) for wrong in _alternatives(result))
        episodes.append(Episode(
            f"e{i:04}", describe_state(world, item, state, style), describe_action(action, style),
            describe_state(world, item, result, style), negatives,
            describe_state(world, item, state, 1 - style), describe_action(action, 1 - style), FAMILY[action],
        ))
    return tuple(episodes)


def exam() -> tuple[tuple[Question, ...], tuple[Answer, ...]]:
    questions = []
    answers = []

    def add(world, item, state, actions, skill, group, style):
        identifier = f"q{len(questions):03}"
        rng = Random(identifier)
        before = describe_state(world, item, state, style)
        options = []
        keys = []
        for action in actions:
            state = transition(state, action)
            candidates = [state, *_alternatives(state)]
            rng.shuffle(candidates)
            options.append(tuple(describe_state(world, item, value, style) for value in candidates))
            keys.append(candidates.index(state))
        questions.append(Question(identifier, before, tuple(describe_action(action, style) for action in actions),
                                  tuple(options)))
        answers.append(Answer(identifier, tuple(keys), skill, group))

    for world, actions in ACTIONS.items():
        for i, action in enumerate(actions):
            held_state = next(state for state in STATES if held_structure(world, state, action))
            known_state = next(state for state in STATES if not held_structure(world, state, action))
            for j in (0, 1):
                item = EXAM_ITEMS[world][(i + j) % len(EXAM_ITEMS[world])]
                add(world, item, held_state, (action,), "transfer", f"{world}-{action}-transfer-{j}", 0)
            group = f"{world}-{action}-paraphrase"
            for style in (0, 2):
                add(world, EXAM_ITEMS[world][i], known_state, (action,), "paraphrase", group, style)

    contrasts = (
        ("physical", "negation", (False, True, False), ("push", "no_push")),
        ("physical", "condition", (True, True, False), ("unblock", "block")),
        ("social", "subject", (False, False, False), ("owner_grants", "peer_grants")),
        ("social", "time", (False, False, False), ("urgent_now", "urgent_before")),
        ("social", "condition", (True, False, False), ("use_in", "use_out")),
    )
    for world, skill, state, actions in contrasts:
        for i in range(3):
            for action in actions:
                add(world, EXAM_ITEMS[world][i], state, (action,), skill, f"{world}-{skill}-{i}", 2)

    chains = (
        ("physical", (False, False, False), ("unblock", "push", "advance")),
        ("physical", (False, True, False), ("push", "block", "advance")),
        ("physical", (True, True, False), ("advance", "block")),
        ("social", (False, False, False), ("owner_grants", "use_in")),
        ("social", (False, False, False), ("urgent_now", "use_in")),
        ("social", (False, False, False), ("peer_grants", "urgent_before", "use_in")),
        ("social", (True, False, False), ("urgent_before", "use_out")),
    )
    for j, (world, state, actions) in enumerate(chains):
        for i in range(2):
            add(world, EXAM_ITEMS[world][(i + j) % len(EXAM_ITEMS[world])], state, actions,
                "composition", f"{world}-chain-{j}-{i}", 2)

    outside = (
        ("얼음 조각을 실내 그릇에 담아 뒀다.", "오랜 시간 실온에 놓아뒀다.",
         ("그릇에 물이 고였다.", "얼음이 더 커졌다.", "그릇에 소금만 남았다.", "물이 다시 얼었다."), 0),
        ("종이 클립과 나무 조각이 나란히 놓여 있다.", "두 물체 가까이 자석을 가져갔다.",
         ("나무 조각만 자석에 붙었다.", "두 물체가 모두 자석에서 멀어졌다.", "종이 클립만 자석에 붙었다.", "두 물체가 모두 녹았다."), 2),
    )
    for before, action, options, answer in outside:
        identifier = f"q{len(questions):03}"
        questions.append(Question(identifier, before, (action,), (options,)))
        answers.append(Answer(identifier, (answer,), "outside_knowledge", identifier))
    return tuple(questions), tuple(answers)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(directory: Path, seed=42, *, size=max(BUDGETS), lessons=None):
    pool = training_pool(seed)
    if not 1 <= size <= len(pool):
        raise ValueError(f"Choose a dataset size between 1 and {len(pool)}.")
    training = []
    for episode in pool[:size]:
        record = asdict(episode)
        lesson = (lessons or {}).get(episode.lesson)
        if lesson is not None:
            record.update(principle=lesson.principle, conditions=lesson.conditions, principle_source="reviewed_teacher")
        training.append(record)
    questions, answers = exam()
    manifest = {"dataset": VERSION, "seed": seed, "training_sha256": digest(training),
                "exam_sha256": digest([asdict(row) for row in questions]),
                "answers_sha256": digest([asdict(row) for row in answers])}
    write_json(directory / "train.json", training)
    write_json(directory / "exam.json", [asdict(row) for row in questions])
    write_json(directory / "answers.json", [asdict(row) for row in answers])
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_partition(directory: Path, name: str):
    manifest = read_json(directory / "manifest.json")
    rows = read_json(directory / f"{name}.json")
    hash_key = "training_sha256" if name == "train" else f"{name}_sha256"
    if manifest["dataset"] != VERSION or manifest[hash_key] != digest(rows):
        raise ValueError(f"The {name} partition does not match its dataset manifest.")
    record = {"train": Episode, "exam": Question, "answers": Answer}[name]
    return tuple(record(**row) for row in rows)
