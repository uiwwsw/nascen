"""Matched trajectory observations and held-out Korean action interventions."""

from dataclasses import dataclass
from itertools import product
from random import Random

from .dataset import Episode, Question
from .world import ACTIONS, FAMILY, STATES, TRAIN_ITEMS, describe_action, describe_state, transition

TEST_ITEMS = {
    "physical": ("고무 원판", "설탕 통", "소형 플라스틱 바퀴", "참나무 판자", "아크릴 큐브", "자작나무 통"),
    "social": ("긴급 조명등", "전자 온도계", "지도판", "구조용 조끼", "모의 열화상 카메라", "야외 방송 장치"),
}

ONE_STEP = (
    ("physical", (False, True, False), ("push",), ("no_push",)),
    ("physical", (True, True, False), ("block",), ("advance",)),
    ("physical", (False, False, False), ("unblock",), ("push",)),
    ("social", (False, False, False), ("owner_grants",), ("peer_grants",)),
    ("social", (False, False, False), ("urgent_now",), ("urgent_before",)),
    ("social", (True, False, False), ("use_in",), ("use_out",)),
)

HELD_ORDERS = (
    ("physical", (False, False, False), ("push", "unblock", "advance"), ("unblock", "push", "advance")),
    ("physical", (False, True, False), ("push", "block", "unblock"), ("block", "unblock", "push")),
    ("physical", (False, True, False), ("block", "push", "advance"), ("push", "advance", "block")),
    ("physical", (True, True, False), ("advance", "block", "unblock"), ("block", "unblock", "advance")),
    ("social", (False, False, False), ("owner_grants", "use_in", "urgent_before"),
     ("use_in", "owner_grants", "urgent_before")),
    ("social", (False, False, False), ("urgent_now", "use_in", "urgent_before"),
     ("urgent_now", "urgent_before", "use_in")),
    ("social", (False, False, False), ("use_in", "urgent_now", "peer_grants"), ("urgent_now", "use_in", "peer_grants")),
    ("social", (False, False, False), ("urgent_before", "use_in", "owner_grants"),
     ("urgent_before", "owner_grants", "use_in")),
)


@dataclass(frozen=True)
class Key:
    id: str
    group: str
    skill: str
    world: str
    state_style: int
    action_style: int
    initial: tuple[bool, ...]
    outcomes: tuple[tuple[bool, ...], ...]
    choices: tuple[tuple[tuple[bool, ...], ...], ...]
    answers: tuple[int, ...]


def training_windows(size: int, seed: int = 42) -> tuple[tuple[Episode, ...], ...]:
    if size < 2 or size % 2:
        raise ValueError("Use a positive, even number of training windows.")
    rng = Random(seed)
    setup = []
    for world, actions in ACTIONS.items():
        held = {order for domain, _, first, second in HELD_ORDERS if world == domain for order in (first, second)}
        pool = [
            (item, state, order)
            for item in TRAIN_ITEMS[world]
            for state in STATES
            for order in product(actions, repeat=3)
            if len(set(order)) > 1 and order not in held
        ]
        if size // 2 > len(pool):
            raise ValueError("Training budget exceeds the available distinct windows.")
        setup.append([(world, *row) for row in rng.sample(pool, size // 2)])
    windows = []
    for index, (world, item, state, order) in enumerate(row for pair in zip(*setup, strict=True) for row in pair):
        style = rng.randrange(2)
        episodes = []
        for step, action in enumerate(order):
            result = transition(state, action)
            episodes.append(Episode(
                f"train-{index:04}-{step}", describe_state(world, item, state, style),
                describe_action(action, style), describe_state(world, item, result, style),
                tuple(describe_state(world, item, option, style) for option in STATES if option != result),
                describe_state(world, item, state, 1 - style), describe_action(action, 1 - style),
                FAMILY[action],
            ))
            state = result
        windows.append(tuple(episodes))
    return tuple(windows)


def held_out_exam() -> tuple[tuple[Question, ...], tuple[Key, ...]]:
    questions, keys = [], []
    for skill, recipes in (("one_step", ONE_STEP), ("order", HELD_ORDERS)):
        for index, (world, initial, first, second) in enumerate(recipes):
            for item_index, item in enumerate(TEST_ITEMS[world]):
                group = f"{skill}-{index}-{item_index}"
                shuffled = []
                for step in range(len(first)):
                    options = list(STATES)
                    Random(f"{group}-{step}").shuffle(options)
                    shuffled.append(tuple(options))
                choices = tuple(shuffled)
                for state_style, action_style in product((0, 2), repeat=2):
                    text = tuple(tuple(describe_state(world, item, option, state_style) for option in options)
                                 for options in choices)
                    initial_text = describe_state(world, item, initial, state_style)
                    for side, order in enumerate((first, second)):
                        state = initial
                        outcomes = []
                        for action in order:
                            state = transition(state, action)
                            outcomes.append(state)
                        identifier = f"{group}-{state_style}{action_style}-{side}"
                        actions = tuple(describe_action(action, action_style) for action in order)
                        questions.append(Question(identifier, initial_text, actions, text))
                        indices = tuple(options.index(result) for options, result in zip(choices, outcomes, strict=True))
                        keys.append(Key(identifier, group, skill, world, state_style, action_style,
                                        initial, tuple(outcomes), choices, indices))
    return tuple(questions), tuple(keys)


def observed_priors(questions: tuple[Question, ...], keys: tuple[Key, ...]) -> tuple[Question, ...]:
    """Privileged diagnostic only. Never called to produce free-run predictions."""
    diagnostics = []
    for question, key in zip(questions, keys, strict=True):
        prior = question.before
        for index, (action, candidates, correct) in enumerate(zip(question.actions, question.options, key.answers, strict=True)):
            diagnostics.append(Question(f"{question.id}/{index}", prior, (action,), (candidates,)))
            prior = candidates[correct]
    return tuple(diagnostics)


def structure_coverage(windows, questions, keys) -> dict:
    """Audit compositional novelty without handing simulator fields to the student."""
    action_lookup = {(world, describe_action(action, style)): action
                     for world, actions in ACTIONS.items() for action in actions for style in range(3)}
    state_lookup = {describe_state(world, item, state, style): (world, state)
                    for world, items in TRAIN_ITEMS.items() for item in items
                    for state in STATES for style in (0, 1)}
    trained = set()
    for window in windows:
        for row in window:
            world, state = state_lookup[row.before]
            trained.add((world, state, action_lookup[world, row.action]))
    required = {
        (key.world, state, action_lookup[key.world, action])
        for question, key in zip(questions, keys, strict=True)
        for state, action in zip((key.initial, *key.outcomes[:-1]), question.actions, strict=True)
    }
    return {"training_structures": len(trained), "exam_structures": len(required),
            "exam_structures_seen_in_training": len(required & trained)}
