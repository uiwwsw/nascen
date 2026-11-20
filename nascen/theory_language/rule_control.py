"""D: retrieve textual state/action prototypes and execute reviewed theory programs."""

import torch

from .world import ACTIONS, FAMILY, STATES, describe_action, describe_state


# Compiled by hand from the reviewed teacher lessons for the stipulated worlds.
# This control never invokes the generator's transition function.
RULE_EFFECTS = {
    "push": ("moving", "or", 0, 1),
    "no_push": ("preserve",),
    "unblock": ("set", 1, True),
    "block": ("clear_motion_and_path",),
    "advance": ("finish_motion",),
    "owner_grants": ("set", 0, True),
    "peer_grants": ("preserve",),
    "urgent_now": ("set", 1, True),
    "urgent_before": ("set", 1, False),
    "use_in": ("record_use",),
    "use_out": ("preserve",),
}


def _apply(state, program):
    match program:
        case ("preserve",):
            return state
        case ("set", index, value):
            return tuple(value if i == index else old for i, old in enumerate(state))
        case ("moving", "or", left, right):
            return state[left] or state[right], state[1], state[2]
        case ("clear_motion_and_path",):
            return False, False, state[2]
        case ("finish_motion",):
            return False, state[1], state[2] or all(state[:2])
        case ("record_use",):
            return state[:2] + (any(state),)
    raise ValueError("Unsupported rule program.")


def _state_text(world, state):
    return describe_state(world, "실험 대상", state, 0)


def reference_texts():
    for world, actions in ACTIONS.items():
        yield from (_state_text(world, state) for state in STATES)
        yield from (describe_action(action, 0) for action in actions)


class RuleControl:
    def __init__(self, vectors, approved):
        self.verified_programs = {key: effect for key, effect in RULE_EFFECTS.items() if FAMILY[key] in approved}
        self.worlds = tuple(ACTIONS)
        self.states = {
            world: torch.stack([vectors[_state_text(world, state)] for state in STATES])
            for world in self.worlds
        }
        self.actions = {
            world: torch.stack([vectors[describe_action(action, 0)] for action in ACTIONS[world]])
            for world in self.worlds
        }

    def predict(self, before, actions, options):
        state_prototypes = torch.cat([self.states[world] for world in self.worlds])
        best = int((state_prototypes @ before).argmax())
        world_index, state_index = divmod(best, len(STATES))
        world = self.worlds[world_index]
        state = STATES[state_index]
        choices = []
        for action, candidates in zip(actions, options, strict=True):
            index = int((self.actions[world] @ action).argmax())
            program = self.verified_programs.get(ACTIONS[world][index])
            if program is None:
                return [*choices, *([None] * (len(actions) - len(choices)))]
            state = _apply(state, program)
            choices.append(int((candidates @ self.states[world][STATES.index(state)]).argmax()))
        return choices
