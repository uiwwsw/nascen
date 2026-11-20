"""Experience-driven learning with a moving agent and local observations."""

from dataclasses import asdict, dataclass
from functools import cache
from math import sqrt
from random import Random

import torch

from .body import Action, EmbodiedWorld, LocalView, shift
from .experiment import MEMORY_SIZE, Experiment, probe, reward_value
from .model import ENCODERS, Learner, context_features
from .policy import QUESTION_STRATEGIES, question_scores
from .world import (
    COLORS, DIRECTIONS, EMBODIED_ACTIONS, HEIGHT, HELD_OUT, SHAPES, TRAIN_PAIRS, WALKS, WIDTH,
    Object, concept_target,
)

MODEL_VERSION = "0.2.0"
SCENE_STEPS = 32


@dataclass(frozen=True)
class EmbodiedConfig:
    encoder: str = "factored"
    question_strategy: str = "uncertainty"
    question_budget: int | None = None
    question_cost: float = 0.08

    def __post_init__(self) -> None:
        if self.encoder not in ENCODERS or self.question_strategy not in QUESTION_STRATEGIES:
            raise ValueError("Choose a valid visual encoder and question strategy.")
        if self.question_budget is not None and (type(self.question_budget) is not int or self.question_budget < 0):
            raise ValueError("Question budget must be a non-negative integer.")
        if not 0 <= self.question_cost <= 1:
            raise ValueError("Question cost must be between 0 and 1.")


@cache
def vision_probes(encoder: str) -> tuple:
    return probe(HELD_OUT, encoder), probe(TRAIN_PAIRS, encoder)


class EmbodiedExperiment(Experiment):
    world: EmbodiedWorld

    def __init__(
        self, seed: int, strategy: str, replay: bool = True, *,
        config: EmbodiedConfig, state: dict | None = None, records: list[dict] | None = None,
    ):
        self.config = config
        super().__init__(seed, strategy, replay, state=state, records=records, question_budget=config.question_budget)

    def _make_learner(self) -> Learner:
        return Learner(self.seed, self.config.encoder, EMBODIED_ACTIONS)

    def _make_world(self, scene: int, state: dict | None = None) -> EmbodiedWorld:
        if state is None:
            return EmbodiedWorld(self.seed, scene)
        objects = [Object(**obj) for obj in state["objects"]]
        return EmbodiedWorld(self.seed, scene, objects, tuple(state["position"]))

    def _probes(self) -> tuple:
        return vision_probes(self.config.encoder)

    def checkpoint(self) -> dict:
        return super().checkpoint() | {
            "model_version": MODEL_VERSION, "config": asdict(self.config), "position": self.world.agent,
        }

    def _select_action(
        self, view: LocalView, options: list[Action], visuals: dict[int, list[float]], rng: Random,
    ) -> Action:
        cost = self.config.question_cost
        if self.strategy == "random" or rng.random() < 0.12:
            selected = rng.choices(options, weights=[1 - cost if option.name == "ask" else 1 for option in options])
            return selected[0]

        asks = [option.target for option in options if option.name == "ask"]
        gains = [(record["visual"], record["learning_gain"]) for record in self.records if record["target"] is not None]
        scores = question_scores(
            self.learner, [visuals[target] for target in asks], self.config.question_strategy, gains, rng,
        )
        query_scores = dict(zip(asks, scores, strict=True))
        zero = [0.0] * self.learner.visual_size

        @torch.no_grad()
        def rank(option: Action) -> float:
            name, target = option.name, option.target
            features = visuals.get(target, zero)
            context = self._context(view, option)
            reward = self.learner.rewards(torch.tensor(features), torch.tensor(context))
            learned = float(reward[self.learner.actions.index(name)].clamp(-1, 1))
            if name in WALKS:
                destination = shift(view.position, WALKS[name])
                visited = self.visits.get(f"cell:{destination}", 0)
                exploration = 0.62 / sqrt(1 + visited)
            elif name == "ask":
                exploration = 0.44 + 0.45 * query_scores[target] - cost
            else:
                base = 0.52 if name in DIRECTIONS else 0.37
                exploration = base / sqrt(1 + self.visits.get(f"{target}:{name}", 0))
            return exploration + 0.16 * learned + rng.random() * 0.1

        return max(options, key=rank)

    @staticmethod
    def _context(view: LocalView, action: Action) -> list[float]:
        center = view.objects[action.target].position if action.name in DIRECTIONS else view.position
        return context_features(view.neighbors(center))

    def advance(self, teach_id: int | None = None, lesson: tuple[int, int] | None = None) -> dict:
        view = self.world.look()
        can_ask = self.question_budget is None or self.questions < self.question_budget
        if lesson is not None and teach_id is None:
            raise ValueError("Choose a visible object to teach.")
        if lesson is not None and (
            len(lesson) != 2 or any(type(index) is not int for index in lesson)
            or lesson[0] not in range(len(COLORS)) or lesson[1] not in range(len(SHAPES))
        ):
            raise ValueError("Choose a color and shape from the mentor vocabulary.")
        if teach_id is not None and (teach_id not in view.objects or not can_ask):
            raise ValueError("Teaching requires a visible object and an available mentor answer.")

        rng = Random(self.seed * 1_000_003 + self.steps * 97 + 43)
        visuals = {obj.id: self.learner.features(obj.pixels) for obj in view.objects.values()}
        proposed = self.world.proposals(view)
        safe = [option for option in proposed if self.world.can_execute(option)]
        rejected = len(proposed) - len(safe)
        options = [
            option for option in safe
            if option.name != "ask" or (can_ask and not self.visits.get(f"{option.target}:ask"))
        ]
        options = options or [Action("wait")]
        selected = Action("ask", teach_id) if teach_id is not None else self._select_action(view, options, visuals, rng)
        target = selected.target
        appearance = visuals[target] if target is not None else [0.0] * self.learner.visual_size
        context = self._context(view, selected)
        prediction = self._predict(appearance, context, selected.name) if target is not None else None
        outcome = self.world.perform(selected)
        answer = lesson
        if selected.name == "ask" and answer is None:
            answer = concept_target(self.world.objects[target])
        label = list(answer) if answer is not None else None
        obj = view.objects.get(target)
        record = {
            "step": self.steps + 1, "scene": self.world.scene, "action": selected.name,
            "object": {"id": obj.id, "x": obj.position[0], "y": obj.position[1]} if obj else None,
            "agent_before": list(view.position), "agent_after": list(self.world.agent),
            "visual": appearance, "context": context, "target": label, "prediction": prediction,
            "moved": outcome, "approval": None, "rejected": rejected,
            "error": abs(float(outcome) - prediction) if outcome is not None else None,
        }
        if label is not None:
            record["question"] = f"What color and shape is object {target}?"
            record["answer_source"] = "operator" if lesson is not None else "simulator"
            record["question_cost"] = self.config.question_cost
            self.questions += 1

        gain = self._update_models(record, rng)
        record["learning_gain"] = gain
        record["reward_parts"] = {
            "help": float(outcome is True), "approval": 0.0, "curiosity": gain,
            "cooperation": float(label is not None and not self.visits.get(f"{target}:ask")),
        }
        record["reward"] = reward_value(record["reward_parts"]) - record.get("question_cost", 0.0)
        self.learner.train_reward(appearance, context, selected.name, record["reward"])
        key = f"{target}:{selected.name}"
        self.visits[key] = self.visits.get(key, 0) + 1
        cell = f"cell:{self.world.agent}"
        self.visits[cell] = self.visits.get(cell, 0) + 1
        self.rejected += rejected
        self.records = [*self.records[-(MEMORY_SIZE - 1):], record]
        self.steps += 1
        if self.steps % SCENE_STEPS == 0:
            self.world = self._make_world(self.steps // SCENE_STEPS)
            self.visits.clear()
        record["metrics"] = self.metrics()
        return record

    def public_world(self) -> dict:
        view = self.world.look()
        return {
            "width": WIDTH, "height": HEIGHT, "scene": self.world.scene + 1, "agent": view.position,
            "tiles": [{"x": x, "y": y, "tile": tile} for (x, y), tile in view.tiles.items()
                      if 0 <= x < WIDTH and 0 <= y < HEIGHT],
            "objects": [
                {"id": obj.id, "x": obj.position[0], "y": obj.position[1],
                 "prediction": self._predict(self.learner.features(obj.pixels), [], "inspect")}
                for obj in view.objects.values()
            ],
        }
