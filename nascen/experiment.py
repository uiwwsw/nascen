"""Run the observation → action → outcome → update loop."""

from dataclasses import asdict
from math import sqrt
from random import Random

import torch

from .model import Learner, context_features
from .world import (
    ACTIONS, AGENT, COLORS, DIRECTIONS, HEIGHT, HELD_OUT, MENTOR, SHAPES, TRAIN_PAIRS,
    WALLS, WIDTH, Object, World, concept_target, observe,
)

STRATEGIES = ("active", "random")
SCENE_LENGTH = 12
MEMORY_SIZE = 96
REPLAY_SIZE = 12
METRIC_WINDOW = 30
WEIGHTS = {"help": 0.35, "approval": 0.25, "curiosity": 0.25, "cooperation": 0.15}


def probe(pairs: tuple[tuple[str, str], ...]) -> tuple[torch.Tensor, torch.Tensor]:
    objects = [
        Object(0, color, shape, 0, 0, 2_000_000 + variant * 997)
        for color, shape in pairs
        for variant in range(8)
    ]
    return torch.tensor([observe(obj) for obj in objects]), torch.tensor([concept_target(obj) for obj in objects])


HELD_PROBE = probe(HELD_OUT)
SEEN_PROBE = probe(TRAIN_PAIRS)


class Experiment:
    def __init__(
        self, seed: int, strategy: str, replay: bool = True, *,
        state: dict | None = None, records: list[dict] | None = None, question_budget: int | None = None,
    ):
        self.seed = seed
        self.strategy = strategy
        self.replay = replay
        self.question_budget = question_budget
        self.learner = Learner(seed)
        self.records = list(records or [])
        if state is None:
            self.steps = 0
            self.questions = 0
            self.rejected = 0
            self.visits: dict[str, int] = {}
            self.world = World(seed, 0)
            self.initial = self.metrics()
        else:
            self.steps = state["steps"]
            self.questions = state["questions"]
            self.rejected = state["rejected"]
            self.visits = state["visits"]
            self.world = World(seed, state["scene"], [Object(**obj) for obj in state["objects"]])
            self.learner.restore(state["weights"])
            self.initial = state["initial"]

    def checkpoint(self) -> dict:
        return {
            "steps": self.steps, "questions": self.questions, "rejected": self.rejected,
            "visits": self.visits, "scene": self.world.scene,
            "objects": [asdict(obj) for obj in self.world.objects.values()],
            "weights": self.learner.checkpoint(), "initial": self.initial,
        }

    @torch.no_grad()
    def accuracy(self, dataset: tuple[torch.Tensor, torch.Tensor]) -> float:
        inputs, target = dataset
        color, shape = self.learner.concepts(inputs)
        correct = (color.argmax(dim=-1) == target[:, 0]) & (shape.argmax(dim=-1) == target[:, 1])
        return correct.float().mean().item()

    def metrics(self) -> dict:
        window = self.records[-METRIC_WINDOW:]
        push_errors = [record["error"] for record in window if record["error"] is not None]
        return {
            "step": self.steps,
            "accuracy": self.accuracy(HELD_PROBE),
            "seen_accuracy": self.accuracy(SEEN_PROBE),
            "prediction_error": sum(push_errors) / len(push_errors) if push_errors else None,
            "mean_reward": sum(record["reward"] for record in window) / len(window) if window else None,
        }

    def _select(self, candidates: list[tuple[Object, str]], rng: Random) -> tuple[Object, str]:
        if self.strategy == "random" or rng.random() < 0.18:
            return rng.choice(candidates)

        with torch.no_grad():
            scores = {}
            for obj in self.world.objects.values():
                visual = torch.tensor(observe(obj))
                context = torch.tensor(context_features(self.world.neighbors(obj)))
                concepts = self.learner.concepts(visual)
                uncertainty = 1 - sum(head.softmax(dim=-1).max().item() for head in concepts) / 2
                scores[obj.id] = (self.learner.rewards(visual, context).tolist(), uncertainty)

        def score(candidate: tuple[Object, str]) -> float:
            obj, action = candidate
            prior, uncertainty = scores[obj.id]
            count = self.visits.get(f"{obj.id}:{action}", 0)
            novelty = 1 / sqrt(1 + count)
            if action == "ask":
                novelty *= 1.25 + uncertainty if count == 0 else 0.20
            elif action == "inspect":
                novelty *= 0.30
            return prior[ACTIONS.index(action)] * 0.25 + novelty + rng.random() * 0.12

        return max(candidates, key=score)

    def advance(self, teach_id: int | None = None) -> dict:
        can_ask = self.question_budget is None or self.questions < self.question_budget
        if teach_id is not None and not can_ask:
            raise ValueError("This run exhausted its mentor question budget.")

        rng = Random(self.seed * 1_000_003 + self.steps * 97 + 43)
        proposed = [(obj, action) for obj in self.world.objects.values() for action in ACTIONS]
        allowed = [(obj, action) for obj, action in proposed if self.world.allowed(obj, action)]
        rejected = len(proposed) - len(allowed)
        self.rejected += rejected
        candidates = allowed if can_ask else [(obj, action) for obj, action in allowed if action != "ask"]

        obj, action = self._select(candidates, rng) if teach_id is None else (self.world.objects[teach_id], "ask")
        visit = f"{obj.id}:{action}"
        count = self.visits.get(visit, 0)
        self.visits[visit] = count + 1
        features = list(observe(obj))
        context = context_features(self.world.neighbors(obj))
        prediction = self._predict(features, context, action)
        moved = self.world.execute(obj, action)
        target = list(concept_target(obj)) if action == "ask" else None
        self.questions += int(target is not None)

        record = {
            "step": self.steps + 1, "scene": self.world.scene, "object": asdict(obj),
            "action": action, "visual": features, "context": context, "target": target,
            "prediction": prediction, "moved": moved, "approval": None,
            "error": abs(float(moved) - prediction) if moved is not None else None,
            "rejected": rejected,
        }
        progress = self._update_models(record, rng)
        record["reward_parts"] = {
            "help": float(moved is True), "approval": 0.0,
            "curiosity": progress, "cooperation": float(action == "ask" and count == 0),
        }
        record["reward"] = reward_value(record["reward_parts"])
        self.learner.train_reward(features, context, action, record["reward"])
        self.records = [*self.records[-(MEMORY_SIZE - 1):], record]
        self.steps += 1
        if self.steps % SCENE_LENGTH == 0:
            self.world = World(self.seed, self.steps // SCENE_LENGTH)
            self.visits.clear()

        record["metrics"] = self.metrics()
        return record

    @torch.no_grad()
    def _predict(self, visual: list, context: list, action: str) -> float | dict:
        if action in DIRECTIONS:
            direction = torch.tensor(tuple(DIRECTIONS).index(action))
            return self.learner.motion(torch.tensor(context), direction).sigmoid().item()
        color, shape = self.learner.concepts(torch.tensor(visual))
        return {"color": tuple(COLORS)[color.argmax().item()], "shape": SHAPES[shape.argmax().item()]}

    @torch.no_grad()
    def _concept_probability(self, record: dict) -> float:
        color, shape = self.learner.concepts(torch.tensor(record["visual"]))
        target_color, target_shape = record["target"]
        return (color.softmax(dim=-1)[target_color] * shape.softmax(dim=-1)[target_shape]).item()

    def _update_models(self, record: dict, rng: Random) -> float:
        if record["target"] is not None:
            recent = self._replay("target", rng)
            before = self._concept_probability(record)
            examples = [(item["visual"], item["target"]) for item in [record, *recent]]
            self.learner.train_concepts(examples)
            return max(0.0, self._concept_probability(record) - before)
        if record["moved"] is not None:
            recent = self._replay("moved", rng)
            moves = list(DIRECTIONS)
            transitions = [(item["context"], moves.index(item["action"]), item["moved"]) for item in [record, *recent]]
            self.learner.train_motion(transitions)
            after = self._predict(record["visual"], record["context"], record["action"])
            return max(0.0, record["error"] - abs(float(record["moved"]) - after))
        return 0.0

    def _replay(self, field: str, rng: Random) -> list[dict]:
        eligible = [record for record in self.records if record[field] is not None] if self.replay else []
        return rng.sample(eligible, min(REPLAY_SIZE, len(eligible)))

    def feedback(self, approval: int) -> dict:
        record = self.records[-1]
        record["approval"] = approval
        record["reward_parts"]["approval"] = float(approval)
        record["reward"] = reward_value(record["reward_parts"])
        self.learner.train_reward(record["visual"], record["context"], record["action"], record["reward"])
        record["metrics"] = self.metrics()
        return record

    def public_world(self) -> dict:
        return {
            "width": WIDTH, "height": HEIGHT, "scene": self.world.scene + 1,
            "agent": AGENT, "mentor": MENTOR, "walls": WALLS,
            "objects": [
                asdict(obj) | {"prediction": self._predict(list(observe(obj)), [], "inspect")}
                for obj in self.world.objects.values()
            ],
        }


def reward_value(components: dict[str, float]) -> float:
    return sum(WEIGHTS[key] * value for key, value in components.items())
