"""Embodied learning that checks evidence before consolidating an experience."""

from random import Random

from .body import Action, LocalView, shift
from .embodied import EmbodiedConfig, EmbodiedExperiment
from .experiment import REPLAY_SIZE
from .knowledge import MIN_LEARNING_SCORE, Knowledge, appearance_claim, label_outcome
from .policy import verification_bonus
from .world import DIRECTIONS

SELECTIVE_VERSION = "0.3.0"


class SelectiveExperiment(EmbodiedExperiment):
    version = SELECTIVE_VERSION

    def __init__(
        self, seed: int, strategy: str, replay: bool = True, *,
        config: EmbodiedConfig, state: dict | None = None, records: list[dict] | None = None,
    ):
        self.knowledge = Knowledge(state["knowledge"] if state is not None else None)
        super().__init__(seed, strategy, replay, config=config, state=state, records=records)

    def checkpoint(self) -> dict:
        return super().checkpoint() | {"knowledge": self.knowledge.checkpoint()}

    def _claim(self, view: LocalView, action: Action, source: str) -> tuple[str, str]:
        observed = view.objects[action.target]
        if action.name in DIRECTIONS:
            destination = shift(observed.position, action.name)
            claim = f"push-into:{view.tiles[destination]}"
            origin = f"push:{self.world.scene}:{observed.id}:{destination[0]}:{destination[1]}:{action.name}"
        else:
            claim = appearance_claim(observed.pixels)
            origin = f"{source}:{self.world.scene}:{observed.id}"
        return claim, origin

    def _action_biases(self, view: LocalView, options: list[Action]) -> dict[Action, float]:
        biases = {}
        for action in options:
            if action.name == "ask" or action.name in DIRECTIONS:
                claim, origin = self._claim(view, action, "simulator")
                biases[action] = verification_bonus(
                    self.knowledge.verify(claim), origin not in self.knowledge.evidence.get(claim, {}),
                )
        return biases

    def _process_experience(self, view: LocalView, action: Action, record: dict, rng: Random) -> tuple[float, bool]:
        observations = []
        for observed in view.objects.values():
            claim = appearance_claim(observed.pixels)
            self.knowledge.observe(claim)
            observations.append({"object": observed.id, "claim": claim, "pixels": list(observed.pixels)})
        record["observations"] = observations

        if record["target"] is None and record["moved"] is None:
            record["selection"] = {"status": "observation", "consolidated": False}
            return 0.0, False

        source = record.get("answer_source", "direct")
        claim, origin = self._claim(view, action, source)
        if record["target"] is not None:
            outcome = label_outcome(record["target"])
        else:
            outcome = "moves" if record["moved"] else "stays"
        record["evidence"] = {"claim": claim, "origin": origin, "source": source, "outcome": outcome}
        duplicate = self.knowledge.evidence.get(claim, {}).get(origin) == outcome
        self.knowledge.record(claim, origin, outcome)
        decision = self.knowledge.assess(claim, outcome, lambda: self._observed_probability(record))
        if duplicate:
            decision["reason"] = "duplicate"
        consolidated = not duplicate and decision.get("score", 0) >= MIN_LEARNING_SCORE
        record["selection"] = decision | {"consolidated": consolidated}
        if not consolidated:
            return 0.0, False

        gain = self._update_models(record, rng)
        self.knowledge.mark_trained(claim, outcome)
        return gain, True

    def _observed_probability(self, record: dict) -> float:
        if record["target"] is not None:
            return self._concept_probability(record)
        prediction = record["prediction"]
        return prediction if record["moved"] else 1 - prediction

    def _consolidated(self, record: dict) -> bool:
        if not record["selection"]["consolidated"]:
            return False
        evidence = record["evidence"]
        return self.knowledge.verify(evidence["claim"]).accepted == evidence["outcome"]

    def _replay(self, field: str, rng: Random) -> list[dict]:
        eligible = [r for r in self.records if r[field] is not None and self._consolidated(r)] if self.replay else []
        return rng.sample(eligible, min(REPLAY_SIZE, len(eligible)))

    def _question_gains(self) -> list[tuple[list[float], float]]:
        return [(r["visual"], r["learning_gain"]) for r in self.records if r["target"] is not None and self._consolidated(r)]
