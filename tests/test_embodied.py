from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from nascen.body import Action, EmbodiedWorld, distance
from nascen.embodied import EmbodiedConfig, EmbodiedExperiment
from nascen.store import InvalidRequest, Store
from nascen.world import Object


def setUpModule():
    torch.set_num_threads(1)


class EmbodiedWorldTests(TestCase):
    def test_view_updates_after_moving_and_contains_no_oracle_fields(self):
        obj = Object(1, "violet", "circle", 4, 0, 11)
        world = EmbodiedWorld(42, 0, [obj], position=(1, 0))
        self.assertEqual(world.look().objects, {})
        with self.assertRaises(ValueError):
            world.perform(Action("ask", 1))
        world.perform(Action("walk east"))
        self.assertEqual(world.agent, (2, 0))
        observed = world.look().objects[1]
        self.assertEqual(len(observed.pixels), 7 * 7 * 3)
        self.assertEqual(set(vars(observed)), {"id", "position", "pixels"})
        with self.assertRaises(ValueError):
            world.perform(Action("east", 1))
        world.perform(Action("walk east"))
        self.assertTrue(world.perform(Action("east", 1)))
        self.assertEqual(world.objects[1].position, (5, 0))

    def test_collision_and_protected_tile_block_both_paths(self):
        obj = Object(1, "blue", "circle", 5, 2, 1)
        world = EmbodiedWorld(42, 0, [obj], position=(4, 2))
        proposal = Action("east", 1)
        self.assertIn(proposal, world.proposals(world.look()))
        self.assertFalse(world.can_execute(proposal))
        with self.assertRaises(ValueError):
            world.perform(proposal)
        with self.assertRaises(ValueError):
            world.perform(Action("walk east"))
        self.assertEqual(world.agent, (4, 2))
        self.assertEqual(world.objects[1], obj)


class EmbodiedLearningTests(TestCase):
    def test_cost_can_make_the_policy_keep_exploring_instead_of_asking(self):
        free = EmbodiedExperiment(42, "active", config=EmbodiedConfig(question_cost=0))
        expensive = EmbodiedExperiment(42, "active", config=EmbodiedConfig(question_cost=1))
        for experiment in (free, expensive):
            for _ in range(50):
                experiment.advance()
        self.assertGreater(free.questions, 0)
        self.assertEqual(expensive.questions, 0)

    def test_only_solicited_labels_are_learned_and_questions_pay_configured_cost(self):
        config = EmbodiedConfig(encoder="shared", question_budget=1, question_cost=0.3)
        run = EmbodiedExperiment(42, "active", config=config)
        before = run.learner.checkpoint()
        event = run.advance(teach_id=2, lesson=(0, 1))
        self.assertEqual(event["target"], [0, 1])
        self.assertEqual(event["answer_source"], "operator")
        self.assertEqual(event["question_cost"], 0.3)
        self.assertEqual(set(event["object"]), {"id", "x", "y"})
        after = run.learner.checkpoint()
        self.assertNotEqual(before["visual_encoder.0.weight"], after["visual_encoder.0.weight"])
        reward = event["reward"]
        self.assertAlmostEqual(run.feedback(1)["reward"], reward + 0.25)

        events = [run.advance() for _ in range(30)]
        self.assertFalse(any(event["target"] is not None for event in events))
        self.assertTrue(any(event["agent_before"] != event["agent_after"] for event in events))
        self.assertTrue(all(distance(event["agent_before"], event["agent_after"]) <= 1 for event in events))
        self.assertEqual(run.questions, 1)

    def test_embodied_run_restores_position_weights_and_question_budget(self):
        config = EmbodiedConfig(question_strategy="progress", question_budget=3)
        with TemporaryDirectory() as directory:
            path = Path(directory)
            store = Store(path / "lab.sqlite3", default_config=config)
            initial = store.get()
            with self.assertRaises(InvalidRequest):
                store.step(initial["run_id"], 1, teach_id=3, scene=1)
            store.step(initial["run_id"], 17)
            resumed = Store(store.path).step(initial["run_id"], 33)
            direct = Store(path / "direct.sqlite3", default_config=config).step(initial["run_id"], 50)
            self.assertEqual(resumed, direct)
            self.assertEqual(resumed["config"]["question_budget"], 3)
            self.assertLessEqual(resumed["questions"], 3)

    def test_legacy_checkpoint_reopens_when_embodied_is_the_new_default(self):
        with TemporaryDirectory() as directory:
            store = Store(Path(directory) / "legacy.sqlite3")
            initial = store.get()
            previous = store.step(initial["run_id"], 5)
            reopened = Store(store.path, default_config=EmbodiedConfig())
            self.assertEqual(reopened.get(), previous)
            self.assertEqual(reopened.step(initial["run_id"], 1)["steps"], 6)
