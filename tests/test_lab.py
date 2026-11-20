from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import torch

from nascen.experiment import Experiment
from nascen.store import StaleRun, Store
from nascen.world import HELD_OUT, MENTOR, Object, World


def setUpModule():
    torch.set_num_threads(1)


class WorldTests(TestCase):
    def test_mentor_tile_is_protected_and_walls_block_movement(self):
        world = World(42, 0, [Object(1, "blue", "square", 5, 2, 7)])
        obj = world.objects[1]
        self.assertEqual(world.destination(obj, "east"), MENTOR)
        self.assertFalse(world.allowed(obj, "east"))
        with self.assertRaises(ValueError):
            world.execute(obj, "east")
        self.assertFalse(world.execute(obj, "north"))
        self.assertEqual(world.objects[1].position, (5, 2))
        self.assertTrue(world.execute(obj, "west"))
        self.assertEqual(world.objects[1].position, (4, 2))

    def test_held_out_combinations_never_appear_in_training_scenes(self):
        for seed in (11, 42, 55):
            for scene in range(10):
                world = World(seed, scene)
                pairs = {(obj.color, obj.shape) for obj in world.objects.values()}
                self.assertTrue(pairs.isdisjoint(HELD_OUT))


class LearningTests(TestCase):
    def test_mentor_experience_updates_concepts_and_feedback_only_updates_preference(self):
        experiment = Experiment(42, "active")
        initial = experiment.learner.checkpoint()
        event = experiment.advance(teach_id=1)
        asked = experiment.learner.checkpoint()
        self.assertEqual(event["action"], "ask")
        self.assertIsNotNone(event["target"])
        self.assertNotEqual(initial["color.weight"], asked["color.weight"])
        self.assertEqual(initial["physics.0.weight"], asked["physics.0.weight"])

        rated = experiment.feedback(-1)
        feedback = experiment.learner.checkpoint()
        self.assertNotEqual(asked["preference.weight"], feedback["preference.weight"])
        self.assertEqual(asked["shape.0.weight"], feedback["shape.0.weight"])
        self.assertEqual(rated["approval"], -1)
        self.assertEqual(experiment.steps, 1)
        self.assertGreater(experiment.rejected, 0)


class StorageTests(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.store = Store(self.path / "lab.sqlite3")

    def test_reopening_and_continuing_matches_uninterrupted_training(self):
        run_id = self.store.get()["run_id"]
        self.store.step(run_id, 13)
        reopened = Store(self.store.path)
        actual = reopened.step(run_id, 17)
        uninterrupted = Store(self.path / "uninterrupted.sqlite3")
        expected = uninterrupted.step(uninterrupted.get()["run_id"], 30)
        self.assertEqual(actual, expected)

    def test_feedback_and_reset_are_stale_safe(self):
        run_id = self.store.get()["run_id"]
        event = self.store.step(run_id, 1)
        self.assertEqual(self.store.feedback(run_id, event["steps"], 1)["latest"]["approval"], 1)
        with self.assertRaises(StaleRun):
            self.store.feedback(run_id, event["steps"], -1)
        reset = self.store.start(run_id, 123, "random", False)
        self.assertEqual(reset["steps"], 0)
        self.assertNotEqual(reset["run_id"], run_id)
        with self.assertRaises(StaleRun):
            self.store.step(run_id, 1)
        with sqlite3.connect(self.store.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM experiences").fetchone()[0], 1)

    def test_teaching_rejects_a_replaced_scene(self):
        run_id = self.store.get()["run_id"]
        self.store.step(run_id, 12)
        with self.assertRaises(StaleRun):
            self.store.step(run_id, 1, teach_id=1, scene=1)

    def test_failed_batch_keeps_checkpoint_and_history_consistent(self):
        before = self.store.get()
        real_advance = Experiment.advance

        def fail_on_second_step(experiment, teach_id):
            if experiment.steps == 1:
                raise RuntimeError("injected failure")
            return real_advance(experiment, teach_id)

        with mock.patch.object(Experiment, "advance", fail_on_second_step):
            with self.assertRaises(RuntimeError):
                self.store.step(before["run_id"], 2)
        self.assertEqual(self.store.get(), before)
        self.assertEqual(self.store.step(before["run_id"], 2)["steps"], 2)

    def test_concurrent_updates_are_serialized(self):
        run_id = self.store.get()["run_id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            requests = [pool.submit(self.store.step, run_id, 5) for _ in range(2)]
        self.assertEqual(sorted(request.result()["steps"] for request in requests), [5, 10])
        self.assertEqual(self.store.get()["steps"], 10)
