import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from nascen.embodied import EmbodiedConfig
from nascen.selective import SelectiveExperiment
from nascen.store import Store


def setUpModule():
    torch.set_num_threads(1)


class SelectiveLearningTests(TestCase):
    def test_matching_independent_evidence_gates_the_gradient_and_duplicates_do_not_train(self):
        run = SelectiveExperiment(42, "active", config=EmbodiedConfig())
        initial = run.learner.checkpoint()
        pending = run.advance(teach_id=2, lesson=(1, 2))
        self.assertEqual(pending["selection"]["status"], "pending")
        self.assertEqual(run.learner.checkpoint(), initial)
        duplicate = run.advance(teach_id=2, lesson=(1, 2))
        self.assertEqual(duplicate["selection"]["reason"], "duplicate")
        self.assertEqual(run.learner.checkpoint(), initial)

        verified = run.advance(teach_id=2)
        self.assertEqual(verified["selection"]["status"], "verified")
        self.assertTrue(verified["selection"]["consolidated"])
        trained = run.learner.checkpoint()
        self.assertNotEqual(trained["shape.0.weight"], initial["shape.0.weight"])
        self.assertEqual(trained["physics.0.weight"], initial["physics.0.weight"])

        rejected = run.advance(teach_id=2)
        self.assertFalse(rejected["selection"]["consolidated"])
        self.assertEqual(run.learner.checkpoint(), trained)

    def test_contradiction_blocks_replay_until_supported_again(self):
        run = SelectiveExperiment(42, "active", config=EmbodiedConfig())
        run.advance(teach_id=2, lesson=(1, 2))
        verified = run.advance(teach_id=2)
        weights = run.learner.checkpoint()
        conflict = run.advance(teach_id=2, lesson=(0, 1))
        self.assertEqual(conflict["selection"]["status"], "conflicted")
        self.assertEqual(run.learner.checkpoint(), weights)
        self.assertFalse(run._consolidated(verified))
        self.assertEqual(run._question_gains(), [])

    def test_autonomous_pushes_collect_evidence_without_mentor_answers(self):
        run = SelectiveExperiment(42, "active", config=EmbodiedConfig(question_budget=0))
        initial = run.learner.checkpoint()
        events = [run.advance() for _ in range(100)]
        direct = [event for event in events if event.get("evidence", {}).get("source") == "direct"]
        self.assertGreater(len(direct), 1)
        self.assertTrue(any(event["selection"]["status"] == "pending" for event in direct))
        self.assertTrue(any(event["selection"]["consolidated"] for event in direct))
        self.assertEqual(run.questions, 0)
        self.assertNotEqual(run.learner.checkpoint()["physics.0.weight"], initial["physics.0.weight"])
        self.assertEqual(run.learner.checkpoint()["shape.0.weight"], initial["shape.0.weight"])


class SelectiveStorageTests(TestCase):
    def test_checkpoint_and_counterexample_history_survive_restart(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lab.sqlite3"
            store = Store(path, selective=True)
            self.assertEqual(store.get()["model_version"], "0.3.0")
            first = store.step(1, 1, teach_id=2, scene=1, lesson=(0, 1))
            self.assertEqual(first["knowledge"]["consolidations"], 0)
            conflict = Store(path).step(1, 1, teach_id=2, scene=1)
            self.assertEqual(conflict["latest"]["selection"]["status"], "conflicted")
            verified = Store(path).step(1, 1, teach_id=2, scene=1, lesson=(1, 2))
            self.assertEqual(verified["knowledge"]["consolidations"], 1)
            knowledge = Store(path).get(include_knowledge=True)
            key = verified["latest"]["evidence"]["claim"]
            finding = next(item for item in knowledge["findings"] if item["claim"] == key)
            self.assertEqual(finding["accepted"], "1:2")
            with sqlite3.connect(path) as db:
                earlier = json.loads(db.execute("select payload from experiences where step = 1").fetchone()[0])
            self.assertEqual(earlier["evidence"]["outcome"], "0:1")
            self.assertTrue(earlier["observations"][0]["pixels"])

    def test_v02_database_keeps_immediate_learning(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lab.sqlite3"
            Store(path, default_config=EmbodiedConfig())
            previous = Store(path, selective=True)
            self.assertEqual(previous.get()["model_version"], "0.2.0")
            taught = previous.step(1, 1, teach_id=2, scene=1)
            self.assertNotIn("selection", taught["latest"])
            self.assertGreater(taught["latest"]["learning_gain"], 0)
