"""Regressions for the engine, teacher supervision and independent evaluation."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from nascen.sentence_encoder import SentenceVectors, SurfaceTokens
from nascen.theory_language.benchmark import run
from nascen.theory_language.dataset import digest, exam, load_partition, prepare, training_pool, write_json
from nascen.theory_language.model import TheoryLearner
from nascen.theory_language.teacher import (
    PROMPT, PROMPT_VERSION, Lesson, collect, load_reviewed, material,
)
from nascen.theory_language.trainer import (
    EncodedEpisodes, LossWeights, episode_texts, fit, lesson_texts, load_checkpoint, save_checkpoint,
    theory_targets, training_loss,
)
from nascen.theory_language.world import EXAM_ITEMS, REFERENCES, TRAIN_ITEMS, describe_action, describe_state


def teaching_fixture():
    keys = list(Lesson.__dataclass_fields__)
    collection = {
        "dataset": "theory-ko-v1", "teacher": {"seed": 42, "model": "fixture"},
        "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(PROMPT),
        "training_material_sha256": digest(material(42)),
        "lessons": {family: material(42)[family]["reference"] for family in REFERENCES},
        "cost": {key: {"wall_seconds": 0, "usage": {"prompt_tokens": 1, "completion_tokens": 1}} for key in REFERENCES},
    }
    assert all(set(row) == set(keys) for row in collection["lessons"].values())
    review = {
        "collection_sha256": digest(collection),
        "principles": {family: {"accepted": True, "reason": "fixture"} for family in REFERENCES},
    }
    return collection, review


class TheoryLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_partitions_hold_out_objects_structure_and_answer_keys(self):
        training = training_pool()
        self.assertGreaterEqual(len(training), 1000)
        self.assertEqual({row.lesson for row in training[:100]}, set(REFERENCES))
        questions, answers = exam()
        self.assertEqual(len(questions), len(answers))
        self.assertTrue(all(item not in row.before for items in EXAM_ITEMS.values() for item in items for row in training))
        in_domain = [row for row, key in zip(questions, answers, strict=True) if key.skill != "outside_knowledge"]
        self.assertTrue(all(item not in row.before for items in TRAIN_ITEMS.values() for item in items for row in in_domain))
        held_input = (
            describe_state("physical", TRAIN_ITEMS["physical"][0], (False, False, False), 0),
            describe_action("push", 0),
        )
        self.assertFalse(any((row.before, row.action) == held_input for row in training))
        self.assertFalse(hasattr(questions[0], "steps"))
        self.assertFalse(hasattr(questions[0], "lesson"))
        self.assertTrue(any(len(row.actions) == 3 for row in questions))

        collection, review = teaching_fixture()
        with TemporaryDirectory() as directory:
            path = Path(directory)
            prepare(path, size=24, lessons=load_reviewed(collection, review))
            rows = load_partition(path, "train")
            self.assertEqual(len(rows), 24)
            self.assertTrue(all(row.principle and row.conditions for row in rows))
            write_json(path / "answers.json", [])
            self.assertEqual(len(load_partition(path, "train")), 24)
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_partition(path, "answers")

    def test_teacher_only_receives_training_observations_and_is_resumable(self):
        calls = []

        def reply(prompt, payload, schema, port, seed, *, max_tokens, ensure_ascii):
            self.assertFalse(ensure_ascii)
            calls.append(payload)
            return payload["reference"], {}

        with TemporaryDirectory() as directory:
            with patch("nascen.theory_language.teacher.request_json", side_effect=reply):
                output = Path(directory) / "lessons.json"
                first = collect(output, "local-model")
                self.assertEqual(collect(output, "local-model"), first)
            self.assertEqual(len(calls), len(REFERENCES))
            self.assertTrue(all(not any(question.before in repr(call) for question in exam()[0]) for call in calls))
            self.assertTrue(all(len(call["observations"]) == 2 for call in calls))

    def test_teacher_text_changes_loss_and_has_gradient(self):
        collection, review = teaching_fixture()
        lessons = load_reviewed(collection, review)
        rows = training_pool()[:24]
        encoder = SurfaceTokens(48)
        data = EncodedEpisodes.build(rows, SentenceVectors(encoder, episode_texts(rows)))
        vectors = SentenceVectors(encoder, lesson_texts(lessons))
        aligned, _ = theory_targets(rows, lessons, vectors, "principle", 42)
        randomized, shuffled = theory_targets(rows, lessons, vectors, "shuffled", 42)
        noisy, noise = theory_targets(rows, lessons, vectors, "noisy", 42)
        self.assertGreater(shuffled["incorrect_assignments"], 0)
        self.assertEqual(noise["incorrect_assignments"], len(rows) // 4)
        self.assertFalse(torch.equal(noisy, aligned))
        model = TheoryLearner(encoder.width, 11)
        index = torch.arange(len(rows))
        loss, parts = training_loss(model, data, index, aligned, LossWeights())
        _, other = training_loss(model, data, index, randomized, LossWeights())
        self.assertNotEqual(float(parts["principle"].detach()), float(other["principle"].detach()))
        loss.backward()
        self.assertGreater(float(model.theory.weight.grad.norm()), 0)
        self.assertGreater(float(model.slot_values.grad.norm()), 0)

    def test_rollout_uses_its_own_state_and_checkpoint_roundtrips(self):
        encoder = SurfaceTokens(32)
        rows = training_pool()[:12]
        data = EncodedEpisodes.build(rows, SentenceVectors(encoder, episode_texts(rows)))
        model, _ = fit(data, encoder.width, 42, 2, 6, LossWeights())
        before = data.before[:1]
        actions = torch.stack((data.action[:1], data.action[1:2]), dim=1)
        with torch.no_grad():
            _, _, routes = model(before, actions)
            state, _, _ = model.step(model.state(before), actions[:, 0])
            _, _, second = model.step(state, actions[:, 1])
            torch.testing.assert_close(routes[:, 1], second)
            forced = model(before, actions, uniform=True)[0]
            baseline = model(before, actions)[0]
            self.assertFalse(torch.allclose(forced, baseline))
            with TemporaryDirectory() as directory:
                path = Path(directory) / "student.pt"
                save_checkpoint(path, model, encoder, {"seed": 42})
                restored, metadata = load_checkpoint(path, encoder)
                self.assertEqual(metadata, {"seed": 42})
                torch.testing.assert_close(restored(before, actions)[0], baseline)
                with self.assertRaisesRegex(ValueError, "original encoder"):
                    load_checkpoint(path, SurfaceTokens(48))

    def test_end_to_end_runs_all_required_arms_without_teacher_calls(self):
        collection, review = teaching_fixture()
        with TemporaryDirectory() as directory:
            path = Path(directory)
            prepare(path, size=24)
            with patch("nascen.theory_language.teacher.request_json", side_effect=AssertionError("unexpected teacher")):
                result = run(path, SurfaceTokens(48), collection, review, LossWeights(),
                             budgets=(24,), seeds=(11,), steps=2, batch_size=8, checkpoint_dir=path / "weights")
            self.assertTrue((path / "weights/principle-24-11.pt").exists())
        self.assertEqual({row["arm"] for row in result["summary"]}, {"A", "B", "C", "D", "E5-copy"})
        self.assertEqual(result["teacher_audit"]["accepted"], 6)
        self.assertTrue(all(len(row["predictions"]) == result["questions"] for row in result["trials"]))


if __name__ == "__main__":
    unittest.main()
