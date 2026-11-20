"""Check the trainable semantic state and its normal inference bundle."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from nascen.sentence_encoder import SurfaceTokens
from nascen.theory_language.observation_data import STATE_CODES, training_views
from nascen.theory_language.observation_encoding import encode_observations
from nascen.theory_language.observation_exam import independent_exam
from nascen.theory_language.observation_model import ObservationLearner
from nascen.theory_language.observation_training import encode_training, fit_observations
from nascen.theory_language.trainer import load_checkpoint, save_checkpoint
from nascen.theory_language.world import ACTIONS


class ObservationLearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_each_state_code_uses_three_correctly_labeled_facts(self):
        model = ObservationLearner(24, 16, 11, 4)
        for code, (world, state) in enumerate(STATE_CODES):
            world_index = tuple(ACTIONS).index(world)
            self.assertEqual(model.facts_per_state[code].tolist(),
                             [6 * world_index + 2 * i + int(value) for i, value in enumerate(state)])

    def test_held_examples_change_one_input_group_and_not_the_answer(self):
        questions, keys = independent_exam()
        variants = {q.id: (q, k) for q, k in zip(questions, keys, strict=True)}
        for question in questions:
            phase, identifier = question.id.split("/")
            if phase != "original":
                continue
            base, key = variants[question.id]
            for part in ("initial", "action", "choices"):
                variant, answer = variants[f"{part}/{identifier}"]
                self.assertEqual(answer.answers, key.answers)
                self.assertEqual(base.before == variant.before, part != "initial")
                self.assertEqual(base.actions == variant.actions, part != "action")
                self.assertEqual(base.options == variant.options, part != "choices")
        states, actions, _ = training_views()
        training = {text for group in (*states.values(), *actions.values()) for text in group}
        for question in questions:
            if question.id.startswith("all/"):
                self.assertNotIn(question.before, training)
                self.assertTrue(all(action not in training for action in question.actions))

    def test_training_updates_fact_reader_and_transition_and_roundtrips(self):
        encoder = SurfaceTokens(48)
        training = encode_training(encoder, windows=60)
        initial = ObservationLearner(encoder.width, 16, 11, seed=11)
        learned, stats = fit_observations(training, encoder.width, seed=11, steps=3, batch_size=12)
        self.assertGreater(stats["parameters"], 0)
        self.assertFalse(torch.equal(learned.fact_encoder[0].weight, initial.fact_encoder[0].weight))
        self.assertFalse(torch.equal(learned.transition[0].weight, initial.transition[0].weight))
        sample = training.states.features[:8]
        actions = training.actions.features[:2].unsqueeze(0)
        options = sample.reshape(1, 2, 4, *sample.shape[1:])
        before = sample[:1]
        with torch.no_grad():
            scores = learned.score_steps(before, actions, options)
            reversed_first = options.clone()
            reversed_first[:, 0] = reversed_first[:, 0].flip(1)
            changed = learned.score_steps(before, actions, reversed_first)
            torch.testing.assert_close(scores[:, 0].flip(-1), changed[:, 0])
            torch.testing.assert_close(scores[:, 1], changed[:, 1])
            with TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "student.pt"
                save_checkpoint(checkpoint, learned, encoder, {"mode": "observation"})
                reloaded, meta = load_checkpoint(checkpoint, encoder)
                self.assertIsInstance(reloaded, ObservationLearner)
                self.assertEqual(meta, {"mode": "observation"})
                torch.testing.assert_close(scores, reloaded.score_steps(before, actions, options))

    def test_encode_observations_accepts_paragraphs_and_single_sentences(self):
        texts = ("하나. 둘! 셋?", "한 문장")
        vectors, cost = encode_observations(SurfaceTokens(32), texts)
        self.assertEqual(vectors[texts[0]].shape, vectors[texts[1]].shape)
        self.assertEqual(cost["unique_sentences"], 5)
        self.assertEqual(int(vectors[texts[0]][1:].ne(0).any(-1).sum()), 3)
        self.assertEqual(int(vectors[texts[1]][1:].ne(0).any(-1).sum()), 1)


if __name__ == "__main__":
    unittest.main()
