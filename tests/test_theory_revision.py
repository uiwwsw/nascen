"""Regressions for candidate leakage and the revised comparison."""

import unittest

import torch

from nascen.sentence_encoder import SentenceVectors, SurfaceTokens
from nascen.theory_language.dataset import exam, training_pool
from nascen.theory_language.evaluation import encode_questions, exam_texts
from nascen.theory_language.revision import CONTRASTS, FRESH_ITEMS, audit_original, predict_revision, score
from nascen.theory_language.revision_model import RevisedLearner, fit_revision
from nascen.theory_language.trainer import EncodedEpisodes, episode_texts


class TheoryRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_previous_choice_geometry_reveals_the_key(self):
        result = audit_original(SurfaceTokens(64))
        self.assertEqual(result["exact_choice_geometry_correct"], result["steps"])

    def test_corrected_choices_are_independent_of_counterfactual_outcome(self):
        questions, answers = exam(full_options=True, items=FRESH_ITEMS)
        pairs = {}
        for question, key in zip(questions, answers, strict=True):
            if key.skill == "outside_knowledge":
                continue
            self.assertTrue(all(len(set(choices)) == 8 for choices in question.options))
            if key.skill in CONTRASTS:
                pairs.setdefault(key.group, []).append((question, key))
        for (left, first), (right, second) in pairs.values():
            self.assertEqual(left.before, right.before)
            self.assertEqual(set(left.options[0]), set(right.options[0]))
            self.assertNotEqual(left.options[0][first.steps[0]], right.options[0][second.steps[0]])

    def test_projected_model_ranks_both_exam_sizes_and_rolls_out(self):
        encoder = SurfaceTokens(48)
        episodes = training_pool(full_options=True)[:24]
        data = EncodedEpisodes.build(episodes, SentenceVectors(encoder, episode_texts(episodes)))
        model = fit_revision(data, encoder.width, "D", seed=11, steps=2, batch_size=8)
        questions, answers = exam(full_options=True, items=FRESH_ITEMS)
        exam_inputs = encode_questions(questions, SentenceVectors(encoder, exam_texts(questions)))
        choices, routes = predict_revision(model, exam_inputs)
        self.assertEqual(len(choices), len(questions))
        self.assertEqual(len(routes), sum(len(row.actions) for row in questions))
        self.assertEqual(routes.shape[-1], 8)
        result = score(questions, answers, choices)
        self.assertEqual(result["changed_steps"] + result["unchanged_steps"], len(routes) - 2)

    def test_change_gate_interpolates_between_previous_and_predicted_state(self):
        model = RevisedLearner(16, 11, "F")
        candidates = torch.eye(16)[:2].reshape(1, 1, 2, 16)
        previous, predicted = model.outcome_vectors(candidates).unbind(dim=2)
        self.assertEqual(int(model.outcome_scores(predicted, candidates, previous, torch.tensor([[-30.]])).argmax()), 0)
        self.assertEqual(int(model.outcome_scores(predicted, candidates, previous, torch.tensor([[30.]])).argmax()), 1)


if __name__ == "__main__":
    unittest.main()
