"""Keep observation budgets and language interventions comparable in Experiment 009."""

from collections import defaultdict
import unittest

import torch

from nascen.sentence_encoder import SentenceVectors, SurfaceTokens
from nascen.theory_language.grounded_data import HELD_ORDERS, held_out_exam, observed_priors, structure_coverage, training_windows
from nascen.theory_language.grounded_rollout import run, training_inputs
from nascen.theory_language.trainer import EncodedEpisodes, episode_texts
from nascen.theory_language.world import describe_action


class GroundedRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_observations_are_seen_in_both_training_arms(self):
        windows = training_windows(24)
        withheld = {
            tuple(describe_action(action, style) for action in order)
            for _, _, first, second in HELD_ORDERS
            for order in (first, second) for style in (0, 1)
        }
        for window in windows:
            self.assertNotIn(tuple(row.action for row in window), withheld)
            self.assertEqual(tuple(row.after for row in window[:-1]), tuple(row.before for row in window[1:]))
        rows = tuple(row for window in windows for row in window)
        data = EncodedEpisodes.build(rows, SentenceVectors(SurfaceTokens(32), episode_texts(rows)))
        selected = torch.tensor([6, 1, 6])
        single, single_before, single_actions = training_inputs(data, selected, False)
        unrolled, unrolled_before, unrolled_actions = training_inputs(data, selected, True)
        torch.testing.assert_close(single.flatten(), unrolled.flatten())
        torch.testing.assert_close(single_actions.flatten(0, 1), unrolled_actions.flatten(0, 1))
        torch.testing.assert_close(single_before[::3], unrolled_before)

    def test_crossed_wording_keeps_same_items_effects_and_choice_order(self):
        questions, keys = held_out_exam()
        variants = defaultdict(list)
        for question, key in zip(questions, keys, strict=True):
            variants[key.group].append((question, key))
        for group in variants.values():
            self.assertEqual(len(group), 8)
            versions = {(key.state_style, key.action_style, question.id[-1]): (question, key) for question, key in group}
            self.assertEqual(len(versions), 8)
            q, key = versions[0, 0, "0"]
            alternative, _ = versions[0, 2, "0"]
            self.assertEqual(q.before, alternative.before)
            self.assertEqual(q.options, alternative.options)
            self.assertNotEqual(q.actions, alternative.actions)
            alternate_state, _ = versions[2, 0, "0"]
            self.assertEqual(q.actions, alternate_state.actions)
            self.assertNotEqual(q.before, alternate_state.before)
            for state, action in ((0, 0), (0, 2), (2, 0), (2, 2)):
                left, k1 = versions[state, action, "0"]
                right, k2 = versions[state, action, "1"]
                self.assertEqual(left.before, right.before)
                self.assertEqual(left.options, right.options)
                self.assertEqual(k1.choices, key.choices)
                self.assertNotEqual(k1.answers[-1], k2.answers[-1])

    def test_observed_prior_is_a_separate_diagnostic(self):
        questions, keys = held_out_exam()
        question, key = next((q, k) for q, k in zip(questions, keys, strict=True) if k.skill == "order")
        diagnostic = observed_priors((question,), (key,))
        self.assertEqual(diagnostic[0].before, question.before)
        self.assertEqual(diagnostic[1].before, question.options[0][key.answers[0]])
        self.assertEqual(diagnostic[1].actions, (question.actions[1],))

    def test_published_budget_covers_each_exam_step_without_reusing_its_order(self):
        result = structure_coverage(training_windows(300), *held_out_exam())
        self.assertEqual(result["exam_structures"], 25)
        self.assertEqual(result["exam_structures_seen_in_training"], result["exam_structures"])

    def test_benchmark_runs_all_conditions(self):
        result = run(SurfaceTokens(32), windows=4, seeds=(11,), steps=1, batch_size=2)
        trials = {row["arm"]: row for row in result["trials"]}
        self.assertEqual(len(trials), 6)
        self.assertEqual(result["observations_per_update"], 6)
        self.assertEqual(trials["candidate-only"]["metrics"]["order_pair"], 0)
        counts = {row["parameters"] for row in trials.values() if "parameters" in row}
        self.assertEqual(len(counts), 1)
        self.assertIn("observed_previous_predictions_sha256", trials["rollout_grounded"])


if __name__ == "__main__":
    unittest.main()
