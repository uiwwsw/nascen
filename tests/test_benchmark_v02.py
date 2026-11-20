from unittest import TestCase

import torch

from nascen.benchmark_v02 import BATCH_SIZE, QUESTION_BUDGET, QUESTION_WINDOW, observe_stream, run
from nascen.world import HELD_OUT, TRAIN_PAIRS, COLORS, SHAPES


class ControlledBenchmarkTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.experiments = run([42])["experiments"]

    def test_encoders_learn_under_matched_inputs_and_gradient_budget(self):
        factored, shared = self.experiments["A"]["trials"]
        initial, final = factored["curve"][0], factored["curve"][-1]
        self.assertGreater(final["held_out_accuracy"], initial["held_out_accuracy"] + 0.25)
        self.assertGreater(shared["curve"][-1]["train_split_accuracy"], shared["curve"][0]["train_split_accuracy"] + 0.25)
        for baseline, raw in zip(factored["curve"], shared["curve"], strict=True):
            for budget in ("questions", "observations", "optimizer_steps", "examples_processed"):
                self.assertEqual(baseline[budget], raw[budget])

    def test_question_strategies_each_use_one_answer_per_window(self):
        trials = self.experiments["B"]["trials"]
        self.assertEqual(len(trials), 6)
        for trial in trials:
            final = trial["curve"][-1]
            self.assertEqual(final["questions"], QUESTION_BUDGET)
            self.assertEqual(final["optimizer_steps"], QUESTION_BUDGET)
            self.assertEqual(final["observations"], QUESTION_BUDGET * QUESTION_WINDOW)
            self.assertEqual(final["examples_processed"], QUESTION_BUDGET * BATCH_SIZE)
            self.assertEqual(
                [index // QUESTION_WINDOW for index in trial["selected_observations"]],
                list(range(QUESTION_BUDGET)),
            )

    def test_replay_reduces_forgetting_after_an_identical_first_phase(self):
        newest, replay = self.experiments["C"]["trials"]
        self.assertEqual(newest["before_switch"], replay["before_switch"])
        self.assertEqual(newest["curve"][-1]["examples_processed"], replay["curve"][-1]["examples_processed"])
        self.assertGreater(replay["old_accuracy_change"], newest["old_accuracy_change"])

    def test_held_out_combinations_never_enter_the_training_stream(self):
        examples = observe_stream(42, TRAIN_PAIRS, 48)
        observed_labels = {(tuple(COLORS)[example.answer[0]], SHAPES[example.answer[1]]) for example in examples}
        self.assertEqual(observed_labels, set(TRAIN_PAIRS))
        self.assertTrue(observed_labels.isdisjoint(HELD_OUT))
