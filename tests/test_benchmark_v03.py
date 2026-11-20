from unittest import TestCase

import torch

from nascen.benchmark_v03 import BATCH_SIZE, OBSERVATIONS, PHASE_OBSERVATIONS, run


class SelectiveBenchmarkTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        report = run([42], ("noisy",))
        cls.full, cls.random, cls.selective = report["scenarios"]["noisy"]["trials"]

    def test_selection_and_random_arms_receive_equal_labels_and_update_budgets(self):
        self.assertEqual(self.full["labels_received"], OBSERVATIONS)
        self.assertEqual(self.random["labels_received"], 250)
        self.assertEqual(self.random["labels_received"], self.selective["labels_received"])
        self.assertEqual(self.random["updates_by_window"], self.selective["updates_by_window"])
        for trial in (self.full, self.random, self.selective):
            end = trial["curve"][-1]
            self.assertEqual(end["observations"], OBSERVATIONS)
            self.assertEqual(end["optimizer_steps"] * BATCH_SIZE, end["examples_processed"])
            self.assertIn(PHASE_OBSERVATIONS, [point["observations"] for point in trial["curve"]])

    def test_noisy_selection_accepts_fewer_incorrect_labels_at_seed_42(self):
        self.assertLess(self.selective["corrupted_labels_trained"], self.random["corrupted_labels_trained"])
        self.assertGreater(self.selective["verification"].get("refuted", 0), 0)
