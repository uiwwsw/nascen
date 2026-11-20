from unittest import TestCase

import torch

from nascen.causal import CausalKernel
from nascen.evaluation import CONDITIONS, fixture_pool, run
from nascen.semantic import PushEvent, PushObservation


class CausalKernelTests(TestCase):
    def test_outcome_vocabulary_is_learned_from_observed_motion(self):
        observed = PushObservation((1, 2), "east", ("empty", "empty", "empty", "agent"))
        learner = CausalKernel([observed])
        self.assertEqual(learner.predict(observed), {(0, 0): 1.0})
        event = PushEvent(observed, (3, 3))
        self.assertEqual(event.offset, (2, 1))
        learner.observe(event)
        self.assertGreater(learner.predict(observed)[(2, 1)], learner.predict(observed)[(0, 0)])
        self.assertEqual(observed.position_after(event.offset), event.after)

    def test_counterexamples_remain_in_program_evidence(self):
        observed = PushObservation((1, 2), "east", ("empty", "empty", "wall", "agent"))
        learner = CausalKernel([observed])
        learner.observe(PushEvent(observed, (2, 2)))
        learner.observe(PushEvent(observed, (1, 2)))
        self.assertEqual(len(learner.events), 2)
        branch = learner.program.describe(learner.offsets)["branches"][0]
        self.assertEqual(branch["counterexamples"], 1)
        self.assertEqual(sum(effect["evidence"] for effect in branch["effects"]), 2)

    def test_experiment_equalizes_pushes_and_holds_out_positions_and_objects(self):
        torch.set_num_threads(1)
        training = fixture_pool(42, evaluation=False)
        held = fixture_pool(101, evaluation=True)
        self.assertTrue({f.objects[0].position for f in training}.isdisjoint({f.objects[0].position for f in held}))
        self.assertTrue(
            {(f.objects[0].color, f.objects[0].shape) for f in training}.isdisjoint(
                {(f.objects[0].color, f.objects[0].shape) for f in held}
            )
        )
        report = run([42], 12)
        self.assertEqual(tuple(trial["condition"] for trial in report["trials"]), CONDITIONS)
        active, random, neural = report["trials"]
        self.assertEqual(random["interventions"], neural["interventions"])
        self.assertTrue(all(len(trial["interventions"]) == 12 for trial in report["trials"]))
        self.assertGreaterEqual(active["curve"][-1]["position_accuracy"], 0.95)
        self.assertEqual(active["program"]["conditions"], [{"field": "destination", "equals": "empty"}])
        self.assertGreater(active["planner_predictions"], 0)
