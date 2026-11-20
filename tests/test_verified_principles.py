"""Check update acceptance, durable verification, and transition scoring invariants."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from nascen.sentence_encoder import SurfaceTokens
from nascen.theory_language.dataset import write_json
from nascen.theory_language.principle_data import TRACK_DOMAIN, TRACK_HELD_STRUCTURE, training_rules
from nascen.theory_language.principle_model import PrincipleLearner
from nascen.theory_language.principle_study import load_bundle, save
from nascen.theory_language.principle_transition import apply_motion_probability
from nascen.theory_language.principle_update import Evidence, read_evidence, verified_study


class VerifiedPrincipleTests(unittest.TestCase):
    def setUp(self):
        self.original = torch.tensor([0.0])
        self.proposal = torch.tensor([1.0])
        self.counterexample = Evidence("new", torch.tensor([0.0]), True)
        self.previous_case = Evidence("old", torch.tensor([1.0]), False)

    def run_with_proposal(self, proposal_predictions, *, recorded=(), observations=(), verification=(), lesson=None):
        def fit(model, lesson, support, labels, *, previous=None):
            if support is not None:
                return self.proposal
            return lesson if lesson is not None else previous

        def infer(model, adapter, facts):
            decisions = (False, False) if torch.equal(adapter, self.original) else proposal_predictions
            return torch.tensor([float(decisions[int(value)]) for value in facts.flatten()])

        with patch("nascen.theory_language.principle_update.study", side_effect=fit) as fitting, \
             patch("nascen.theory_language.principle_update.predict", side_effect=infer):
            result = verified_study(
                object(), lesson, observations, previous=self.original, recorded=recorded, verification=verification,
            )
        return result, sum(call.args[2] is not None for call in fitting.call_args_list)

    def test_agreement_does_not_run_the_optimizer(self):
        result, fits = self.run_with_proposal((True, True), observations=(self.previous_case,))
        self.assertEqual(fits, 0)
        self.assertEqual(result.status, "unchanged")
        self.assertTrue(torch.equal(result.adapter, self.original))
        self.assertEqual(result.evidence, (self.previous_case,))

    def test_conflict_waits_if_verification_reuses_the_training_observation(self):
        result, fits = self.run_with_proposal((True, False), observations=(self.counterexample,),
                                              verification=(self.counterexample,))
        self.assertEqual(fits, 0)
        self.assertEqual(result.status, "verification_needed")
        self.assertTrue(torch.equal(result.adapter, self.original))

    def test_accepts_a_corrected_prediction_when_prior_evidence_is_preserved(self):
        result, fits = self.run_with_proposal((True, False), observations=(self.counterexample,),
                                              recorded=(self.previous_case,))
        self.assertEqual(fits, 1)
        self.assertEqual(result.status, "updated")
        self.assertEqual(result.evidence, (self.previous_case, self.counterexample))

    def test_rejects_a_regression_and_persists_the_old_evidence(self):
        result, fits = self.run_with_proposal((True, True), observations=(self.counterexample,),
                                              recorded=(self.previous_case,))
        self.assertEqual(fits, 1)
        self.assertEqual(result.status, "rejected")
        self.assertTrue(torch.equal(result.adapter, self.original))
        self.assertEqual(result.evidence, (self.previous_case,))

    def test_revised_lesson_cannot_overwrite_a_verified_outcome(self):
        result, fits = self.run_with_proposal((True, True), lesson=self.proposal, recorded=(self.previous_case,))
        self.assertEqual(fits, 0)
        self.assertEqual(result.status, "rejected")
        self.assertTrue(torch.equal(result.adapter, self.original))

    def test_rejects_conflicting_labels_even_if_phrase_order_and_spacing_change(self):
        facts = ("전원 연결.", "덮개 닫힘.", "코드 검증.", "정비 모드.")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "observations.json"
            write_json(path, [{"facts": facts, "lit": True},
                              {"facts": [" " + fact + " " for fact in reversed(facts)], "lit": False}])
            with self.assertRaisesRegex(ValueError, "same facts"):
                read_evidence(path, SurfaceTokens(16))

    def test_checkpoint_retains_validated_facts_and_domain(self):
        encoder = SurfaceTokens(16)
        model = PrincipleLearner(encoder.width, 11)
        case = Evidence("verified", torch.randn(4, encoder.width), False)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checked.pt"
            save(path, model, encoder, adapter=model.prior.detach(), metadata={},
                 domain=TRACK_DOMAIN, evidence=(case,))
            _, bundle = load_bundle(path, encoder)
            self.assertEqual(bundle["domain"], TRACK_DOMAIN)
            restored = Evidence(**bundle["verified_evidence"][0])
            self.assertEqual(restored.key, case.key)
            torch.testing.assert_close(restored.facts, case.facts)

    def test_bridge_changes_only_the_motion_distribution(self):
        original = torch.arange(1.0, 17.0).reshape(1, 1, -1)
        original /= original.sum()
        moving = torch.tensor([[0.12]])
        result = apply_motion_probability(original, moving)
        torch.testing.assert_close(result[..., 8:], original[..., 8:])
        torch.testing.assert_close(result[..., :8].reshape(1, 1, 2, 4).sum(-2),
                                   original[..., :8].reshape(1, 1, 2, 4).sum(-2))
        torch.testing.assert_close(result[..., 4:8].sum(-1) / result[..., :8].sum(-1), moving)
        torch.testing.assert_close(result.sum(-1), original.sum(-1))
        self.assertNotIn(TRACK_HELD_STRUCTURE, {rule.structure for rule in training_rules(TRACK_DOMAIN)})


if __name__ == "__main__":
    unittest.main()
