"""Keep held mechanisms, study exposure, and checkpoint inference separate."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from nascen.sentence_encoder import SurfaceTokens
from nascen.theory_language.dataset import digest
from nascen.theory_language.principle_data import (
    STATES, held_rules, outcome, reference, split_states, study_manifest, training_rules,
)
from nascen.theory_language.principle_model import fit_curriculum, predict, study
from nascen.theory_language.principle_study import (
    benchmark, evidence_vectors, lesson_vector, load, prepare_curriculum, save, validate_facts,
)
from nascen.theory_language.principle_teacher import PROMPT


class PrincipleStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_holds_out_structures_and_keeps_query_combinations_separate(self):
        training = {rule.structure for rule in training_rules()}
        support, query = split_states()
        self.assertEqual(set(support) | set(query), set(STATES))
        self.assertFalse(set(support) & set(query))
        for rule in held_rules().values():
            self.assertNotIn(rule.structure, training)
            self.assertEqual({outcome(rule, state) for state in query}, {False, True})
            self.assertEqual(len({tuple(state[i] for i in rule.structure) for state in query}), 8)

    def test_learning_persists_as_weights_and_can_continue_without_rule_text(self):
        encoder = SurfaceTokens(48)
        curriculum, _ = prepare_curriculum(encoder)
        model = fit_curriculum(curriculum, encoder.width, seed=11, steps=3)
        lesson = lesson_vector(encoder, reference(next(iter(held_rules().values()))))
        facts = evidence_vectors(encoder, [("전원 접점이 연결되어 있다.", "보호 덮개가 닫혀 있다.",
                                           "확인 코드가 검증되었다.", "정비 모드가 켜져 있다.")])
        adapter = study(model, lesson, None, None)
        self.assertFalse(torch.equal(adapter, model.write(None)))
        target = (predict(model, adapter, facts) < 0.5).float()
        changed = study(model, None, facts, target, previous=adapter, steps=2)
        self.assertFalse(torch.equal(changed, adapter))
        self.assertTrue(torch.equal(study(model, None, None, None, previous=changed), changed))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "student.pt"
            save(checkpoint, model, encoder, adapter=changed, metadata={"observations_added": 1})
            bundle = torch.load(checkpoint, weights_only=True)
            self.assertEqual(set(bundle), {"schema", "encoder", "width", "weights", "adapter", "metadata"})
            restored, delta = load(checkpoint, encoder)
            torch.testing.assert_close(predict(restored, delta, facts), predict(model, changed, facts))
            torch.testing.assert_close(predict(restored, delta, facts[:, [2, 0, 3, 1]]),
                                       predict(model, changed, facts))

    def test_benchmark_uses_same_exposure_and_matched_irrelevant_lessons(self):
        rules = held_rules()
        packet = study_manifest() | {
            "prompt_sha256": digest(PROMPT), "teacher": {"model": "fixture", "seed": 42},
            "lessons": {key: asdict(reference(rule)) for key, rule in rules.items()},
            "cost": {key: {"wall_seconds": 0, "usage": {}} for key in rules},
        }
        review = {"collection_sha256": digest(packet),
                  "lessons": {key: {"accepted": True, "reason": "fixture"} for key in rules}}
        result = benchmark(SurfaceTokens(32), packet, review, seeds=(11,), steps=2)
        self.assertEqual(len(result["trials"]), len(rules) * 4 * 3)
        self.assertEqual(result["teacher_cost"]["requests"], len(rules))
        self.assertTrue(all(source != matched and rules[source].both == rules[matched].both
                            for source, matched in result["irrelevant_pairing"].items()))
        for row in result["trials"]:
            self.assertEqual(row["novel_relevant_total"], 8 - row["budget"])

    def test_requires_observation_sentences_instead_of_characters(self):
        with self.assertRaises(ValueError):
            validate_facts("가나다라")
        validate_facts(("전원이 있다.", "열렸다.", "승인됐다.", "정비 중이다."))


if __name__ == "__main__":
    unittest.main()
