import unittest
from unittest.mock import patch

import torch

from nascen.consent import Case, reference
from nascen.distillation import Student, features, predict
from nascen.principle_experiment import (
    EXAM_SETTINGS, HELD_FACTS, TRAIN_SETTINGS, cases, compile_transfer, exam, facts,
    render_transfer, revision_material, training_pool,
)
from nascen.principles import (
    DATASET, FIELD_VALUES, allowed_values, case_region, constraints, execute, hidden_bounds, reference_packet, teaching_rules,
    train_principle,
)


def clause(decision, principle, **facts):
    return {
        "decision": decision, "principle": principle,
        "when": {field: facts.get(field, "any") for field in FIELD_VALUES},
    }


class PrincipleExperimentTests(unittest.TestCase):
    def test_disjoint_constraints_preserve_first_match_and_coverage_gaps(self):
        rules = teaching_rules(reference_packet()["teaching"])
        for active in (rules, rules[:-1]):
            regions = constraints(active)
            for case in cases(TRAIN_SETTINGS):
                matching = [
                    region.outcome for region in regions
                    if all(a & b for a, b in zip(region.region, case_region(case), strict=True))
                ]
                actual = execute(active, case)
                self.assertEqual(matching, [] if actual[0] is None else [actual])
                if active == rules:
                    self.assertEqual(actual, reference(case))

    def test_hidden_bounds_contain_every_partial_region_completion(self):
        regions = constraints(teaching_rules(reference_packet()["teaching"]))
        student = Student(len(Case("studio", "agreed", False, False, False).features()), 11)
        lower, upper = hidden_bounds(student, allowed_values(regions))
        all_cases = cases(TRAIN_SETTINGS + EXAM_SETTINGS)
        for i, rule in enumerate(regions):
            members = [
                case for case in all_cases
                if all(a & b for a, b in zip(rule.region, case_region(case), strict=True))
            ]
            hidden = student.hidden(features(members))
            self.assertTrue((hidden >= lower[i] - 1e-6).all())
            self.assertTrue((hidden <= upper[i] + 1e-6).all())

    def test_exam_uses_weights_without_access_to_teacher_or_interpreter(self):
        torch.set_num_threads(1)
        regions = constraints(teaching_rules(reference_packet()["teaching"]))
        trained = train_principle(regions, 11)
        self.assertEqual(trained["training_examples"], 0)
        self.assertEqual(trained["training_regions"], 7)
        self.assertNotIn("rules", trained)
        inputs = features(cases(EXAM_SETTINGS))
        untrained = Student(inputs.shape[1], 11)
        self.assertNotEqual(trained["weights"], {name: weight.tolist() for name, weight in untrained.state_dict().items()})
        outcomes = predict(trained, inputs)
        with patch("nascen.principle_experiment.execute", side_effect=AssertionError), patch(
            "nascen.principles.request_json", side_effect=AssertionError
        ):
            report = exam({"dataset": DATASET, "epoch_count": 180, "students": [trained]})
        self.assertEqual(report["runs"][0]["text_predictions"], ["/".join(outcome) for outcome in outcomes])

    def test_feedback_and_examples_exclude_new_places_and_withheld_facts(self):
        pool = training_pool()
        self.assertEqual(len(pool), 78)
        self.assertTrue(set(map(facts, pool)).isdisjoint(HELD_FACTS))
        self.assertTrue(set(TRAIN_SETTINGS).isdisjoint(EXAM_SETTINGS))
        wrong = reference_packet()["teaching"] | {"rules": [clause("allow", "permission")]}
        feedback = revision_material({"teaching": wrong})["counterexamples"]
        self.assertEqual(len(feedback), 5)
        for row in feedback:
            case = Case(**{key: value for key, value in row.items() if key not in ("decision", "principle")})
            self.assertIn(case, pool)
            self.assertEqual((row["decision"], row["principle"]), reference(case))

    def test_controlled_transfer_text_rejects_extra_and_conflicting_facts(self):
        for case in cases(EXAM_SETTINGS):
            self.assertEqual(compile_transfer(render_transfer(case)), case)
        question = render_transfer(Case("studio", "agreed", True, False, True))
        with self.assertRaises(ValueError):
            compile_transfer(question + " There is time to act without serious harm.")


if __name__ == "__main__":
    unittest.main()
