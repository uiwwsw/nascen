from dataclasses import replace
from pathlib import Path
from unittest import TestCase, mock

import torch

from nascen.consent import Case, reference
from nascen.generalization import compile_inputs, grade_cases, run
from nascen.teacher import read_lessons

LESSONS = Path(__file__).resolve().parents[1] / "docs" / "experiment-002-lessons.json"


class GeneralizationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with mock.patch("socket.create_connection", side_effect=AssertionError("network contacted")):
            cls.result = run(LESSONS, (42,))

    def test_holdout_excludes_entire_policy_combinations_across_settings(self):
        _, training = read_lessons(LESSONS)
        cases_by_id = {question.id: question.case for question, _ in training}

        def key(case: Case) -> tuple:
            return case.owner, case.urgent, case.alternative, case.limited

        for fold in self.result["conditional"]:
            cases = [Case(**{name: row[name] for name in ("setting", "owner", "urgent", "alternative", "limited")})
                     for row in fold["cases"]]
            held = {key(case) for case in cases}
            used = {key(cases_by_id[identifier]) for identifier in fold["training_used"]}
            self.assertTrue(held.isdisjoint(used))
            self.assertEqual(len(cases), 12)
            self.assertEqual(len(held), 2)
            for left, right in zip(cases[::2], cases[1::2], strict=True):
                self.assertNotEqual(left.urgent, right.urgent)
                self.assertEqual(replace(left, urgent=right.urgent), right)
                self.assertNotEqual(reference(left)[0], reference(right)[0])

    def test_supported_wording_matches_structured_diagnostic(self):
        report = self.result["surface"]
        supported = report["arms"]["supported_text"]["trials"]
        structured = report["arms"]["structured_diagnostic"]["trials"]
        self.assertEqual(supported, structured)
        self.assertEqual(report["questions"], 50)
        self.assertEqual(supported[0]["answered"], 50)

    def test_unsupported_input_is_not_graded_as_an_answer(self):
        case = Case("workshop", "agreed", False, False, False)
        missing = compile_inputs(["This has no recognizable facts."])
        result = grade_cases({"seed": 42}, [case], missing)
        self.assertEqual(missing, [None])
        self.assertEqual(result["answered"], 0)
        self.assertEqual(result["decision_accuracy"], 0)
        self.assertEqual(result["predictions"], [{"decision": None, "principle": None}])
