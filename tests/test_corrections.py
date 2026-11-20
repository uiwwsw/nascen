import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import torch

from nascen.consent import PRINCIPLES, reference
from nascen.corrections import (
    HELD_CONDITIONS, collect, evaluate, exam_cases, local_feedback, practice_pool, training_data,
)
from nascen.teacher import Lesson

LESSONS = Path(__file__).resolve().parents[1] / "docs" / "experiment-002-lessons.json"


def fixture_teacher(question, answer, seed):
    decision, principle = reference(question.case)
    return Lesson(question.id, decision, principle, PRINCIPLES[principle][1])


class CorrectionTests(TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_practice_pool_excludes_exam_cases_and_the_unpracticed_condition(self):
        _, baseline = training_data(LESSONS)
        practice = {question.case for question in practice_pool()}
        exam = set(exam_cases())
        self.assertEqual((len(practice), len(exam)), (18, 24))
        self.assertTrue(practice.isdisjoint(exam))
        self.assertTrue(exam.isdisjoint(question.case for question, _ in baseline))
        self.assertNotIn(HELD_CONDITIONS, {(case.alternative, case.limited) for case in practice})
        self.assertEqual(sum((case.alternative, case.limited) == HELD_CONDITIONS for case in exam), 6)

    def test_teacher_payload_contains_only_practice_facts_and_student_answer(self):
        question = practice_pool()[0]
        answer = ("allow", "permission")
        with mock.patch("nascen.corrections.request_lessons", return_value=[fixture_teacher(question, answer, 42)]) as teacher:
            result = local_feedback(question, answer, 42, 8766)
        ids, _, rows, port, seed = teacher.call_args.args
        self.assertEqual((ids, port, seed), ([question.id], 8766, 42))
        self.assertEqual(set(rows[0]), {"id", "facts", "student"})
        self.assertEqual(rows[0]["student"], {"decision": "allow", "principle": "permission"})
        self.assertEqual(result.id, question.id)

    def test_training_rejects_bad_correction_and_matched_arm_replays_offline(self):
        with TemporaryDirectory() as directory:
            seen = []
            def imperfect_teacher(question, answer, seed):
                seen.append(question.id)
                if len(seen) == 1:
                    gold, principle = reference(question.case)
                    wrong = "allow" if gold != "allow" else "ask"
                    return Lesson(question.id, wrong, principle, "A deliberately incorrect correction.")
                return fixture_teacher(question, answer, seed)

            trace = collect(LESSONS, Path(directory) / "trace.json", "fixture", imperfect_teacher, 3, (42,))
        active = trace["turns"]["42"]["mistakes"]
        random = trace["turns"]["42"]["random"]
        self.assertEqual(len(active), len(random))
        self.assertTrue(all(turn["grader_found_error"] for turn in active))
        self.assertFalse(active[0]["accepted"])
        self.assertFalse(any(identifier.startswith("exam-") for identifier in seen))
        with mock.patch("socket.create_connection", side_effect=AssertionError("network contacted")):
            report = evaluate(LESSONS, trace)
        self.assertEqual(report["trials"][0]["accepted"], sum(turn["accepted"] for turn in active))
        self.assertEqual(report["trials"][0]["curve"][0], report["trials"][0]["curve"][1] | {"step": 0})
        self.assertFalse(report["teacher_available_to_exam"])
        active[0]["prior"]["decision"] = "tampered"
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            evaluate(LESSONS, trace)

    def test_interrupted_collection_resumes_without_reasking_saved_rounds(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            calls = []
            def interrupted_teacher(question, answer, seed):
                calls.append(question.id)
                if len(calls) == 2:
                    raise OSError("interrupted")
                return fixture_teacher(question, answer, seed)
            with self.assertRaises(OSError):
                collect(LESSONS, path, "fixture", interrupted_teacher, 2, (42,))
            partial = json.loads(path.read_text())
            self.assertEqual(len(partial["turns"]["42"]["mistakes"]), 1)

            with mock.patch(__name__ + ".fixture_teacher", wraps=fixture_teacher) as teacher:
                resumed = collect(LESSONS, path, "fixture", teacher, 2, (42,))
            self.assertEqual(resumed["turns"]["42"]["mistakes"][0], partial["turns"]["42"]["mistakes"][0])
            self.assertEqual(teacher.call_args_list[0].args[0].id, calls[1])
