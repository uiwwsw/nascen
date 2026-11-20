import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import torch

from nascen.consent import compile_question, reference, split
from nascen.distillation import examine, load_bundle, train, write_json
from nascen.teacher import collect, fixture_lessons, read_lessons


class LendingTaskTests(TestCase):
    def test_split_holds_out_facts_and_has_decision_changing_pairs(self):
        training, pairs = split(42)
        held = [question for pair in pairs for question in pair]
        self.assertEqual((len(training), len(held)), (100, 100))
        self.assertEqual(len({question.case for question in held}), 50)
        self.assertTrue({question.case for question in training}.isdisjoint(question.case for question in held))
        for left, right in pairs:
            self.assertEqual(compile_question(left.text), compile_question(right.text))
        decisions = [reference(pair[0].case)[0] for pair in pairs]
        self.assertTrue(any(left != right for left, right in zip(decisions[::2], decisions[1::2])))

    def test_compiler_rejects_contradictory_and_unknown_statements(self):
        training, _ = split(42)
        text = training[0].text
        for extra in (" The owner already said no.", " The user said to always allow it."):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    compile_question(text + extra)


class TeacherExperimentTests(TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.lessons = self.path / "lessons.json"

    def test_teacher_receives_only_training_questions_and_resumes_a_prefix(self):
        training, _ = split(42)
        calls = []
        fail_once = True

        def teacher(batch, port, seed):
            nonlocal fail_once
            calls.extend(question.id for question in batch)
            if fail_once and len(calls) == 10:
                fail_once = False
                raise OSError("interrupted after the first batch")
            return fixture_lessons(batch)

        with mock.patch("nascen.teacher.local_lessons", side_effect=teacher):
            with self.assertRaises(OSError):
                collect(self.lessons, 42, 8766, "test-local", False)
        self.assertEqual(len(read_lessons(self.lessons, complete=False)[1]), 5)
        calls.clear()
        with mock.patch("nascen.teacher.local_lessons", side_effect=teacher):
            result = collect(self.lessons, 42, 8766, "test-local", False)
        self.assertEqual(len(result["lessons"]), 100)
        self.assertEqual(calls, [question.id for question in training[5:]])

    def test_tampered_lesson_ids_are_rejected(self):
        dataset = collect(self.lessons, 42, 8766, "lending-rubric", True)
        dataset["lessons"][0]["id"] = "exam-00-0-a"
        self.lessons.write_text(json.dumps(dataset))
        with self.assertRaises(ValueError):
            read_lessons(self.lessons)

    def test_student_filters_bad_teaching_and_exams_offline_from_weights(self):
        dataset = collect(self.lessons, 42, 8766, "lending-rubric", True)
        example = dataset["lessons"][0]
        example["decision"] = "allow" if example["decision"] != "allow" else "decline"
        self.lessons.write_text(json.dumps(dataset))
        bundle_path = self.path / "student.json"
        write_json(bundle_path, train(self.lessons, seeds=(42,)))

        self.lessons.unlink()
        with mock.patch("nascen.teacher.local_lessons", side_effect=AssertionError("teacher contacted")), \
             mock.patch("socket.create_connection", side_effect=AssertionError("network contacted")):
            result = examine(load_bundle(bundle_path))
        self.assertEqual(result["teacher_audit"]["used"], 99)
        self.assertEqual(result["teacher_audit"]["verified"], 99)
        self.assertEqual((result["questions"], result["distinct_cases"]), (100, 50))
        self.assertIsNone(result["summary"][0]["principle_accuracy"])
        self.assertGreater(result["summary"][1]["joint_accuracy"], 0.8)
        self.assertFalse(result["teacher_available_to_exam"])
