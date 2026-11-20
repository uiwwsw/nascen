import copy
from collections import Counter
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from nascen.semantic_dataset import PRINCIPLES, RELATION_PAIRS, budget_scenes, digest, splits
from nascen.semantic_experiment import judge, run
from nascen.semantic_teacher import PrincipleLesson, collect, lesson_manifest, load_lessons
from nascen.sentence_encoder import SurfaceTokens
from nascen.teacher import request_json


def synthetic_collection():
    return lesson_manifest() | {
        "teacher": {"kind": "test_fixture", "model": "synthetic"},
        "relations": {pair.id: pair.reference if i else "contradiction" for i, pair in enumerate(RELATION_PAIRS)},
        "principles": {
            rule: asdict(PrincipleLesson("첫 조건이 충족된 상태이다.", "상태가 변할 수 있다.", "나머지 조건만 충족되었다.",
                                        "기존 상태가 유지된다.", reference))
            for rule, reference in PRINCIPLES.items()
        },
        "cost": {name: [{"wall_seconds": 0, "usage": {"prompt_tokens": 1, "completion_tokens": 2}}]
                 for name in (*PRINCIPLES, "relations-0", "relations-7")},
    }


class SemanticExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_budget_and_exam_use_whole_disjoint_groups(self):
        train, exam = splits()
        self.assertFalse({row.group for row in train} & {row.group for row in exam})
        self.assertFalse({text for row in train for text in (*row.context, *row.options)}
                         & {text for row in exam for text in (*row.context, *row.options)})
        for budget in (1, 2, 3, 4):
            self.assertEqual(Counter(row.principle for row in budget_scenes(train, budget)),
                             {rule: 2 * budget for rule in PRINCIPLES})
        self.assertTrue(all(row.principle is None for row in exam))
        self.assertTrue(all(len(row.options) == 4 for row in exam
                            if row.skill == "composition" and row.topic == "physical"))

    def test_teacher_only_gets_training_text_and_keeps_progress(self):
        fixture = synthetic_collection()
        calls = []

        def respond(prompt, material, schema, port, seed, *, max_tokens, ensure_ascii):
            self.assertFalse(ensure_ascii)
            calls.append(material)
            if isinstance(material, list):
                return {"pairs": [{"id": row["id"], "relation": fixture["relations"][row["id"]]}
                                  for row in material]}, {}
            key = next(key for key, reference in PRINCIPLES.items() if reference == material["reference_rule"])
            return fixture["principles"][key], {}

        with TemporaryDirectory() as directory, patch("nascen.semantic_teacher.request_json", side_effect=respond):
            path = Path(directory) / "lessons.json"
            result = collect(path, "local-model", 8766, 42)
            self.assertEqual(collect(path, "local-model", 8766, 42), result)
            self.assertEqual(len(calls), 6)
        lecture_material = repr(calls)
        self.assertTrue(all(sentence not in lecture_material for row in splits()[1] for sentence in row.context))
        _, audit = load_lessons(result)
        self.assertEqual(audit["errors"], 1)

    def test_unicode_material_reaches_the_teacher_as_readable_korean(self):
        with patch("nascen.teacher.build_opener") as build:
            build.return_value.open.return_value = BytesIO(
                b'{"choices":[{"message":{"content":"{}"}}]}'
            )
            request_json("분류하라", ["한글 문장"], {}, 8766, 42, max_tokens=16, ensure_ascii=False)
            request = build.return_value.open.call_args.args[0]
        content = json.loads(request.data)["messages"][1]["content"]
        self.assertIn("한글 문장", content)

    def test_runs_paired_arms_and_keeps_independent_checkpoint(self):
        teacher = synthetic_collection()
        review = {
            "lessons_sha256": digest(teacher),
            "principles": {rule: {"accepted": rule != "scope", "reason": "fixture"} for rule in PRINCIPLES},
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "student.pt"
            # Surface tokens serve only as a dependency-free test double for the E5 encoder.
            result = run(teacher, review, SurfaceTokens(32), budgets=(1,), seeds=(11, 22), epochs=2,
                         checkpoint=checkpoint)
            stored = torch.load(checkpoint, weights_only=True)
        self.assertEqual(stored["seed"], 11)
        self.assertEqual(set(stored), {"schema", "seed", "encoder", "weights"})
        self.assertEqual({trial["arm"] for trial in result["trials"]}, {"A", "B", "C", "D"})
        self.assertEqual(result["relation_audit"]["errors"], 1)
        rows = {row["arm"]: row for row in result["summary"]}
        self.assertEqual(rows["B"]["trained_parameters"], rows["C"]["trained_parameters"])
        self.assertGreater(rows["C"]["training_payload_bytes"], rows["B"]["training_payload_bytes"])
        self.assertEqual(result["teacher_cost"]["C"]["prompt_tokens"], 6)

        invalid = copy.deepcopy(review)
        invalid["lessons_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "match this teacher collection"):
            run(teacher, invalid, SurfaceTokens(), budgets=(1,), seeds=(11,), epochs=1)

    def test_abstentions_and_pair_failures_are_counted(self):
        _, exam = splits()
        scores, failures = judge(exam, [None, *(row.answer for row in exam[1:])])
        self.assertEqual(scores["paraphrase_both_correct"], 0.5)
        self.assertEqual(scores["exception_both_correct"], 1)
        self.assertEqual(failures[0]["id"], exam[0].id)
        self.assertIsNone(failures[0]["predicted"])


if __name__ == "__main__":
    unittest.main()
