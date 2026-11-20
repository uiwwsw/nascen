import unittest
from unittest.mock import patch

import torch

from nascen.composition import grade, verified_cohort
from nascen.composition_split import CASE_BUDGET, DEPTHS, crosses_groups, examination, necessary, training_problems
from nascen.composition_student import TransitionStudent, decode_tables, execute, fit, predict
from nascen.composition_teacher import DATASET, program_table, verify, verified_table
from nascen.composition_world import SCALES, STATES, Problem, digest, library, trace


def program(rule):
    return "/".join("".join(STATES[state] for state in cycle) for cycle in rule.cycles)


def teacher_record(world, batches):
    offset = 0
    rows = []
    for values in batches:
        rows.append({"start": offset, "programs": values, "wall_seconds": 1.25,
                     "usage": {"prompt_tokens": 80, "completion_tokens": 40}})
        offset += len(values)
    return {"dataset": DATASET, "manifest_sha256": digest([rule.card() for rule in world]), "batches": rows}


class CompositionTests(unittest.TestCase):
    def test_library_prefixes_are_nested_and_conditions_are_functionally_distinct(self):
        full = library()
        self.assertEqual(len({rule.table() for rule in full}), 1000)
        for size in SCALES:
            self.assertEqual(library(size), full[:size])
        self.assertTrue(all(program_table(program(rule)) == rule.table() for rule in full))

    def test_verifier_accepts_equivalence_and_rejects_an_incorrect_teacher_program(self):
        world = library()
        first = program(world[0]).split("/")
        first[0] = first[0][1:] + first[0][:1]
        self.assertEqual(verified_table(world[0], "/".join(first)), world[0].table())
        self.assertIsNone(verified_table(world[0], "az"))
        self.assertIsNone(verified_table(world[0], program(world[1])))

        record = teacher_record(world, [[program(world[0]), "ae/ed"], [program(world[2]), program(world[3])]])
        tables, audit = verify(record, world, 4)
        self.assertEqual(tables, [rule.table() for rule in world[:4]])
        self.assertEqual(audit["simulator_fills"], 1)
        self.assertEqual(audit["rejected_ids"], [1])
        self.assertEqual(audit["teacher_wall_seconds"], 2.5)
        selected, _, cost = verified_cohort(record, world, audit, size=2)
        self.assertEqual(selected, [world[0], world[2]])
        self.assertEqual(cost["proposals_paid_for"], 4)
        self.assertEqual(cost["prompt_tokens"], 160)

    def test_training_and_exam_share_rules_but_never_a_cross_group_pair(self):
        world = library()
        for size in SCALES:
            train = training_problems(size, 11)
            self.assertEqual(len(set(train)), CASE_BUDGET)
            self.assertEqual({rule for problem in train for rule in problem.rules}, set(range(size)))
            self.assertTrue(all(not crosses_groups(problem.rules) for problem in train))
            exam = examination(world, size, count=8)
            self.assertEqual(len(exam), 8 * len(DEPTHS))
            for left, right in exam:
                self.assertEqual(left.state, right.state)
                self.assertNotEqual(trace(world, left)[-1], trace(world, right)[-1])
                for problem in (left, right):
                    self.assertTrue(crosses_groups(problem.rules))
                    self.assertTrue(necessary(world, problem, trace(world, problem)[-1]))

    def test_interpreter_composes_in_order_against_the_independent_simulator(self):
        world = library(10)
        problems = [problem for pair in examination(world, 10, count=4) for problem in pair]
        tables = decode_tables([program_table(program(rule)) for rule in world])
        result = grade(world, problems, lambda cases: execute(tables, cases), False)
        self.assertEqual(result["traces_correct"], len(problems))
        self.assertEqual(result["changing_pairs_correct"], len(problems) // 2)
        for group in result["by_depth"].values():
            self.assertEqual(group["reordered_pairs_correct"], 2)

    def test_learned_transition_prediction_uses_weights_without_a_teacher(self):
        torch.set_num_threads(1)
        world = library(10)
        model = TransitionStudent(10, 11)
        atomic = torch.tensor(decode_tables([rule.table() for rule in world]))
        fit(model, atomic)
        restored = TransitionStudent(10, 22)
        restored.load_state_dict(model.state_dict())
        questions = [Problem(0, (0, 1)), Problem(0, (1, 0)), Problem(7, (3, 2, 5, 0, 1))]
        expected = [trace(world, problem) for problem in questions]
        with patch("nascen.composition_student.execute", side_effect=AssertionError), patch(
            "nascen.composition_teacher.request_json", side_effect=AssertionError
        ):
            self.assertEqual(predict(restored, questions), expected)


if __name__ == "__main__":
    unittest.main()
