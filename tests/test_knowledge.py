from unittest import TestCase

from nascen.knowledge import Knowledge, appearance_claim, learning_value
from nascen.world import Object, render_sprite


class KnowledgeTests(TestCase):
    def test_same_origin_cannot_verify_and_corrections_are_separate_votes(self):
        knowledge = Knowledge()
        self.assertEqual(knowledge.record("claim", "same-source", "a").status, "pending")
        self.assertEqual(knowledge.record("claim", "same-source", "a").votes, {"a": 1})
        self.assertEqual(knowledge.record("claim", "other-source", "b").status, "conflicted")
        accepted = knowledge.record("claim", "third-source", "a")
        self.assertEqual(accepted.accepted, "a")
        self.assertEqual(knowledge.check("claim", "b")[0], "refuted")
        self.assertEqual(knowledge.record("claim", "fourth-source", "b").status, "conflicted")
        revised = knowledge.record("claim", "same-source", "b")
        self.assertEqual(revised.accepted, "b")
        self.assertEqual(knowledge.check("claim", "a")[0], "refuted")

    def test_unlabeled_observation_remains_pending(self):
        knowledge = Knowledge()
        knowledge.observe("object")
        reopened = Knowledge(knowledge.checkpoint())
        self.assertEqual(reopened.findings()[0]["evidence"], {})
        self.assertEqual(reopened.counts()["pending"], 1)
        self.assertEqual(reopened.counts()["consolidations"], 0)

    def test_novelty_and_learning_value_are_independent(self):
        reference = learning_value(0.5, 0)
        familiar = learning_value(0.5, 4)
        confident = learning_value(0.9, 0)
        self.assertEqual(reference.value, familiar.value)
        self.assertLess(familiar.novelty, reference.novelty)
        self.assertEqual(reference.novelty, confident.novelty)
        self.assertLess(confident.value, reference.value)

    def test_claim_depends_on_pixels_rather_than_object_identifier(self):
        left = Object(1, "blue", "square", 0, 0, 42)
        right = Object(99, "blue", "square", 7, 5, 42)
        self.assertEqual(appearance_claim(render_sprite(left)), appearance_claim(render_sprite(right)))
