"""Keep the headline numbers in README.md tied to the published result files."""

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def load(name: str) -> dict:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


class PublishedNumbersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def assertPublished(self, text: str) -> None:
        self.assertIn(text, self.readme)

    def test_experiment_001_rule_discovery(self) -> None:
        rows = {row["condition"]: row for row in load("experiment-001.json")["summary"]}
        for condition in ("causal_active", "causal_random"):
            self.assertEqual(rows[condition]["mean_final_position_accuracy"], 1.0)
            self.assertEqual(rows[condition]["mean_discovery_step"], 3)
        self.assertPublished("100% 맞혔습니다")
        self.assertPublished("평균 3회")

    def test_experiment_004_corrections(self) -> None:
        rows = {row["strategy"]: row for row in load("experiment-004.json")["summary"]}
        mistakes, random = rows["mistakes"], rows["random"]
        self.assertEqual(mistakes["steps"], random["steps"])
        self.assertPublished(f"{pct(mistakes['baseline_decision'])}에서 {pct(mistakes['final_decision'])}")
        self.assertPublished(f"질문 횟수를 {random['steps']}회로")
        self.assertPublished(f"**{pct(random['final_decision'])}**")

    def test_experiment_005_principles(self) -> None:
        runs = load("experiment-005.json")["runs"]
        learned = [run for run in runs if run["arm"] == "learned_principle"]
        examples = [run for run in runs if run["arm"] == "examples_78"]
        self.assertTrue(learned and examples)
        for run in learned:
            facts = run["facts"]
            self.assertEqual(facts["answered"], 96)
            self.assertEqual(facts["correct_decisions"], 81)
            self.assertEqual(facts["urgency_pairs_correct"], 0)
        for run in examples:
            self.assertEqual(run["facts"]["correct_decisions"], 93)
        self.assertPublished("**81/96**")
        self.assertPublished("**0/12**")
        self.assertPublished("사례 78개로 학습하면 **93/96**")

    def test_experiment_006_composition(self) -> None:
        data = load("experiment-006.json")
        for scale in ("10", "100", "1000"):
            results = data["scales"][scale]["results"]
            for arm in ("learned_principles", "stored_principles"):
                self.assertTrue(all(r["decisions_correct"] == 512 for r in results[arm]), (scale, arm))
        coverage = data["scales"]["100"]["results"]["examples_coverage_first"]
        self.assertTrue(all(r["decisions_correct"] == 512 for r in coverage))
        verified = data["verified_only_100"]["results"]["learned_principles"]
        self.assertTrue(all(r["decisions_correct"] == 512 for r in verified))
        self.assertPublished("각각 **512/512**")


if __name__ == "__main__":
    unittest.main()
