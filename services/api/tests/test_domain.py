import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain import Evidence, Report, Verdict


class ReportContractTests(unittest.TestCase):
    def test_inconclusive_requires_a_reason(self) -> None:
        report = Report(verdict=Verdict.INCONCLUSIVE, confidence=None)
        with self.assertRaisesRegex(ValueError, "require"):
            report.validate()

    def test_report_keeps_versioned_evidence(self) -> None:
        report = Report(
            verdict=Verdict.LIKELY_SYNTHETIC,
            confidence=0.82,
            evidence=(Evidence(source="fine_tuned_model", status="completed", detail="score available", version="0.1.0", score=0.82),),
            model_versions={"fine_tuned_model": "0.1.0"},
        )
        self.assertEqual(report.as_dict()["evidence"][0]["score"], 0.82)

    def test_out_of_range_score_is_rejected(self) -> None:
        report = Report(
            verdict=Verdict.LIKELY_AUTHENTIC,
            confidence=0.4,
            evidence=(Evidence(source="baseline", status="completed", detail="bad score", score=1.1),),
        )
        with self.assertRaisesRegex(ValueError, "scores"):
            report.validate()


if __name__ == "__main__":
    unittest.main()
