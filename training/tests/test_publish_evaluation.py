from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import BENCHMARK_SCHEMA_VERSION, RUNNER_VERSION, standard_conditions  # noqa: E402
from publish_evaluation import (  # noqa: E402
    EvaluationPublicationError,
    build_publication,
    load_benchmark_report,
    load_error_analysis,
    write_publication,
)


REVISION = "a" * 40


def digest(character: str) -> str:
    return "sha256:" + character * 64


def file_digest(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def benchmark_document(
    *,
    detector_id: str,
    detector_version: str,
    detector_family: str,
    auroc_offset: float,
    heldout_digest: str = digest("b"),
) -> dict[str, object]:
    conditions = []
    for index, condition in enumerate(standard_conditions()):
        conditions.append(
            {
                "id": condition.identifier,
                "kind": condition.kind,
                "parameters": dict(condition.parameters),
                "sample_count": 10,
                "labels": {"camera_origin": 5, "fully_synthetic": 5},
                "auroc": 0.90 - index * 0.01 + auroc_offset,
                "mean_synthetic_probability": {
                    "camera_origin": 0.15 + index * 0.01,
                    "fully_synthetic": 0.85 - index * 0.01,
                },
                "latency_ms": {
                    "scorer_wall": {"count": 10, "mean": 7.5, "p50": 7.0, "p95": 9.0},
                    "detector_reported": {"count": 10, "mean": 4.5, "p50": 4.0, "p95": 6.0},
                },
            }
        )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "purpose": "held_out_robustness_evaluation_not_calibration_or_deployment",
        "runner": {"name": "veritas-face-benchmark", "version": RUNNER_VERSION},
        "source_revision": REVISION,
        "detector": {
            "id": detector_id,
            "version": detector_version,
            "family": detector_family,
        },
        "scorer_identity": f"{detector_version}-private-adapter",
        "inputs": {
            "record_manifest_sha256": digest("a"),
            "heldout_test_records_sha256": heldout_digest,
            "source_manifest_sha256": digest("c"),
            "split_manifest_sha256": digest("d"),
        },
        "conditions": conditions,
        "limitations": ["Private aggregate test result for the fixed v1 matrix."],
    }


def candidate_calibration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "calibration_version": "heldout-portraits-2026.09",
        "validation_manifest_sha256": digest("e"),
        "detectors": [
            {
                "id": "portrait-candidate",
                "version": "release-2026.09.1",
                "family": "mobilenetv3-small",
                "points": [
                    {"raw_probability": 0, "calibrated_probability": 0.02},
                    {"raw_probability": 0.5, "calibrated_probability": 0.46},
                    {"raw_probability": 1, "calibrated_probability": 0.98},
                ],
            }
        ],
    }


class EvaluationPublicationTests(unittest.TestCase):
    def _write_inputs(self, root: Path, *, heldout_digest: str = digest("b")) -> tuple[Path, Path, Path]:
        candidate = root / "candidate.json"
        baseline = root / "baseline.json"
        calibration = root / "calibration.json"
        candidate.write_text(
            json.dumps(
                benchmark_document(
                    detector_id="portrait-candidate",
                    detector_version="release-2026.09.1",
                    detector_family="mobilenetv3-small",
                    auroc_offset=0.02,
                    heldout_digest=heldout_digest,
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        baseline.write_text(
            json.dumps(
                benchmark_document(
                    detector_id="portrait-baseline",
                    detector_version="baseline-2026.09.1",
                    detector_family="spectral-baseline",
                    auroc_offset=-0.04,
                    heldout_digest=heldout_digest,
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        calibration.write_text(json.dumps(candidate_calibration(), indent=2, sort_keys=True), encoding="utf-8")
        return candidate, baseline, calibration

    def _write_error_analysis(self, root: Path, candidate: Path, baseline: Path) -> Path:
        analysis = root / "error-analysis.json"
        analysis.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "candidate_benchmark_sha256": file_digest(candidate),
                    "baseline_benchmark_sha256": file_digest(baseline),
                    "review_scope": "private_record_level_review_without_identifiers",
                    "findings": [
                        {
                            "condition_id": "gaussian_blur_2",
                            "category": "robustness_degradation",
                            "affected_count": 2,
                            "summary": "Blurred portraits warrant manual review before relying on detector evidence.",
                        }
                    ],
                    "additional_limitations": [
                        "This small reviewed evaluation does not establish demographic fairness.",
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return analysis

    def test_creates_a_checksum_bound_public_aggregate_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate_path, baseline_path, calibration_path = self._write_inputs(root)
            error_analysis_path = self._write_error_analysis(root, candidate_path, baseline_path)
            candidate = load_benchmark_report(candidate_path)
            baseline = load_benchmark_report(baseline_path)
            analysis = load_error_analysis(
                error_analysis_path,
                candidate=candidate,
                candidate_benchmark_sha256=file_digest(candidate_path),
                baseline_benchmark_sha256=file_digest(baseline_path),
            )
            publication = build_publication(
                candidate=candidate,
                baseline=baseline,
                calibration_document=candidate_calibration(),
                error_analysis=analysis,
                candidate_benchmark_sha256=file_digest(candidate_path),
                baseline_benchmark_sha256=file_digest(baseline_path),
                calibration_artifact_sha256=file_digest(calibration_path),
                error_analysis_sha256=file_digest(error_analysis_path),
            )
            output = root / "public-evaluation"
            write_publication(output, publication)

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "evaluation.md",
                    "calibration-curve.svg",
                    "condition-comparison.svg",
                    "latency-comparison.svg",
                    "evaluation-manifest.json",
                },
            )
            markdown = (output / "evaluation.md").read_text(encoding="utf-8")
            manifest = json.loads((output / "evaluation-manifest.json").read_text(encoding="utf-8"))
            self.assertIn("portrait-candidate@release-2026.09.1", markdown)
            self.assertIn("Reviewed error analysis", markdown)
            self.assertIn("detector_reported", markdown)
            self.assertNotIn("record_id", markdown)
            self.assertEqual(manifest["candidate"]["benchmark_sha256"], file_digest(candidate_path))
            self.assertEqual(manifest["error_analysis"]["finding_count"], 1)
            self.assertEqual(manifest["error_analysis"]["artifact_sha256"], file_digest(error_analysis_path))
            self.assertEqual(set(manifest["artifacts"]), {
                "evaluation.md",
                "calibration-curve.svg",
                "condition-comparison.svg",
                "latency-comparison.svg",
            })
            with self.assertRaisesRegex(EvaluationPublicationError, "already exists"):
                write_publication(output, publication)

    def test_rejects_comparisons_that_do_not_share_the_exact_heldout_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate_path, baseline_path, _calibration_path = self._write_inputs(root)
            baseline_document = benchmark_document(
                detector_id="portrait-baseline",
                detector_version="baseline-2026.09.1",
                detector_family="spectral-baseline",
                auroc_offset=-0.04,
                heldout_digest=digest("f"),
            )
            baseline_path.write_text(json.dumps(baseline_document), encoding="utf-8")
            analysis_path = self._write_error_analysis(root, candidate_path, baseline_path)
            candidate = load_benchmark_report(candidate_path)
            baseline = load_benchmark_report(baseline_path)
            analysis = load_error_analysis(
                analysis_path,
                candidate=candidate,
                candidate_benchmark_sha256=file_digest(candidate_path),
                baseline_benchmark_sha256=file_digest(baseline_path),
            )

            with self.assertRaisesRegex(EvaluationPublicationError, "same held-out benchmark inputs"):
                build_publication(
                    candidate=candidate,
                    baseline=baseline,
                    calibration_document=candidate_calibration(),
                    error_analysis=analysis,
                    candidate_benchmark_sha256=file_digest(candidate_path),
                    baseline_benchmark_sha256=file_digest(baseline_path),
                    calibration_artifact_sha256=digest("f"),
                    error_analysis_sha256=file_digest(analysis_path),
                )

    def test_rejects_error_analysis_that_attempts_to_name_private_record_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate_path, baseline_path, _calibration_path = self._write_inputs(root)
            analysis_path = self._write_error_analysis(root, candidate_path, baseline_path)
            payload = json.loads(analysis_path.read_text(encoding="utf-8"))
            payload["findings"][0]["summary"] = "The record_id for one failure is private."
            analysis_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(EvaluationPublicationError, "must not include private"):
                load_error_analysis(
                    analysis_path,
                    candidate=load_benchmark_report(candidate_path),
                    candidate_benchmark_sha256=file_digest(candidate_path),
                    baseline_benchmark_sha256=file_digest(baseline_path),
                )


if __name__ == "__main__":
    unittest.main()
