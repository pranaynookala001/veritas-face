from contextlib import redirect_stderr
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_model import main  # noqa: E402
from model_release import (  # noqa: E402
    AUTHENTIC_VERDICT_THRESHOLD,
    MEANINGFUL_DISAGREEMENT,
    SYNTHETIC_VERDICT_THRESHOLD,
    ModelReleaseValidationError,
    build_model_card,
    build_release_manifest,
    build_threshold_configuration,
    canonical_json_bytes,
    sha256_bytes,
    validate_calibration_artifact,
    validate_run_metadata,
)


def digest(value: bytes) -> str:
    return f"sha256:{sha256(value).hexdigest()}"


CHECKSUM = "sha256:" + "a" * 64


def valid_run_metadata() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "purpose": "fine_tune_only_no_threshold_or_test_reporting",
        "source_revision": "b" * 40,
        "record_manifest_sha256": CHECKSUM,
        "pretrained_weights_sha256": "sha256:" + "c" * 64,
        "checkpoint_sha256": "sha256:" + "d" * 64,
        "selected_epoch": 4,
        "validation_model_selection_auroc": 0.76,
        "model": {
            "architecture": "mobilenet_v3_small",
            "output_label": "fully_synthetic",
        },
        "preprocessing": {
            "image_size": 224,
            "normalization": "ImageNet mean/std via torchvision weight transforms",
        },
    }


def valid_calibration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "calibration_version": "heldout-portraits-2026.09",
        "validation_manifest_sha256": "sha256:" + "e" * 64,
        "detectors": [
            {
                "id": "synthetic-portrait-classifier",
                "version": "release-2026.09.1",
                "family": "mobilenetv3-small",
                "points": [
                    {"raw_probability": 0, "calibrated_probability": 0.01},
                    {"raw_probability": 0.5, "calibrated_probability": 0.48},
                    {"raw_probability": 1, "calibrated_probability": 0.99},
                ],
            }
        ],
    }


class ModelReleaseTests(unittest.TestCase):
    def test_release_metadata_links_exact_training_calibration_and_policy_inputs(self) -> None:
        run = validate_run_metadata(valid_run_metadata())
        calibration_document = valid_calibration()
        calibration = validate_calibration_artifact(
            calibration_document,
            detector_id="synthetic-portrait-classifier",
            detector_version="release-2026.09.1",
            detector_family="mobilenetv3-small",
        )
        calibration_sha256 = sha256_bytes(canonical_json_bytes(calibration_document))

        configuration = build_threshold_configuration(calibration, calibration_sha256)
        card = build_model_card(
            model_version="release-2026.09.1",
            detector_id="synthetic-portrait-classifier",
            detector_family="mobilenetv3-small",
            license_spdx="Apache-2.0",
            run=run,
            checkpoint_sha256=run.checkpoint_sha256,
            onnx_sha256="sha256:" + "f" * 64,
            calibration=calibration,
            calibration_artifact_sha256=calibration_sha256,
        )

        self.assertEqual(configuration["schema_version"], "1.0")
        self.assertEqual(configuration["calibration"]["artifact_sha256"], calibration_sha256)
        self.assertEqual(
            configuration["thresholds"],
            {
                "synthetic_probability_at_least": SYNTHETIC_VERDICT_THRESHOLD,
                "authentic_probability_at_most": AUTHENTIC_VERDICT_THRESHOLD,
                "independent_family_disagreement_at_least": MEANINGFUL_DISAGREEMENT,
            },
        )
        self.assertIn("`normalized_nchw`", card)
        self.assertIn("`synthetic_probability`", card)
        self.assertIn(run.record_manifest_sha256, card)
        self.assertIn(calibration.validation_manifest_sha256, card)
        self.assertIn("does not publish benchmark metrics", card)
        self.assertIn("not proof of origin", card)

    def test_release_manifest_requires_a_checksum_for_every_distributable_artifact(self) -> None:
        artifacts = {
            "calibration.json": CHECKSUM,
            "LICENSE.txt": "sha256:" + "b" * 64,
            "model-card.md": "sha256:" + "c" * 64,
            "portrait-classifier.onnx": "sha256:" + "d" * 64,
            "verdict-policy.json": "sha256:" + "e" * 64,
        }
        manifest = build_release_manifest(
            model_version="release-2026.09.1",
            detector_id="synthetic-portrait-classifier",
            detector_family="mobilenetv3-small",
            artifacts=artifacts,
        )

        self.assertEqual(manifest["artifacts"], dict(sorted(artifacts.items())))
        incomplete = dict(artifacts)
        incomplete.pop("LICENSE.txt")
        with self.assertRaisesRegex(ModelReleaseValidationError, "every required artifact"):
            build_release_manifest(
                model_version="release-2026.09.1",
                detector_id="synthetic-portrait-classifier",
                detector_family="mobilenetv3-small",
                artifacts=incomplete,
            )

    def test_rejects_a_calibration_curve_or_release_that_cannot_safely_apply(self) -> None:
        wrong_family = valid_calibration()
        wrong_family["detectors"][0]["family"] = "another-family"
        with self.assertRaisesRegex(ModelReleaseValidationError, "family does not match"):
            validate_calibration_artifact(
                wrong_family,
                detector_id="synthetic-portrait-classifier",
                detector_version="release-2026.09.1",
                detector_family="mobilenetv3-small",
            )

        non_monotonic = valid_calibration()
        non_monotonic["detectors"][0]["points"][2]["calibrated_probability"] = 0.2
        with self.assertRaisesRegex(ModelReleaseValidationError, "must not decrease"):
            validate_calibration_artifact(
                non_monotonic,
                detector_id="synthetic-portrait-classifier",
                detector_version="release-2026.09.1",
                detector_family="mobilenetv3-small",
            )

    def test_rejects_metadata_without_the_completed_selected_checkpoint_audit_fields(self) -> None:
        incomplete = valid_run_metadata()
        incomplete.pop("checkpoint_sha256")
        with self.assertRaisesRegex(ModelReleaseValidationError, "checkpoint_sha256"):
            validate_run_metadata(incomplete)

        incompatible = valid_run_metadata()
        incompatible["preprocessing"]["image_size"] = 256
        with self.assertRaisesRegex(ModelReleaseValidationError, "image size"):
            validate_run_metadata(incompatible)

    def test_export_cli_rejects_checkpoint_checksum_mismatch_before_importing_torch_or_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"not-a-model")
            metadata = valid_run_metadata()
            metadata["checkpoint_sha256"] = digest(b"another-checkpoint")
            run_metadata = root / "run-metadata.json"
            run_metadata.write_text(json.dumps(metadata), encoding="utf-8")
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps(valid_calibration()), encoding="utf-8")
            license_file = root / "LICENSE.txt"
            license_file.write_text("operator-approved licence text\n", encoding="utf-8")
            output_directory = root / "release"
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                result = main(
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--run-metadata",
                        str(run_metadata),
                        "--calibration-artifact",
                        str(calibration),
                        "--license-file",
                        str(license_file),
                        "--license-spdx",
                        "Apache-2.0",
                        "--model-version",
                        "release-2026.09.1",
                        "--output-dir",
                        str(output_directory),
                    ]
                )

        self.assertEqual(result, 1)
        self.assertIn("does not match run metadata.checkpoint_sha256", stderr.getvalue())
        self.assertFalse(output_directory.exists())


if __name__ == "__main__":
    unittest.main()
