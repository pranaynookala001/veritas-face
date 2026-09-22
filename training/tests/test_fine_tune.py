import copy
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fine_tune import (  # noqa: E402
    RecordManifestValidationError,
    binary_auroc,
    build_run_metadata,
    main,
    validate_record_manifest,
    verify_record_files,
)
from validate_manifests import canonical_sha256, load_json  # noqa: E402


MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"
CONTENT = {
    "train/camera.jpg": b"camera train",
    "validation/camera.jpg": b"camera validation",
    "test/camera.jpg": b"camera test",
    "train/synthetic.png": b"synthetic train",
    "validation/synthetic.png": b"synthetic validation",
    "test/synthetic.png": b"synthetic test",
}


def digest(value: bytes) -> str:
    return f"sha256:{sha256(value).hexdigest()}"


class FineTuneRecordManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = load_json(MANIFEST_DIR / "portrait-sources-v1.json")
        self.splits = load_json(MANIFEST_DIR / "portrait-splits-v1.json")
        self.records = self._valid_record_manifest()

    def _valid_record_manifest(self) -> dict[str, object]:
        source_digest = canonical_sha256(self.sources)
        split_digest = canonical_sha256(self.splits)
        synthetic_sources = {
            "train": (
                "stable-diffusion-v1-5-text-to-image",
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
                "latent_diffusion",
            ),
            "validation": (
                "flux-1-schnell-text-to-image",
                "black-forest-labs/FLUX.1-schnell",
                "741f7c3ce8b383c54771c7003378a50191e9efe9",
                "rectified_flow",
            ),
            "test": (
                "stable-diffusion-xl-base-1-0-text-to-image",
                "stabilityai/stable-diffusion-xl-base-1.0",
                "462165984030d82259a11f4367a4eed129e94a7b",
                "latent_diffusion_xl",
            ),
        }
        records: list[dict[str, object]] = []
        for index, split in enumerate(("train", "validation", "test"), start=1):
            camera_path = f"{split}/camera.jpg"
            records.append(
                {
                    "record_id": f"camera-{split}",
                    "relative_path": camera_path,
                    "sha256": digest(CONTENT[camera_path]),
                    "label": "camera_origin",
                    "source_id": "wikimedia-commons-cc-portrait-photographs",
                    "split": split,
                    "commons_file_page_url": f"https://commons.wikimedia.org/wiki/File:Example-{split}.jpg",
                    "source_file_sha1": f"source-sha1-{index}",
                    "license_spdx": "CC-BY-4.0",
                    "attribution": f"Photographer {index}",
                    "subject_group_id": f"subject-{split}",
                    "capture_group_id": f"capture-{split}",
                }
            )
            source_id, model_id, revision, family = synthetic_sources[split]
            synthetic_path = f"{split}/synthetic.png"
            synthetic_digest = digest(CONTENT[synthetic_path])
            records.append(
                {
                    "record_id": f"synthetic-{split}",
                    "relative_path": synthetic_path,
                    "sha256": synthetic_digest,
                    "label": "fully_synthetic",
                    "source_id": source_id,
                    "split": split,
                    "generator_model_id": model_id,
                    "generator_revision": revision,
                    "generator_family": family,
                    "prompt_template_id": "portrait-neutral-v1",
                    "seed": index,
                    "output_sha256": synthetic_digest,
                }
            )
        return {
            "schema_version": "1.0",
            "manifest_version": "private-portrait-records-2026.09.1",
            "source_manifest_sha256": source_digest,
            "split_manifest_sha256": split_digest,
            "records": records,
        }

    def test_valid_private_records_produce_auditable_summary(self) -> None:
        summary = validate_record_manifest(self.records, self.sources, self.splits)

        self.assertEqual(summary.record_count, 6)
        self.assertEqual(summary.records_per_split, {"test": 2, "train": 2, "validation": 2})
        self.assertEqual(
            summary.labels_per_split,
            {
                "test": ("camera_origin", "fully_synthetic"),
                "train": ("camera_origin", "fully_synthetic"),
                "validation": ("camera_origin", "fully_synthetic"),
            },
        )
        self.assertRegex(summary.record_manifest_sha256, r"^sha256:[0-9a-f]{64}$")

    def test_rejects_camera_subject_leaking_across_splits(self) -> None:
        invalid_records = copy.deepcopy(self.records)
        records = invalid_records["records"]
        assert isinstance(records, list)
        records[2]["subject_group_id"] = "subject-train"

        with self.assertRaisesRegex(RecordManifestValidationError, "appears in both"):
            validate_record_manifest(invalid_records, self.sources, self.splits)

    def test_rejects_generator_or_output_digest_that_does_not_match_catalog(self) -> None:
        invalid_records = copy.deepcopy(self.records)
        records = invalid_records["records"]
        assert isinstance(records, list)
        records[1]["generator_revision"] = "a" * 40

        with self.assertRaisesRegex(RecordManifestValidationError, "catalogued generator"):
            validate_record_manifest(invalid_records, self.sources, self.splits)

        invalid_records = copy.deepcopy(self.records)
        records = invalid_records["records"]
        assert isinstance(records, list)
        records[1]["output_sha256"] = digest(b"different output")
        with self.assertRaisesRegex(RecordManifestValidationError, "must match"):
            validate_record_manifest(invalid_records, self.sources, self.splits)

    def test_verifies_private_files_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative_path, content in CONTENT.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            verify_record_files(self.records, root)

            (root / "train" / "camera.jpg").write_bytes(b"tampered")
            with self.assertRaisesRegex(RecordManifestValidationError, "does not match"):
                verify_record_files(self.records, root)

    def test_run_metadata_captures_exact_run_inputs_without_a_threshold(self) -> None:
        summary = validate_record_manifest(self.records, self.sources, self.splits)
        metadata = build_run_metadata(
            record_summary=summary,
            source_revision="a" * 40,
            seed=20260922,
            image_size=224,
            epochs=8,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0001,
            package_versions={"torch": "2.6.0", "torchvision": "0.21.0"},
        )

        self.assertEqual(metadata["model"]["architecture"], "mobilenet_v3_small")
        self.assertEqual(metadata["purpose"], "fine_tune_only_no_threshold_or_test_reporting")
        self.assertNotIn("threshold", metadata)
        self.assertEqual(metadata["record_manifest_sha256"], summary.record_manifest_sha256)

    def test_binary_auroc_handles_ties_and_rejects_incomplete_labels(self) -> None:
        self.assertEqual(binary_auroc([0.1, 0.4, 0.4, 0.9], [0, 1, 0, 1]), 0.875)
        with self.assertRaisesRegex(RecordManifestValidationError, "both labels"):
            binary_auroc([0.1, 0.2], [1, 1])

    def test_cli_prints_canonical_digest_for_a_valid_private_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            record_path = Path(temporary_directory) / "portrait-records-v1.json"
            record_path.write_text(json.dumps(self.records), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(["--record-manifest", str(record_path)])

        self.assertEqual(result, 0)
        self.assertIn("private training records valid", output.getvalue())
        self.assertIn("canonical=sha256:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
