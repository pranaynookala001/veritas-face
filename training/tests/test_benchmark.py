import copy
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from importlib.util import find_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import (  # noqa: E402
    BenchmarkCondition,
    BenchmarkValidationError,
    ScoreResult,
    ScoredRecord,
    build_benchmark_report,
    command_scorer,
    parse_scorer_command,
    run_benchmark,
    standard_conditions,
    summarize_condition,
    validate_heldout_benchmark,
    verify_heldout_files,
    write_benchmark_report,
)
from validate_manifests import canonical_sha256, load_json  # noqa: E402


MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"
REVISION = "a" * 40


def digest(content: bytes) -> str:
    return f"sha256:{sha256(content).hexdigest()}"


def record_contents() -> dict[str, bytes]:
    return {
        "train/camera.png": b"camera train",
        "validation/camera.png": b"camera validation",
        "test/camera-profile.png": b"camera profile",
        "test/camera-frontal.png": b"camera frontal",
        "train/synthetic.png": b"synthetic train",
        "validation/synthetic.png": b"synthetic validation",
        "test/synthetic-profile.png": b"synthetic profile",
        "test/synthetic-frontal.png": b"synthetic frontal",
    }


def private_records(contents: dict[str, bytes]) -> dict[str, object]:
    sources = load_json(MANIFEST_DIR / "portrait-sources-v1.json")
    splits = load_json(MANIFEST_DIR / "portrait-splits-v1.json")
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
    for number, split in enumerate(("train", "validation"), start=1):
        camera_path = f"{split}/camera.png"
        records.append(
            {
                "record_id": f"camera-{split}",
                "relative_path": camera_path,
                "sha256": digest(contents[camera_path]),
                "label": "camera_origin",
                "source_id": "wikimedia-commons-cc-portrait-photographs",
                "split": split,
                "commons_file_page_url": f"https://commons.wikimedia.org/wiki/File:Example-{split}.png",
                "source_file_sha1": f"source-sha1-{number}",
                "license_spdx": "CC-BY-4.0",
                "attribution": f"Photographer {number}",
                "subject_group_id": f"subject-{split}",
                "capture_group_id": f"capture-{split}",
            }
        )
        source_id, model_id, revision, family = synthetic_sources[split]
        synthetic_path = f"{split}/synthetic.png"
        synthetic_digest = digest(contents[synthetic_path])
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
                "seed": number,
                "output_sha256": synthetic_digest,
            }
        )

    source_id, model_id, revision, family = synthetic_sources["test"]
    for number, (kind, pose_category) in enumerate(
        (("profile", "left_profile"), ("frontal", "frontal")), start=3
    ):
        camera_path = f"test/camera-{kind}.png"
        records.append(
            {
                "record_id": f"camera-test-{kind}",
                "relative_path": camera_path,
                "sha256": digest(contents[camera_path]),
                "label": "camera_origin",
                "source_id": "wikimedia-commons-cc-portrait-photographs",
                "split": "test",
                "commons_file_page_url": f"https://commons.wikimedia.org/wiki/File:Example-test-{kind}.png",
                "source_file_sha1": f"source-sha1-{number}",
                "license_spdx": "CC-BY-4.0",
                "attribution": f"Photographer {number}",
                "subject_group_id": f"subject-test-{kind}",
                "capture_group_id": f"capture-test-{kind}",
                "pose_category": pose_category,
            }
        )
        synthetic_path = f"test/synthetic-{kind}.png"
        synthetic_digest = digest(contents[synthetic_path])
        records.append(
            {
                "record_id": f"synthetic-test-{kind}",
                "relative_path": synthetic_path,
                "sha256": synthetic_digest,
                "label": "fully_synthetic",
                "source_id": source_id,
                "split": "test",
                "generator_model_id": model_id,
                "generator_revision": revision,
                "generator_family": family,
                "prompt_template_id": "portrait-neutral-v1",
                "seed": number,
                "output_sha256": synthetic_digest,
                "pose_category": pose_category,
            }
        )
    return {
        "schema_version": "1.0",
        "manifest_version": "private-benchmark-records-2026.09.1",
        "source_manifest_sha256": canonical_sha256(sources),
        "split_manifest_sha256": canonical_sha256(splits),
        "records": records,
    }


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = load_json(MANIFEST_DIR / "portrait-sources-v1.json")
        self.splits = load_json(MANIFEST_DIR / "portrait-splits-v1.json")
        self.contents = record_contents()
        self.records = private_records(self.contents)

    def test_validated_benchmark_uses_only_test_records_and_requires_profile_coverage(self) -> None:
        benchmark = validate_heldout_benchmark(self.records, self.sources, self.splits)

        self.assertEqual(len(benchmark.records), 4)
        self.assertTrue(all(record.relative_path.startswith("test/") for record in benchmark.records))
        self.assertRegex(benchmark.heldout_records_sha256, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(benchmark.heldout_records_sha256, benchmark.record_manifest_sha256)

        no_profile = copy.deepcopy(self.records)
        raw_records = no_profile["records"]
        assert isinstance(raw_records, list)
        for record in raw_records:
            if record["split"] == "test":
                record["pose_category"] = "frontal"
        with self.assertRaisesRegex(BenchmarkValidationError, "no profile-pose"):
            validate_heldout_benchmark(no_profile, self.sources, self.splits)

        profile_has_one_label = copy.deepcopy(self.records)
        raw_records = profile_has_one_label["records"]
        assert isinstance(raw_records, list)
        for record in raw_records:
            if record["record_id"] == "synthetic-test-profile":
                record["pose_category"] = "frontal"
        with self.assertRaisesRegex(BenchmarkValidationError, "profile-pose records must contain both labels"):
            validate_heldout_benchmark(profile_has_one_label, self.sources, self.splits)

    def test_heldout_file_verification_never_requires_training_or_validation_pixels(self) -> None:
        benchmark = validate_heldout_benchmark(self.records, self.sources, self.splits)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for record in benchmark.records:
                path = root / record.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.contents[record.relative_path])
            verify_heldout_files(benchmark, root)

            (root / benchmark.records[0].relative_path).write_bytes(b"tampered")
            with self.assertRaisesRegex(BenchmarkValidationError, "does not match"):
                verify_heldout_files(benchmark, root)

    def test_standard_condition_matrix_covers_every_required_robustness_dimension(self) -> None:
        self.assertEqual(
            {condition.kind for condition in standard_conditions()},
            {"clean", "jpeg", "resize", "crop", "filters", "profile_pose", "blur", "occlusion"},
        )

    def test_scorer_command_is_shell_free_and_requires_one_image_placeholder(self) -> None:
        command = parse_scorer_command('adapter --image "{image_path}" --mode release')
        self.assertEqual(command, ("adapter", "--image", "{image_path}", "--mode", "release"))
        for invalid in ("adapter", "adapter {image_path} {image_path}", ""):
            with self.assertRaisesRegex(BenchmarkValidationError, "placeholder|must not be empty"):
                parse_scorer_command(invalid)

        scorer = command_scorer('python3 -c "import sys; print(\'{\\\"synthetic_probability\\\": 0.6}\')" {image_path}')
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = scorer(Path(temporary_directory) / "temporary.png")
        self.assertEqual(result, ScoreResult(0.6, None))

    def test_condition_summary_uses_auroc_without_selecting_a_threshold(self) -> None:
        condition = BenchmarkCondition("jpeg_q75", "jpeg", {"quality": 75})
        summary = summarize_condition(
            condition,
            (
                ScoredRecord("camera_origin", ScoreResult(0.1, 3.0), 5.0),
                ScoredRecord("camera_origin", ScoreResult(0.2, 4.0), 6.0),
                ScoredRecord("fully_synthetic", ScoreResult(0.8, 5.0), 7.0),
                ScoredRecord("fully_synthetic", ScoreResult(0.9, 6.0), 8.0),
            ),
        )
        self.assertEqual(summary["auroc"], 1.0)
        self.assertNotIn("threshold", summary)
        self.assertEqual(summary["latency_ms"]["detector_reported"]["p95"], 6.0)

        with self.assertRaisesRegex(BenchmarkValidationError, "mixes reported and missing"):
            summarize_condition(
                condition,
                (
                    ScoredRecord("camera_origin", ScoreResult(0.1), 1.0),
                    ScoredRecord("fully_synthetic", ScoreResult(0.8, 2.0), 2.0),
                ),
            )

    def test_report_links_digests_and_refuses_to_overwrite_private_results(self) -> None:
        benchmark = validate_heldout_benchmark(self.records, self.sources, self.splits)
        report = build_benchmark_report(
            benchmark=benchmark,
            source_revision=REVISION,
            detector_id="synthetic-portrait-classifier",
            detector_version="release-2026.09.1",
            detector_family="mobilenetv3-small",
            scorer_identity="release-2026.09.1-cpp-adapter",
            conditions=(),
        )
        self.assertEqual(
            report["purpose"], "held_out_robustness_evaluation_not_calibration_or_deployment"
        )
        self.assertNotIn("thresholds", report)
        self.assertIn(benchmark.heldout_records_sha256, json.dumps(report))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "benchmark.json"
            write_benchmark_report(output, report)
            with self.assertRaisesRegex(BenchmarkValidationError, "already exists"):
                write_benchmark_report(output, report)

    @unittest.skipUnless(find_spec("PIL"), "Pillow is installed in the private evaluation environment")
    def test_runner_rejects_a_digest_valid_file_that_is_not_a_decodable_image(self) -> None:
        benchmark = validate_heldout_benchmark(self.records, self.sources, self.splits)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for record in benchmark.records:
                path = root / record.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.contents[record.relative_path])
            verify_heldout_files(benchmark, root)
            with self.assertRaisesRegex(BenchmarkValidationError, "could not be decoded"):
                run_benchmark(
                    benchmark=benchmark,
                    data_root=root,
                    scorer=lambda _path: ScoreResult(0.5),
                )

    @unittest.skipUnless(find_spec("PIL"), "Pillow is installed in the private evaluation environment")
    def test_runner_materializes_all_conditions_and_only_exposes_transformed_images(self) -> None:
        from PIL import Image

        image_contents: dict[str, bytes] = {}
        for path, color in (
            ("train/camera.png", (200, 0, 0)),
            ("validation/camera.png", (180, 0, 0)),
            ("test/camera-profile.png", (160, 0, 0)),
            ("test/camera-frontal.png", (140, 0, 0)),
            ("train/synthetic.png", (0, 0, 200)),
            ("validation/synthetic.png", (0, 0, 180)),
            ("test/synthetic-profile.png", (0, 0, 160)),
            ("test/synthetic-frontal.png", (0, 0, 140)),
        ):
            output = io.BytesIO()
            Image.new("RGB", (24, 20), color).save(output, format="PNG")
            image_contents[path] = output.getvalue()
        records = private_records(image_contents)
        benchmark = validate_heldout_benchmark(records, self.sources, self.splits)

        seen_paths: list[Path] = []

        def scorer(path: Path) -> ScoreResult:
            seen_paths.append(path)
            with Image.open(path) as image:
                red, _green, blue = image.getpixel((0, 0))
            return ScoreResult(0.9 if blue > red else 0.1, latency_ms=1.5)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative_path, content in image_contents.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            verify_heldout_files(benchmark, root)
            reports = run_benchmark(benchmark=benchmark, data_root=root, scorer=scorer)

        self.assertEqual(len(reports), 8)
        self.assertEqual(len(seen_paths), 30)
        self.assertTrue(all(path.name.startswith(tuple(condition.identifier for condition in standard_conditions())) for path in seen_paths))
        self.assertEqual({report["kind"] for report in reports}, {condition.kind for condition in standard_conditions()})
        self.assertTrue(all(report["auroc"] == 1.0 for report in reports))


if __name__ == "__main__":
    unittest.main()
