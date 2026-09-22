#!/usr/bin/env python3
"""Run a private, held-out robustness benchmark for a Veritas Face detector.

The runner deliberately works from the immutable *test* records of a private
record manifest.  It validates the catalog and split plan before it reads a
pixel, verifies every test-image digest, applies fixed robustness conditions,
and passes only a temporary transformed image to an explicitly identified
scorer.  It does not calibrate a detector, choose a threshold, or publish a
performance claim.

The scorer is an explicit command containing one ``{image_path}`` placeholder.
It must run the same private face-selection, preprocessing, and inference path
as the release under evaluation, then write one JSON object to stdout:

    {"synthetic_probability": 0.73, "latency_ms": 14.2}

``latency_ms`` is optional.  No shell is used to invoke the scorer.  Images
materialised for scoring live in a temporary directory and are deleted when
the benchmark exits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
from time import perf_counter

from fine_tune import (
    LABEL_TO_TARGET,
    RecordManifestValidationError,
    binary_auroc,
    load_record_manifest,
    validate_record_manifest,
)
from validate_manifests import (
    ManifestValidationError,
    canonical_sha256,
    load_json,
)


BENCHMARK_SCHEMA_VERSION = "1.0"
RUNNER_VERSION = "1.0"
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
POSE_CATEGORIES = frozenset({"frontal", "left_profile", "right_profile"})
PROFILE_POSES = frozenset({"left_profile", "right_profile"})
SCORER_TIMEOUT_SECONDS = 60


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark could be non-held-out, ambiguous, or unreproducible."""


@dataclass(frozen=True)
class HeldoutRecord:
    """The minimal private facts needed to score one approved test record."""

    record_id: str
    relative_path: str
    sha256: str
    label: str
    pose_category: str


@dataclass(frozen=True)
class HeldoutBenchmark:
    """Validated held-out records and anonymous audit digests for one run."""

    records: tuple[HeldoutRecord, ...]
    record_manifest_sha256: str
    heldout_records_sha256: str
    source_manifest_sha256: str
    split_manifest_sha256: str


@dataclass(frozen=True)
class BenchmarkCondition:
    """A fixed transformation or annotation-defined subgroup."""

    identifier: str
    kind: str
    parameters: Mapping[str, object]
    profile_only: bool = False


@dataclass(frozen=True)
class ScoreResult:
    """One finite detector probability and its optional reported inference time."""

    synthetic_probability: float
    latency_ms: float | None = None


@dataclass(frozen=True)
class ScoredRecord:
    """Private intermediate data used only to aggregate one condition."""

    label: str
    score: ScoreResult
    wall_latency_ms: float


def standard_conditions() -> tuple[BenchmarkCondition, ...]:
    """Return the versioned condition matrix used by every v1 benchmark run."""
    return (
        BenchmarkCondition("clean", "clean", {}),
        BenchmarkCondition("jpeg_q75", "jpeg", {"quality": 75, "subsampling": "4:4:4"}),
        BenchmarkCondition("resize_half", "resize", {"scale": 0.5, "resample": "lanczos"}),
        BenchmarkCondition("center_crop_80", "crop", {"retained_fraction": 0.8, "resample": "lanczos"}),
        BenchmarkCondition(
            "desaturate_contrast",
            "filters",
            {"color_factor": 0.5, "contrast_factor": 1.2},
        ),
        BenchmarkCondition("gaussian_blur_2", "blur", {"radius_px": 2.0}),
        BenchmarkCondition("center_occlusion_25", "occlusion", {"area_fraction": 0.25, "fill": "#808080"}),
        BenchmarkCondition(
            "profile_pose",
            "profile_pose",
            {"included_pose_categories": sorted(PROFILE_POSES)},
            profile_only=True,
        ),
    )


def validate_heldout_benchmark(
    record_document: Mapping[str, object],
    source_document: Mapping[str, object],
    split_document: Mapping[str, object],
) -> HeldoutBenchmark:
    """Reject anything except labelled, profile-annotated records from ``test``.

    The shared record validator establishes source provenance, label integrity,
    and split disjointness.  This second check makes the later test benchmark
    fail closed if pose coverage is absent or if either label is missing from
    the profile-pose subgroup.
    """
    try:
        summary = validate_record_manifest(record_document, source_document, split_document)
    except RecordManifestValidationError as error:
        raise BenchmarkValidationError(str(error)) from error

    raw_records = record_document.get("records")
    if not isinstance(raw_records, list):
        raise BenchmarkValidationError("record manifest.records must be a list")

    heldout_records: list[HeldoutRecord] = []
    heldout_raw_records: list[Mapping[str, object]] = []
    for index, raw_record in enumerate(raw_records):
        location = f"records[{index}]"
        if not isinstance(raw_record, dict):
            raise BenchmarkValidationError(f"{location} must be an object")
        if raw_record.get("split") != "test":
            continue
        pose_category = raw_record.get("pose_category")
        if pose_category not in POSE_CATEGORIES:
            raise BenchmarkValidationError(
                f"{location}.pose_category must be one of {sorted(POSE_CATEGORIES)}"
            )
        record_id = _nonempty_string(raw_record.get("record_id"), f"{location}.record_id")
        relative_path = _safe_relative_path(
            raw_record.get("relative_path"), f"{location}.relative_path"
        )
        digest = _sha256(raw_record.get("sha256"), f"{location}.sha256")
        label = raw_record.get("label")
        if label not in LABEL_TO_TARGET:
            raise BenchmarkValidationError(f"{location}.label is not a supported benchmark label")
        heldout_records.append(
            HeldoutRecord(
                record_id=record_id,
                relative_path=relative_path,
                sha256=digest,
                label=label,
                pose_category=pose_category,
            )
        )
        heldout_raw_records.append(raw_record)

    if not heldout_records:
        raise BenchmarkValidationError("record manifest contains no held-out test records")
    _require_both_labels(heldout_records, "held-out test records")
    profile_records = [record for record in heldout_records if record.pose_category in PROFILE_POSES]
    if not profile_records:
        raise BenchmarkValidationError("held-out test records contain no profile-pose examples")
    _require_both_labels(profile_records, "held-out profile-pose records")

    return HeldoutBenchmark(
        records=tuple(heldout_records),
        record_manifest_sha256=summary.record_manifest_sha256,
        heldout_records_sha256=canonical_sha256(heldout_raw_records),
        source_manifest_sha256=canonical_sha256(source_document),
        split_manifest_sha256=canonical_sha256(split_document),
    )


def verify_heldout_files(benchmark: HeldoutBenchmark, data_root: Path) -> None:
    """Hash only the held-out files before the scorer can receive their pixels."""
    root = data_root.resolve()
    if not root.is_dir():
        raise BenchmarkValidationError(f"data root is not a directory: {data_root}")
    for record in benchmark.records:
        image_path = _record_path(root, record.relative_path)
        if not image_path.is_file():
            raise BenchmarkValidationError(f"held-out image is missing: {record.relative_path}")
        if _file_sha256(image_path) != record.sha256:
            raise BenchmarkValidationError(
                f"held-out image sha256 does not match manifest for {record.relative_path!r}"
            )


def parse_scorer_command(command: str) -> tuple[str, ...]:
    """Parse one explicit scorer command without shell evaluation or ambiguity."""
    try:
        arguments = tuple(shlex.split(command))
    except ValueError as error:
        raise BenchmarkValidationError(f"invalid scorer command: {error}") from error
    if not arguments:
        raise BenchmarkValidationError("scorer command must not be empty")
    placeholder_count = sum(argument.count("{image_path}") for argument in arguments)
    if placeholder_count != 1:
        raise BenchmarkValidationError(
            "scorer command must contain exactly one {image_path} placeholder"
        )
    return arguments


def command_scorer(command: str) -> Callable[[Path], ScoreResult]:
    """Return a scorer that invokes an adapter with one temporary image path."""
    command_arguments = parse_scorer_command(command)

    def score(image_path: Path) -> ScoreResult:
        arguments = tuple(
            argument.replace("{image_path}", str(image_path)) for argument in command_arguments
        )
        try:
            completed = subprocess.run(
                arguments,
                check=False,
                capture_output=True,
                text=True,
                timeout=SCORER_TIMEOUT_SECONDS,
            )
        except OSError as error:
            raise BenchmarkValidationError(f"scorer could not start: {error}") from error
        except subprocess.TimeoutExpired as error:
            raise BenchmarkValidationError(
                f"scorer exceeded {SCORER_TIMEOUT_SECONDS} seconds"
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip()
            if len(detail) > 500:
                detail = detail[:500] + "…"
            raise BenchmarkValidationError(
                f"scorer exited with status {completed.returncode}" + (f": {detail}" if detail else "")
            )
        return _parse_scorer_result(completed.stdout)

    return score


def run_benchmark(
    *,
    benchmark: HeldoutBenchmark,
    data_root: Path,
    scorer: Callable[[Path], ScoreResult],
) -> tuple[dict[str, object], ...]:
    """Create temporary transformed images, score them, and aggregate metrics."""
    pillow = _load_pillow()
    root = data_root.resolve()
    conditions = standard_conditions()
    condition_reports: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="veritas-face-benchmark-") as temporary_directory:
        workspace = Path(temporary_directory)
        for condition in conditions:
            records = tuple(
                record
                for record in benchmark.records
                if not condition.profile_only or record.pose_category in PROFILE_POSES
            )
            _require_both_labels(records, f"benchmark condition {condition.identifier!r}")
            scored_records: list[ScoredRecord] = []
            for index, record in enumerate(records):
                source_path = _record_path(root, record.relative_path)
                suffix = ".jpg" if condition.kind == "jpeg" else ".png"
                transformed_path = workspace / f"{condition.identifier}-{index:05d}{suffix}"
                _materialize_transformation(pillow, source_path, transformed_path, condition)
                started = perf_counter()
                result = scorer(transformed_path)
                wall_latency_ms = (perf_counter() - started) * 1000
                _validate_score_result(result)
                scored_records.append(
                    ScoredRecord(
                        label=record.label,
                        score=result,
                        wall_latency_ms=wall_latency_ms,
                    )
                )
            condition_reports.append(summarize_condition(condition, scored_records))
    return tuple(condition_reports)


def summarize_condition(
    condition: BenchmarkCondition, scored_records: Sequence[ScoredRecord]
) -> dict[str, object]:
    """Aggregate discrimination and latency facts without inventing a threshold."""
    if not scored_records:
        raise BenchmarkValidationError(f"benchmark condition {condition.identifier!r} has no scores")
    labels = [record.label for record in scored_records]
    if set(labels) != set(LABEL_TO_TARGET):
        raise BenchmarkValidationError(
            f"benchmark condition {condition.identifier!r} must contain both labels"
        )
    scores = [record.score.synthetic_probability for record in scored_records]
    targets = [LABEL_TO_TARGET[record.label] for record in scored_records]
    try:
        auroc = binary_auroc(scores, targets)
    except RecordManifestValidationError as error:
        raise BenchmarkValidationError(str(error)) from error

    probabilities_by_label = {
        label: [record.score.synthetic_probability for record in scored_records if record.label == label]
        for label in sorted(LABEL_TO_TARGET)
    }
    reported_latency = [
        record.score.latency_ms
        for record in scored_records
        if record.score.latency_ms is not None
    ]
    if reported_latency and len(reported_latency) != len(scored_records):
        raise BenchmarkValidationError(
            f"benchmark condition {condition.identifier!r} mixes reported and missing scorer latency"
        )

    report: dict[str, object] = {
        "id": condition.identifier,
        "kind": condition.kind,
        "parameters": dict(condition.parameters),
        "sample_count": len(scored_records),
        "labels": dict(sorted(Counter(labels).items())),
        "auroc": auroc,
        "mean_synthetic_probability": {
            label: _mean(values) for label, values in probabilities_by_label.items()
        },
        "latency_ms": {
            "scorer_wall": _latency_summary([record.wall_latency_ms for record in scored_records]),
            "detector_reported": _latency_summary(reported_latency) if reported_latency else None,
        },
    }
    return report


def build_benchmark_report(
    *,
    benchmark: HeldoutBenchmark,
    source_revision: str,
    detector_id: str,
    detector_version: str,
    detector_family: str,
    scorer_identity: str,
    conditions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the non-sensitive private output document for a completed run."""
    if not GIT_REVISION_PATTERN.fullmatch(source_revision):
        raise BenchmarkValidationError("source_revision must be a 40-character lowercase Git revision")
    detector = {
        "id": _nonempty_string(detector_id, "detector_id"),
        "version": _nonempty_string(detector_version, "detector_version"),
        "family": _nonempty_string(detector_family, "detector_family"),
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "purpose": "held_out_robustness_evaluation_not_calibration_or_deployment",
        "runner": {"name": "veritas-face-benchmark", "version": RUNNER_VERSION},
        "source_revision": source_revision,
        "detector": detector,
        "scorer_identity": _nonempty_string(scorer_identity, "scorer_identity"),
        "inputs": {
            "record_manifest_sha256": benchmark.record_manifest_sha256,
            "heldout_test_records_sha256": benchmark.heldout_records_sha256,
            "source_manifest_sha256": benchmark.source_manifest_sha256,
            "split_manifest_sha256": benchmark.split_manifest_sha256,
        },
        "conditions": [dict(condition) for condition in conditions],
        "limitations": [
            "Scores are probabilistic detector evidence, not proof of image origin.",
            "This benchmark does not calibrate the detector or choose a verdict threshold.",
            "Results apply only to the reviewed held-out records and fixed conditions in this run.",
            "V1 evaluates fully synthetic portraits only; it does not establish face-swap or general authenticity performance.",
        ],
    }


def write_benchmark_report(path: Path, report: Mapping[str, object]) -> None:
    """Create, but never overwrite, a private benchmark result file."""
    if path.exists():
        raise BenchmarkValidationError(f"benchmark output already exists: {path}")
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(payload)
    except OSError as error:
        raise BenchmarkValidationError(f"could not create benchmark output {path}: {error}") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-manifest", required=True, type=Path, help="private record manifest")
    parser.add_argument("--data-root", required=True, type=Path, help="private root containing records")
    parser.add_argument(
        "--source-manifest-dir",
        type=Path,
        default=Path(__file__).with_name("manifests"),
        help="directory containing portrait-sources-v1.json and portrait-splits-v1.json",
    )
    parser.add_argument(
        "--scorer-command",
        required=True,
        help="adapter command with exactly one {image_path} placeholder; no shell is used",
    )
    parser.add_argument(
        "--scorer-identity",
        required=True,
        help="immutable release or adapter identity recorded without command paths",
    )
    parser.add_argument("--detector-id", required=True, help="detector identifier")
    parser.add_argument("--detector-version", required=True, help="immutable detector release version")
    parser.add_argument("--detector-family", required=True, help="detector family identifier")
    parser.add_argument(
        "--source-revision",
        required=True,
        help="40-character Git revision for this runner source snapshot",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new private JSON result path; existing results are never overwritten",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs, run the scorer matrix, then create one private JSON result."""
    args = parse_args(argv)
    try:
        source_document = load_json(args.source_manifest_dir / "portrait-sources-v1.json")
        split_document = load_json(args.source_manifest_dir / "portrait-splits-v1.json")
        benchmark = validate_heldout_benchmark(
            load_record_manifest(args.record_manifest), source_document, split_document
        )
        verify_heldout_files(benchmark, args.data_root)
        reports = run_benchmark(
            benchmark=benchmark,
            data_root=args.data_root,
            scorer=command_scorer(args.scorer_command),
        )
        report = build_benchmark_report(
            benchmark=benchmark,
            source_revision=args.source_revision,
            detector_id=args.detector_id,
            detector_version=args.detector_version,
            detector_family=args.detector_family,
            scorer_identity=args.scorer_identity,
            conditions=reports,
        )
        write_benchmark_report(args.output, report)
    except (BenchmarkValidationError, ManifestValidationError, RecordManifestValidationError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1
    print(
        "held-out benchmark complete: "
        f"{len(benchmark.records)} records; {len(reports)} conditions; output={args.output}"
    )
    return 0


def _load_pillow() -> Mapping[str, object]:
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
    except ImportError as error:
        raise BenchmarkValidationError(
            "benchmark image transformations require Pillow in the private evaluation environment"
        ) from error
    return {
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageEnhance": ImageEnhance,
        "ImageFilter": ImageFilter,
        "ImageOps": ImageOps,
    }


def _materialize_transformation(
    pillow: Mapping[str, object],
    source_path: Path,
    destination_path: Path,
    condition: BenchmarkCondition,
) -> None:
    image_module = pillow["Image"]
    image_ops = pillow["ImageOps"]
    image_enhance = pillow["ImageEnhance"]
    image_filter = pillow["ImageFilter"]
    image_draw = pillow["ImageDraw"]
    try:
        with image_module.open(source_path) as opened:  # type: ignore[union-attr]
            image = image_ops.exif_transpose(opened).convert("RGB")  # type: ignore[union-attr]
    except (OSError, ValueError) as error:
        raise BenchmarkValidationError(
            f"held-out image could not be decoded: {source_path.name}"
        ) from error

    if condition.kind == "clean" or condition.kind == "profile_pose":
        transformed = image
    elif condition.kind == "jpeg":
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination_path, format="JPEG", quality=75, subsampling=0, optimize=False)
        return
    elif condition.kind == "resize":
        reduced_size = (max(1, round(image.width * 0.5)), max(1, round(image.height * 0.5)))
        reduced = image.resize(reduced_size, image_module.Resampling.LANCZOS)  # type: ignore[union-attr]
        transformed = reduced.resize(image.size, image_module.Resampling.LANCZOS)  # type: ignore[union-attr]
    elif condition.kind == "crop":
        cropped_width = max(1, round(image.width * 0.8))
        cropped_height = max(1, round(image.height * 0.8))
        left = (image.width - cropped_width) // 2
        top = (image.height - cropped_height) // 2
        cropped = image.crop((left, top, left + cropped_width, top + cropped_height))
        transformed = cropped.resize(image.size, image_module.Resampling.LANCZOS)  # type: ignore[union-attr]
    elif condition.kind == "filters":
        desaturated = image_enhance.Color(image).enhance(0.5)  # type: ignore[union-attr]
        transformed = image_enhance.Contrast(desaturated).enhance(1.2)  # type: ignore[union-attr]
    elif condition.kind == "blur":
        transformed = image.filter(image_filter.GaussianBlur(radius=2.0))  # type: ignore[union-attr]
    elif condition.kind == "occlusion":
        transformed = image.copy()
        occlusion_width = max(1, round(image.width * 0.5))
        occlusion_height = max(1, round(image.height * 0.5))
        left = (image.width - occlusion_width) // 2
        top = (image.height - occlusion_height) // 2
        image_draw.Draw(transformed).rectangle(  # type: ignore[union-attr]
            (left, top, left + occlusion_width, top + occlusion_height),
            fill=(128, 128, 128),
        )
    else:
        raise BenchmarkValidationError(f"unsupported benchmark condition {condition.kind!r}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.save(destination_path, format="PNG", optimize=False)


def _parse_scorer_result(stdout: str) -> ScoreResult:
    if len(stdout.encode("utf-8")) > 64 * 1024:
        raise BenchmarkValidationError("scorer response exceeds 64 KiB")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkValidationError("scorer did not return valid JSON") from error
    if not isinstance(payload, dict) or set(payload) - {"synthetic_probability", "latency_ms"}:
        raise BenchmarkValidationError(
            "scorer JSON must contain only synthetic_probability and optional latency_ms"
        )
    score = _finite_probability(payload.get("synthetic_probability"), "scorer synthetic_probability")
    latency = payload.get("latency_ms")
    if latency is None:
        return ScoreResult(score)
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise BenchmarkValidationError("scorer latency_ms must be a finite non-negative number")
    return ScoreResult(score, float(latency))


def _validate_score_result(result: ScoreResult) -> None:
    if not isinstance(result, ScoreResult):
        raise BenchmarkValidationError("scorer must return a ScoreResult")
    _finite_probability(result.synthetic_probability, "scorer synthetic_probability")
    if result.latency_ms is not None:
        if (
            isinstance(result.latency_ms, bool)
            or not isinstance(result.latency_ms, (int, float))
            or not math.isfinite(result.latency_ms)
            or result.latency_ms < 0
        ):
            raise BenchmarkValidationError("scorer latency_ms must be a finite non-negative number")


def _record_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise BenchmarkValidationError(f"relative path escapes data root: {relative_path}")
    return path


def _safe_relative_path(value: object, location: str) -> str:
    relative_path = _nonempty_string(value, location)
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise BenchmarkValidationError(f"{location} must be a safe relative path")
    return path.as_posix()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BenchmarkValidationError(f"cannot read {path}: {error}") from error
    return f"sha256:{digest.hexdigest()}"


def _require_both_labels(records: Sequence[HeldoutRecord], description: str) -> None:
    labels = {record.label for record in records}
    if labels != set(LABEL_TO_TARGET):
        missing = sorted(set(LABEL_TO_TARGET) - labels)
        raise BenchmarkValidationError(f"{description} must contain both labels; missing {', '.join(missing)}")


def _finite_probability(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BenchmarkValidationError(f"{location} must be a finite probability")
    if not 0 <= float(value) <= 1:
        raise BenchmarkValidationError(f"{location} must be from 0 through 1")
    return float(value)


def _nonempty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkValidationError(f"{location} must be a non-empty string")
    return value.strip()


def _sha256(value: object, location: str) -> str:
    result = _nonempty_string(value, location)
    if not SHA256_PATTERN.fullmatch(result):
        raise BenchmarkValidationError(f"{location} must be sha256:<64 lowercase hex>")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise BenchmarkValidationError("mean requires at least one value")
    return sum(values) / len(values)


def _latency_summary(values: Sequence[float]) -> Mapping[str, float | int]:
    if not values:
        raise BenchmarkValidationError("latency summary requires at least one value")
    ordered = sorted(values)
    percentile_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(ordered),
        "mean": _mean(ordered),
        "p50": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[percentile_index],
    }


if __name__ == "__main__":
    raise SystemExit(main())
