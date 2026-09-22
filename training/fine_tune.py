"""Private-record validation and reproducibility helpers for fine-tuning.

The notebook in this directory deliberately takes its pixels from a private
Kaggle input.  This module validates the accompanying private record manifest
before PyTorch is imported or a model is trained.  It uses only the standard
library so the policy checks can run in the repository quality gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

from validate_manifests import (
    LABELS,
    SPLITS,
    ManifestValidationError,
    canonical_sha256,
    load_json,
    validate_manifests,
)


RECORD_SCHEMA_VERSION = "1.0"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
LABEL_TO_TARGET = {"camera_origin": 0, "fully_synthetic": 1}


class RecordManifestValidationError(ValueError):
    """Raised when private training records cannot support a safe run."""


@dataclass(frozen=True)
class RecordManifestSummary:
    """Audit facts that identify one immutable private training input."""

    record_count: int
    record_manifest_sha256: str
    records_per_split: Mapping[str, int]
    labels_per_split: Mapping[str, tuple[str, ...]]


def load_record_manifest(path: Path) -> Mapping[str, object]:
    """Load the private JSON record manifest without reading image pixels."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RecordManifestValidationError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RecordManifestValidationError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(document, dict):
        raise RecordManifestValidationError(f"{path} must contain a JSON object")
    return document


def validate_record_manifest(
    record_document: Mapping[str, object],
    source_document: Mapping[str, object],
    split_document: Mapping[str, object],
) -> RecordManifestSummary:
    """Validate private records against the checked-in source and split plans.

    The manifest lists relative image paths and SHA-256 digests, but no image
    bytes.  Camera-origin records are identity and capture group disjoint;
    synthetic records are pinned to their catalogued text-to-image generator.
    """
    try:
        source_summary = validate_manifests(source_document, split_document)
    except ManifestValidationError as error:
        raise RecordManifestValidationError(str(error)) from error

    _require_equal(record_document, "schema_version", RECORD_SCHEMA_VERSION, "record manifest")
    _require_nonempty_string(record_document, "manifest_version", "record manifest")
    _require_equal(
        record_document,
        "source_manifest_sha256",
        source_summary.source_sha256,
        "record manifest",
    )
    _require_equal(
        record_document,
        "split_manifest_sha256",
        source_summary.split_sha256,
        "record manifest",
    )
    records = _require_list(record_document, "records", "record manifest")
    if not records:
        raise RecordManifestValidationError("record manifest must contain at least one record")

    sources = _require_list(source_document, "sources", "source manifest")
    sources_by_id = {
        source["id"]: source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    }
    split_sources = _split_sources(split_document)

    record_ids: set[str] = set()
    relative_paths: set[str] = set()
    content_digests: set[str] = set()
    subject_splits: dict[str, str] = {}
    capture_splits: dict[str, str] = {}
    labels_per_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    records_per_split: Counter[str] = Counter()

    for index, record in enumerate(records):
        location = f"records[{index}]"
        if not isinstance(record, dict):
            raise RecordManifestValidationError(f"{location} must be an object")
        record_id = _require_nonempty_string(record, "record_id", location)
        if record_id in record_ids:
            raise RecordManifestValidationError(f"record manifest repeats record_id {record_id!r}")
        record_ids.add(record_id)

        split = _require_nonempty_string(record, "split", location)
        if split not in SPLITS:
            raise RecordManifestValidationError(f"{location}.split must be one of {sorted(SPLITS)}")
        label = _require_nonempty_string(record, "label", location)
        if label not in LABELS:
            raise RecordManifestValidationError(f"{location}.label must be one of {sorted(LABELS)}")
        source_id = _require_nonempty_string(record, "source_id", location)
        source = sources_by_id.get(source_id)
        if source is None:
            raise RecordManifestValidationError(f"{location}.source_id is not in the source manifest")
        if source.get("label") != label:
            raise RecordManifestValidationError(f"{location}.label does not match source {source_id!r}")
        if source_id not in split_sources[split]:
            raise RecordManifestValidationError(
                f"{location}.source_id {source_id!r} is not assigned to split {split!r}"
            )

        relative_path = _validate_relative_path(record, location)
        if relative_path in relative_paths:
            raise RecordManifestValidationError(
                f"record manifest repeats relative_path {relative_path!r}"
            )
        relative_paths.add(relative_path)
        digest = _require_sha256(record, "sha256", location)
        if digest in content_digests:
            raise RecordManifestValidationError(
                f"record manifest repeats image sha256 {digest!r}"
            )
        content_digests.add(digest)

        record_requirements = source.get("record_requirements")
        if not isinstance(record_requirements, list):
            raise RecordManifestValidationError(f"source {source_id!r} has invalid record_requirements")
        for required_key in record_requirements:
            if not isinstance(required_key, str):
                raise RecordManifestValidationError(
                    f"source {source_id!r} has invalid record requirement"
                )
            _require_record_value(record, required_key, location)

        if label == "camera_origin":
            _validate_camera_record(record, split, subject_splits, capture_splits, location)
        else:
            _validate_synthetic_record(record, source, digest, location)

        labels_per_split[split].add(label)
        records_per_split[split] += 1

    for split in sorted(SPLITS):
        missing_labels = LABELS - labels_per_split[split]
        if missing_labels:
            raise RecordManifestValidationError(
                f"record manifest split {split!r} is missing labels: "
                f"{', '.join(sorted(missing_labels))}"
            )

    return RecordManifestSummary(
        record_count=len(records),
        record_manifest_sha256=canonical_sha256(record_document),
        records_per_split=dict(sorted(records_per_split.items())),
        labels_per_split={
            split: tuple(sorted(labels_per_split[split])) for split in sorted(SPLITS)
        },
    )


def verify_record_files(record_document: Mapping[str, object], data_root: Path) -> None:
    """Verify every private record exists under *data_root* and matches its digest."""
    records = _require_list(record_document, "records", "record manifest")
    root = data_root.resolve()
    for index, record in enumerate(records):
        location = f"records[{index}]"
        if not isinstance(record, dict):
            raise RecordManifestValidationError(f"{location} must be an object")
        relative_path = _validate_relative_path(record, location)
        expected_digest = _require_sha256(record, "sha256", location)
        path = (root / relative_path).resolve()
        if path != root and root not in path.parents:
            raise RecordManifestValidationError(f"{location}.relative_path escapes data_root")
        if not path.is_file():
            raise RecordManifestValidationError(f"{location} image is missing: {relative_path}")
        actual_digest = _file_sha256(path)
        if actual_digest != expected_digest:
            raise RecordManifestValidationError(
                f"{location} image sha256 does not match manifest for {relative_path!r}"
            )


def build_run_metadata(
    *,
    record_summary: RecordManifestSummary,
    source_revision: str,
    seed: int,
    image_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    package_versions: Mapping[str, str],
) -> dict[str, object]:
    """Return serializable, non-sensitive metadata for one training run."""
    if not GIT_REVISION_PATTERN.fullmatch(source_revision):
        raise RecordManifestValidationError("source_revision must be a 40-character lowercase Git revision")
    if not isinstance(seed, int) or seed < 0:
        raise RecordManifestValidationError("seed must be a non-negative integer")
    if not isinstance(image_size, int) or image_size < 32:
        raise RecordManifestValidationError("image_size must be an integer of at least 32")
    if not isinstance(epochs, int) or epochs < 1:
        raise RecordManifestValidationError("epochs must be a positive integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise RecordManifestValidationError("batch_size must be a positive integer")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise RecordManifestValidationError("learning_rate must be positive")
    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise RecordManifestValidationError("weight_decay must be non-negative")

    return {
        "schema_version": "1.0",
        "purpose": "fine_tune_only_no_threshold_or_test_reporting",
        "source_revision": source_revision,
        "record_manifest_sha256": record_summary.record_manifest_sha256,
        "record_count": record_summary.record_count,
        "records_per_split": dict(record_summary.records_per_split),
        "labels_per_split": {
            split: list(labels) for split, labels in record_summary.labels_per_split.items()
        },
        "seed": seed,
        "model": {
            "architecture": "mobilenet_v3_small",
            "pretrained_weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
            "output_label": "fully_synthetic",
        },
        "preprocessing": {
            "image_size": image_size,
            "normalization": "ImageNet mean/std via torchvision weight transforms",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "package_versions": dict(sorted(package_versions.items())),
    }


def binary_auroc(scores: Sequence[float], targets: Sequence[int]) -> float:
    """Calculate binary AUROC with average ranks for tied scores.

    This small standard-library implementation keeps the notebook free of a
    scikit-learn version dependency.  It is a model-selection metric only;
    callers must not interpret it as an operating threshold or probability.
    """
    if len(scores) != len(targets) or not scores:
        raise RecordManifestValidationError("AUROC requires equally sized, non-empty scores and targets")
    ranked: list[tuple[float, int]] = []
    for index, (score, target) in enumerate(zip(scores, targets)):
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise RecordManifestValidationError(f"AUROC score at index {index} must be finite")
        if isinstance(target, bool) or target not in (0, 1):
            raise RecordManifestValidationError(f"AUROC target at index {index} must be 0 or 1")
        ranked.append((float(score), target))
    ranked.sort(key=lambda item: item[0])

    positive_count = sum(target for _, target in ranked)
    negative_count = len(ranked) - positive_count
    if not positive_count or not negative_count:
        raise RecordManifestValidationError("AUROC requires both labels")

    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        positive_rank_sum += average_rank * sum(target for _, target in ranked[start:end])
        start = end
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def write_json(path: Path, document: Mapping[str, object]) -> None:
    """Write a canonical JSON audit artifact to an already-approved output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _split_sources(split_document: Mapping[str, object]) -> Mapping[str, set[str]]:
    raw_splits = split_document.get("splits")
    if not isinstance(raw_splits, dict):
        raise RecordManifestValidationError("split manifest.splits must be an object")
    result: dict[str, set[str]] = {}
    for split in SPLITS:
        raw_split = raw_splits.get(split)
        if not isinstance(raw_split, dict):
            raise RecordManifestValidationError(f"split manifest split {split!r} must be an object")
        source_ids = raw_split.get("source_ids")
        if not isinstance(source_ids, list) or not all(isinstance(item, str) for item in source_ids):
            raise RecordManifestValidationError(
                f"split manifest split {split!r}.source_ids must be a string list"
            )
        result[split] = set(source_ids)
    return result


def _validate_relative_path(record: Mapping[str, object], location: str) -> str:
    value = _require_nonempty_string(record, "relative_path", location)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise RecordManifestValidationError(f"{location}.relative_path must be a safe relative path")
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise RecordManifestValidationError(
            f"{location}.relative_path must end in one of {sorted(SUPPORTED_IMAGE_SUFFIXES)}"
        )
    return path.as_posix()


def _validate_camera_record(
    record: Mapping[str, object],
    split: str,
    subject_splits: dict[str, str],
    capture_splits: dict[str, str],
    location: str,
) -> None:
    for key, seen_splits in (
        ("subject_group_id", subject_splits),
        ("capture_group_id", capture_splits),
    ):
        group_id = _require_nonempty_string(record, key, location)
        prior_split = seen_splits.setdefault(group_id, split)
        if prior_split != split:
            raise RecordManifestValidationError(
                f"{location}.{key} {group_id!r} appears in both {prior_split!r} and {split!r}"
            )


def _validate_synthetic_record(
    record: Mapping[str, object], source: Mapping[str, object], digest: str, location: str
) -> None:
    generator = source.get("generator")
    if not isinstance(generator, dict):
        raise RecordManifestValidationError(f"{location} synthetic source has no generator")
    for record_key, source_key in (
        ("generator_model_id", "model_id"),
        ("generator_revision", "revision"),
        ("generator_family", "family"),
    ):
        if record.get(record_key) != generator.get(source_key):
            raise RecordManifestValidationError(
                f"{location}.{record_key} does not match its catalogued generator"
            )
    if _require_sha256(record, "output_sha256", location) != digest:
        raise RecordManifestValidationError(f"{location}.output_sha256 must match {location}.sha256")
    seed = record.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RecordManifestValidationError(f"{location}.seed must be a non-negative integer")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_equal(
    document: Mapping[str, object], key: str, expected: object, location: str
) -> None:
    if document.get(key) != expected:
        raise RecordManifestValidationError(f"{location}.{key} does not match the checked-in manifest")


def _require_nonempty_string(document: Mapping[str, object], key: str, location: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RecordManifestValidationError(f"{location}.{key} must be a non-empty string")
    return value


def _require_record_value(document: Mapping[str, object], key: str, location: str) -> None:
    value = document.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RecordManifestValidationError(f"{location}.{key} is required by its source")


def _require_sha256(document: Mapping[str, object], key: str, location: str) -> str:
    value = _require_nonempty_string(document, key, location)
    if not SHA256_PATTERN.fullmatch(value):
        raise RecordManifestValidationError(f"{location}.{key} must be sha256:<64 lowercase hex>")
    return value


def _require_list(document: Mapping[str, object], key: str, location: str) -> list[object]:
    value = document.get(key)
    if not isinstance(value, list):
        raise RecordManifestValidationError(f"{location}.{key} must be a list")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a private record manifest and print its canonical audit digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-manifest",
        required=True,
        type=Path,
        help="private JSON manifest containing image paths and record-level provenance",
    )
    parser.add_argument(
        "--source-manifest-dir",
        type=Path,
        default=Path(__file__).with_name("manifests"),
        help="directory containing portrait-sources-v1.json and portrait-splits-v1.json",
    )
    args = parser.parse_args(argv)
    try:
        source_document = load_json(args.source_manifest_dir / "portrait-sources-v1.json")
        split_document = load_json(args.source_manifest_dir / "portrait-splits-v1.json")
        summary = validate_record_manifest(
            load_record_manifest(args.record_manifest), source_document, split_document
        )
    except (ManifestValidationError, RecordManifestValidationError) as error:
        print(f"private training record validation failed: {error}", file=sys.stderr)
        return 1
    print(
        "private training records valid: "
        f"{summary.record_count} records; records={summary.records_per_split}; "
        f"canonical={summary.record_manifest_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
