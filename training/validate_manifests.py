"""Validate the versioned, license-aware training source and split manifests.

The repository deliberately contains no portraits, crops, generated outputs, or
model checkpoints.  These manifests are the reviewable acquisition contract:
they pin sources and generator revisions, state the permitted data path, and
prevent label, identity, capture, and generator-family leakage before assets
are placed in private storage.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse


SCHEMA_VERSION = "1.0"
SPLITS = frozenset({"train", "validation", "test"})
LABELS = frozenset({"camera_origin", "fully_synthetic"})
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ManifestValidationError(ValueError):
    """Raised when a manifest cannot support a traceable evaluation."""


@dataclass(frozen=True)
class ManifestSummary:
    source_count: int
    source_sha256: str
    split_sha256: str
    synthetic_families: Mapping[str, str]


def canonical_sha256(document: object) -> str:
    """Return a stable digest suitable for a later calibration audit link."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{sha256(encoded.encode('utf-8')).hexdigest()}"


def load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ManifestValidationError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ManifestValidationError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{path} must contain a JSON object")
    return value


def validate_manifests(
    source_document: Mapping[str, object], split_document: Mapping[str, object]
) -> ManifestSummary:
    """Validate source provenance, licensed use, and leakage-resistant splits."""
    _require_version(source_document, "source manifest")
    _require_version(split_document, "split manifest")
    sources = _require_list(source_document, "sources", "source manifest")
    if not sources:
        raise ManifestValidationError("source manifest must contain at least one source")

    sources_by_id: dict[str, Mapping[str, object]] = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ManifestValidationError(f"source manifest sources[{index}] must be an object")
        source_id = _require_string(source, "id", f"sources[{index}]")
        if source_id in sources_by_id:
            raise ManifestValidationError(f"source manifest repeats source id {source_id!r}")
        _validate_source(source, f"sources[{index}]")
        sources_by_id[source_id] = source

    split_map = _require_mapping(split_document, "splits", "split manifest")
    if set(split_map) != SPLITS:
        raise ManifestValidationError("split manifest must define exactly train, validation, and test")
    constraints = _require_mapping(split_document, "constraints", "split manifest")
    _validate_constraints(constraints)

    labels_by_split: dict[str, set[str]] = {}
    synthetic_families: dict[str, str] = {}
    for split_name in sorted(SPLITS):
        split = split_map[split_name]
        if not isinstance(split, dict):
            raise ManifestValidationError(f"split {split_name!r} must be an object")
        _require_string(split, "purpose", f"split {split_name!r}")
        source_ids = _require_list(split, "source_ids", f"split {split_name!r}")
        if not source_ids:
            raise ManifestValidationError(f"split {split_name!r} must name at least one source")
        if len(source_ids) != len(set(source_ids)):
            raise ManifestValidationError(f"split {split_name!r} repeats a source id")

        labels: set[str] = set()
        for source_id in source_ids:
            if not isinstance(source_id, str) or source_id not in sources_by_id:
                raise ManifestValidationError(
                    f"split {split_name!r} references an unknown source {source_id!r}"
                )
            source = sources_by_id[source_id]
            label = source["label"]
            assert isinstance(label, str)
            labels.add(label)
            if label == "fully_synthetic":
                generator = source["generator"]
                assert isinstance(generator, dict)
                family = generator["family"]
                assert isinstance(family, str)
                previous_split = synthetic_families.setdefault(family, split_name)
                if previous_split != split_name:
                    raise ManifestValidationError(
                        f"synthetic generator family {family!r} appears in both "
                        f"{previous_split!r} and {split_name!r}"
                    )
        labels_by_split[split_name] = labels

    required_labels = constraints["required_labels_per_split"]
    assert isinstance(required_labels, list)
    for split_name, labels in labels_by_split.items():
        missing = set(required_labels) - labels
        if missing:
            raise ManifestValidationError(
                f"split {split_name!r} is missing required labels: {', '.join(sorted(missing))}"
            )

    return ManifestSummary(
        source_count=len(sources_by_id),
        source_sha256=canonical_sha256(source_document),
        split_sha256=canonical_sha256(split_document),
        synthetic_families=dict(sorted(synthetic_families.items())),
    )


def _validate_source(source: Mapping[str, object], location: str) -> None:
    label = _require_string(source, "label", location)
    if label not in LABELS:
        raise ManifestValidationError(f"{location}.label must be one of {sorted(LABELS)}")
    asset_origin = _require_string(source, "asset_origin", location)
    _require_url(source, "catalog_url", location)
    license_data = _require_mapping(source, "license", location)
    _validate_license(license_data, f"{location}.license")
    record_requirements = _require_list(source, "record_requirements", location)
    if not record_requirements or not all(isinstance(item, str) and item for item in record_requirements):
        raise ManifestValidationError(f"{location}.record_requirements must be non-empty strings")

    if label == "camera_origin":
        if asset_origin != "public_photograph":
            raise ManifestValidationError(f"{location} camera-origin source must be a public photograph")
        _require_string(source, "split_group", location)
        _require_list(source, "selection_constraints", location)
        if "kind" not in license_data or "allowed_spdx" not in license_data:
            raise ManifestValidationError(
                f"{location}.license must record per-asset licence verification"
            )
    else:
        if asset_origin != "text_to_image_generation":
            raise ManifestValidationError(f"{location} synthetic source must be text-to-image generation")
        generator = _require_mapping(source, "generator", location)
        _validate_generator(generator, f"{location}.generator")
        if "spdx" not in license_data:
            raise ManifestValidationError(f"{location}.license must identify a generator licence")


def _validate_license(license_data: Mapping[str, object], location: str) -> None:
    _require_url(license_data, "terms_url", location)
    reviewed_on = _require_string(license_data, "reviewed_on", location)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", reviewed_on):
        raise ManifestValidationError(f"{location}.reviewed_on must use YYYY-MM-DD")
    if "spdx" in license_data and not isinstance(license_data["spdx"], str):
        raise ManifestValidationError(f"{location}.spdx must be a string")
    if "allowed_spdx" in license_data:
        allowed = license_data["allowed_spdx"]
        if not isinstance(allowed, list) or not allowed or not all(isinstance(item, str) for item in allowed):
            raise ManifestValidationError(f"{location}.allowed_spdx must be a non-empty string list")


def _validate_generator(generator: Mapping[str, object], location: str) -> None:
    _require_string(generator, "family", location)
    _require_string(generator, "model_id", location)
    revision = _require_string(generator, "revision", location)
    if not REVISION_PATTERN.fullmatch(revision):
        raise ManifestValidationError(f"{location}.revision must be a 40-character lowercase Git revision")
    if generator.get("mode") != "text_to_image":
        raise ManifestValidationError(f"{location}.mode must be text_to_image")
    for key in ("identity_reference_allowed", "image_conditioning_allowed"):
        if generator.get(key) is not False:
            raise ManifestValidationError(f"{location}.{key} must be false")
    for key in ("prompt_log_required", "seed_log_required"):
        if generator.get(key) is not True:
            raise ManifestValidationError(f"{location}.{key} must be true")


def _validate_constraints(constraints: Mapping[str, object]) -> None:
    labels = constraints.get("required_labels_per_split")
    if not isinstance(labels, list) or set(labels) != LABELS:
        raise ManifestValidationError(
            "constraints.required_labels_per_split must contain camera_origin and fully_synthetic"
        )
    for key in (
        "identity_groups_must_be_disjoint",
        "capture_groups_must_be_disjoint",
        "synthetic_generator_families_must_be_split_disjoint",
        "synthetic_records_must_be_text_to_image_only",
    ):
        if constraints.get(key) is not True:
            raise ManifestValidationError(f"constraints.{key} must be true")
    if constraints.get("raw_artifacts_committed_to_git") is not False:
        raise ManifestValidationError("constraints.raw_artifacts_committed_to_git must be false")


def _require_version(document: Mapping[str, object], description: str) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ManifestValidationError(f"{description} schema_version must be {SCHEMA_VERSION}")
    _require_string(document, "manifest_version", description)


def _require_string(document: Mapping[str, object], key: str, location: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{location}.{key} must be a non-empty string")
    return value


def _require_list(document: Mapping[str, object], key: str, location: str) -> list[object]:
    value = document.get(key)
    if not isinstance(value, list):
        raise ManifestValidationError(f"{location}.{key} must be a list")
    return value


def _require_mapping(document: Mapping[str, object], key: str, location: str) -> Mapping[str, object]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{location}.{key} must be an object")
    return value


def _require_url(document: Mapping[str, object], key: str, location: str) -> str:
    value = _require_string(document, key, location)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ManifestValidationError(f"{location}.{key} must be an https URL")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(__file__).with_name("manifests"),
        help="directory containing portrait-sources-v1.json and portrait-splits-v1.json",
    )
    args = parser.parse_args(argv)
    source_path = args.manifest_dir / "portrait-sources-v1.json"
    split_path = args.manifest_dir / "portrait-splits-v1.json"
    try:
        summary = validate_manifests(load_json(source_path), load_json(split_path))
    except ManifestValidationError as error:
        print(f"training manifest validation failed: {error}", file=sys.stderr)
        return 1

    families = ", ".join(
        f"{family} ({split})" for family, split in summary.synthetic_families.items()
    )
    print(
        "training manifests valid: "
        f"{summary.source_count} sources; held-out generator families: {families}; "
        f"source={summary.source_sha256}; splits={summary.split_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
