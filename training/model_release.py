"""Pure, auditable helpers for private Veritas Face model-release bundles.

The repository deliberately has no trained weights, record-level manifests, or
model releases.  These helpers validate the non-image metadata that may safely
travel with a private release bundle.  ``export_model.py`` performs the
PyTorch/ONNX work after these checks succeed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Mapping


RELEASE_SCHEMA_VERSION = "1.0"
THRESHOLD_CONFIGURATION_SCHEMA_VERSION = "1.0"
VERDICT_POLICY_VERSION = "veritas-face-v1"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

INPUT_NAME = "normalized_nchw"
OUTPUT_NAME = "synthetic_probability"
IMAGE_SIZE = 224
SYNTHETIC_VERDICT_THRESHOLD = 0.85
AUTHENTIC_VERDICT_THRESHOLD = 0.15
MEANINGFUL_DISAGREEMENT = 0.20


class ModelReleaseValidationError(ValueError):
    """Raised when a private release would not have an auditable contract."""


@dataclass(frozen=True)
class RunProvenance:
    """The non-sensitive identity facts required from one completed fine-tune."""

    source_revision: str
    record_manifest_sha256: str
    checkpoint_sha256: str
    selected_epoch: int
    pretrained_weights_sha256: str
    architecture: str
    output_label: str
    image_size: int


@dataclass(frozen=True)
class CalibrationProvenance:
    """The exact calibration release that may accompany one detector release."""

    calibration_version: str
    validation_manifest_sha256: str
    detector_id: str
    detector_version: str
    detector_family: str


def canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    """Return stable UTF-8 bytes used for generated JSON and release checksums."""
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    """Return the repository's explicit SHA-256 identifier form."""
    return f"sha256:{sha256(content).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash a private artifact in bounded chunks without loading it all at once."""
    digest = sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ModelReleaseValidationError(f"cannot read {path}: {error}") from error
    return f"sha256:{digest.hexdigest()}"


def load_json_object(path: Path, label: str) -> Mapping[str, object]:
    """Load a small JSON object without accepting ambiguous top-level shapes."""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelReleaseValidationError(f"cannot read {label}: {error}") from error
    if len(content) > 256 * 1024:
        raise ModelReleaseValidationError(f"{label} exceeds 256 KiB")
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelReleaseValidationError(f"{label} is not valid JSON") from error
    return _mapping(document, label)


def validate_run_metadata(document: Mapping[str, object]) -> RunProvenance:
    """Require the completed notebook's selected checkpoint and fixed contract."""
    if _string(document.get("schema_version"), "run metadata.schema_version") != "1.0":
        raise ModelReleaseValidationError("unsupported run metadata schema version")
    if document.get("purpose") != "fine_tune_only_no_threshold_or_test_reporting":
        raise ModelReleaseValidationError("run metadata has an unsupported training purpose")

    model = _mapping(document.get("model"), "run metadata.model")
    preprocessing = _mapping(document.get("preprocessing"), "run metadata.preprocessing")
    architecture = _string(model.get("architecture"), "run metadata.model.architecture")
    output_label = _string(model.get("output_label"), "run metadata.model.output_label")
    image_size = _integer(preprocessing.get("image_size"), "run metadata.preprocessing.image_size")
    if architecture != "mobilenet_v3_small":
        raise ModelReleaseValidationError("run metadata must identify mobilenet_v3_small")
    if output_label != "fully_synthetic":
        raise ModelReleaseValidationError("run metadata output label must be fully_synthetic")
    if image_size != IMAGE_SIZE:
        raise ModelReleaseValidationError(f"run metadata image size must be {IMAGE_SIZE}")
    if preprocessing.get("normalization") != "ImageNet mean/std via torchvision weight transforms":
        raise ModelReleaseValidationError("run metadata has an incompatible normalization contract")

    source_revision = _string(document.get("source_revision"), "run metadata.source_revision")
    if not GIT_REVISION_PATTERN.fullmatch(source_revision):
        raise ModelReleaseValidationError("run metadata.source_revision must be a lowercase Git revision")
    selected_epoch = _integer(document.get("selected_epoch"), "run metadata.selected_epoch")
    if selected_epoch < 1:
        raise ModelReleaseValidationError("run metadata.selected_epoch must be positive")
    validation_auroc = document.get("validation_model_selection_auroc")
    if not _finite_number(validation_auroc) or not 0 <= float(validation_auroc) <= 1:
        raise ModelReleaseValidationError(
            "run metadata.validation_model_selection_auroc must be a probability"
        )

    return RunProvenance(
        source_revision=source_revision,
        record_manifest_sha256=_sha256(
            document.get("record_manifest_sha256"), "run metadata.record_manifest_sha256"
        ),
        checkpoint_sha256=_sha256(
            document.get("checkpoint_sha256"), "run metadata.checkpoint_sha256"
        ),
        selected_epoch=selected_epoch,
        pretrained_weights_sha256=_sha256(
            document.get("pretrained_weights_sha256"), "run metadata.pretrained_weights_sha256"
        ),
        architecture=architecture,
        output_label=output_label,
        image_size=image_size,
    )


def validate_calibration_artifact(
    document: Mapping[str, object],
    *,
    detector_id: str,
    detector_version: str,
    detector_family: str,
) -> CalibrationProvenance:
    """Validate the API calibration schema and require an exact release match."""
    _exact_keys(
        document,
        {"schema_version", "calibration_version", "validation_manifest_sha256", "detectors"},
        "calibration artifact",
    )
    if _string(document.get("schema_version"), "calibration artifact.schema_version") != "1.0":
        raise ModelReleaseValidationError("unsupported calibration artifact schema version")
    calibration_version = _string(
        document.get("calibration_version"), "calibration artifact.calibration_version"
    )
    validation_manifest_sha256 = _sha256(
        document.get("validation_manifest_sha256"),
        "calibration artifact.validation_manifest_sha256",
    )
    detectors = document.get("detectors")
    if not isinstance(detectors, list) or not detectors:
        raise ModelReleaseValidationError("calibration artifact.detectors must be a non-empty list")

    release_matches: list[Mapping[str, object]] = []
    seen_releases: set[tuple[str, str]] = set()
    for index, raw_detector in enumerate(detectors):
        location = f"calibration artifact.detectors[{index}]"
        detector = _mapping(raw_detector, location)
        _exact_keys(detector, {"id", "version", "family", "points"}, location)
        current_id = _string(detector.get("id"), f"{location}.id")
        current_version = _string(detector.get("version"), f"{location}.version")
        current_family = _string(detector.get("family"), f"{location}.family")
        release_key = (current_id, current_version)
        if release_key in seen_releases:
            raise ModelReleaseValidationError("calibration artifact repeats a detector release")
        seen_releases.add(release_key)
        _validate_calibration_points(detector.get("points"), f"{location}.points")
        if release_key == (detector_id, detector_version):
            if current_family != detector_family:
                raise ModelReleaseValidationError(
                    "calibration artifact detector family does not match the release"
                )
            release_matches.append(detector)

    if len(release_matches) != 1:
        raise ModelReleaseValidationError(
            "calibration artifact must contain exactly one entry for the exported detector release"
        )
    return CalibrationProvenance(
        calibration_version=calibration_version,
        validation_manifest_sha256=validation_manifest_sha256,
        detector_id=detector_id,
        detector_version=detector_version,
        detector_family=detector_family,
    )


def build_threshold_configuration(
    calibration: CalibrationProvenance,
    calibration_artifact_sha256: str,
) -> dict[str, object]:
    """Build the immutable configuration for the API's conservative verdict policy."""
    _sha256(calibration_artifact_sha256, "calibration_artifact_sha256")
    return {
        "schema_version": THRESHOLD_CONFIGURATION_SCHEMA_VERSION,
        "policy_version": VERDICT_POLICY_VERSION,
        "detector": {
            "id": calibration.detector_id,
            "version": calibration.detector_version,
            "family": calibration.detector_family,
        },
        "calibration": {
            "version": calibration.calibration_version,
            "artifact_sha256": calibration_artifact_sha256,
            "validation_manifest_sha256": calibration.validation_manifest_sha256,
        },
        "thresholds": {
            "synthetic_probability_at_least": SYNTHETIC_VERDICT_THRESHOLD,
            "authentic_probability_at_most": AUTHENTIC_VERDICT_THRESHOLD,
            "independent_family_disagreement_at_least": MEANINGFUL_DISAGREEMENT,
        },
        "scope": "fully_synthetic_portraits_only",
    }


def build_model_card(
    *,
    model_version: str,
    detector_id: str,
    detector_family: str,
    license_spdx: str,
    run: RunProvenance,
    checkpoint_sha256: str,
    onnx_sha256: str,
    calibration: CalibrationProvenance,
    calibration_artifact_sha256: str,
) -> str:
    """Render a factual model card without inventing performance claims."""
    for label, value in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("onnx_sha256", onnx_sha256),
        ("calibration_artifact_sha256", calibration_artifact_sha256),
    ):
        _sha256(value, label)
    return f"""# Veritas Face model card\n\n+## Model identity\n\n+- Detector: `{detector_id}`\n+- Release: `{model_version}`\n+- Detector family: `{detector_family}`\n+- Architecture: `{run.architecture}` with a one-logit sigmoid head\n+- Output: `synthetic_probability` is probabilistic evidence for the `fully_synthetic` label; it is not proof of origin.\n+- Licence: `{license_spdx}`; the exact applicable licence text is packaged as `LICENSE.txt`.\n\n+## Input and runtime contract\n\n+The ONNX file accepts one float32 ImageNet-normalized RGB tensor named `{INPUT_NAME}` with shape `1 × 3 × {run.image_size} × {run.image_size}`. It returns one float32 probability tensor named `{OUTPUT_NAME}` with shape `1 × 1`. The Veritas Face CPU service validates this exact contract before serving it.\n\n+## Training provenance and reproducibility\n\n+- Fine-tuning source revision: `{run.source_revision}`\n+- Private record-manifest digest: `{run.record_manifest_sha256}`\n+- Pinned pretrained-weights digest: `{run.pretrained_weights_sha256}`\n+- Selected checkpoint epoch: `{run.selected_epoch}`\n+- Selected-checkpoint digest: `{checkpoint_sha256}`\n+- ONNX artifact digest: `{onnx_sha256}`\n\n+The private record manifest, portrait pixels, face crops, prompts, seeds, and checkpoint are not included in this bundle or this repository. The training source catalog and split rules are documented in `docs/training-data.md`.\n\n+## Calibration and verdict policy\n\n+- Calibration release: `{calibration.calibration_version}`\n+- Held-out calibration-manifest digest: `{calibration.validation_manifest_sha256}`\n+- Calibration artifact digest: `{calibration_artifact_sha256}`\n+- Exact detector release calibrated: `{calibration.detector_id}@{calibration.detector_version}`\n\n+`calibration.json` contains the release-specific monotonic probability mapping. `verdict-policy.json` records the conservative application thresholds: synthetic probability at least `0.85`, authentic probability at most `0.15`, and independent-family disagreement at least `0.20` forces `inconclusive`. These values apply only after face-quality gates pass and exact-release calibration succeeds.\n\n+## Evaluation status and limitations\n\n+This export does not publish benchmark metrics or establish accuracy for any population, generator, camera, demographic group, or transformation. Benchmarking, calibration plots, error analysis, and latency comparison are separate milestones. The model is limited to still images with one dominant human face and fully synthetic portraits; it does not claim to detect face swaps, identity deepfakes, or general image authenticity. Missing metadata or provenance is never evidence of authenticity.\n\n+## Release integrity\n\n+`release-manifest.json` lists SHA-256 values for every distributable file in this bundle. Verify those checksums before deployment, and configure the API only with the exact paired `calibration.json`.\n"""


def build_release_manifest(
    *,
    model_version: str,
    detector_id: str,
    detector_family: str,
    artifacts: Mapping[str, str],
) -> dict[str, object]:
    """Describe all bundle artifacts except this manifest itself, avoiding a hash cycle."""
    expected_files = {
        "calibration.json",
        "LICENSE.txt",
        "model-card.md",
        "portrait-classifier.onnx",
        "verdict-policy.json",
    }
    if set(artifacts) != expected_files:
        raise ModelReleaseValidationError("release manifest must checksum every required artifact")
    for name, digest in artifacts.items():
        _sha256(digest, f"artifact checksum for {name}")
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "model": {
            "id": detector_id,
            "version": model_version,
            "family": detector_family,
            "format": "onnx",
            "input_name": INPUT_NAME,
            "output_name": OUTPUT_NAME,
        },
        "artifacts": dict(sorted(artifacts.items())),
    }


def _validate_calibration_points(value: object, location: str) -> None:
    if not isinstance(value, list) or len(value) < 2:
        raise ModelReleaseValidationError(f"{location} must contain at least two entries")
    points: list[tuple[float, float]] = []
    for index, raw_point in enumerate(value):
        point_location = f"{location}[{index}]"
        point = _mapping(raw_point, point_location)
        _exact_keys(point, {"raw_probability", "calibrated_probability"}, point_location)
        raw = point.get("raw_probability")
        calibrated = point.get("calibrated_probability")
        if not _finite_number(raw) or not 0 <= float(raw) <= 1:
            raise ModelReleaseValidationError(f"{point_location}.raw_probability must be a probability")
        if not _finite_number(calibrated) or not 0 <= float(calibrated) <= 1:
            raise ModelReleaseValidationError(
                f"{point_location}.calibrated_probability must be a probability"
            )
        points.append((float(raw), float(calibrated)))
    if points[0][0] != 0 or points[-1][0] != 1:
        raise ModelReleaseValidationError(f"{location} must start at 0 and end at 1")
    for lower, upper in zip(points, points[1:]):
        if lower[0] >= upper[0]:
            raise ModelReleaseValidationError(f"{location} raw probabilities must strictly increase")
        if lower[1] > upper[1]:
            raise ModelReleaseValidationError(f"{location} calibrated probabilities must not decrease")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ModelReleaseValidationError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ModelReleaseValidationError(f"{label} has unexpected or missing fields")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelReleaseValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelReleaseValidationError(f"{label} must be an integer")
    return value


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if not SHA256_PATTERN.fullmatch(digest):
        raise ModelReleaseValidationError(f"{label} must be sha256:<64 lowercase hex>")
    return digest
