#!/usr/bin/env python3
"""Export one selected private MobileNetV3 checkpoint as an auditable ONNX bundle.

This command never downloads weights or data.  It accepts only an already
selected private checkpoint, its completed fine-tuning metadata, an exact
release calibration artifact, and an operator-supplied licence text.  The
resulting bundle is intentionally written to a new directory so an incomplete
export cannot overwrite a prior release.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from model_release import (
    IMAGE_SIZE,
    INPUT_NAME,
    OUTPUT_NAME,
    ModelReleaseValidationError,
    build_model_card,
    build_release_manifest,
    build_threshold_configuration,
    canonical_json_bytes,
    load_json_object,
    sha256_bytes,
    sha256_file,
    validate_calibration_artifact,
    validate_run_metadata,
)


DEFAULT_DETECTOR_ID = "synthetic-portrait-classifier"
DEFAULT_DETECTOR_FAMILY = "mobilenetv3-small"
ONNX_FILE_NAME = "portrait-classifier.onnx"
CALIBRATION_FILE_NAME = "calibration.json"
THRESHOLD_FILE_NAME = "verdict-policy.json"
MODEL_CARD_FILE_NAME = "model-card.md"
LICENSE_FILE_NAME = "LICENSE.txt"
RELEASE_MANIFEST_FILE_NAME = "release-manifest.json"


def _nonempty(value: str, label: str) -> str:
    if not value.strip():
        raise ModelReleaseValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _load_license(path: Path) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelReleaseValidationError(f"cannot read license file: {error}") from error
    if not content.strip():
        raise ModelReleaseValidationError("license file must not be empty")
    if len(content) > 256 * 1024:
        raise ModelReleaseValidationError("license file exceeds 256 KiB")
    return content


def export_onnx_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    """Load the fixed training architecture and export the C++ service contract."""
    try:
        import torch
        from torch import nn
        from torchvision.models import mobilenet_v3_small
    except ImportError as error:
        raise ModelReleaseValidationError(
            "ONNX export requires torch, torchvision, and onnx in the private release environment"
        ) from error

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ModelReleaseValidationError("selected checkpoint could not be loaded safely") from error
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model_state_dict", "epoch"}:
        raise ModelReleaseValidationError(
            "selected checkpoint must contain only model_state_dict and epoch"
        )
    if not isinstance(checkpoint["epoch"], int) or checkpoint["epoch"] < 1:
        raise ModelReleaseValidationError("selected checkpoint epoch must be a positive integer")
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ModelReleaseValidationError("selected checkpoint model_state_dict must be an object")

    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ModelReleaseValidationError(
            "selected checkpoint does not match the fixed MobileNetV3-Small binary architecture"
        ) from error
    model.eval()

    class ProbabilityModel(nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, normalized_nchw):
            return torch.sigmoid(self.classifier(normalized_nchw))

    try:
        torch.onnx.export(
            ProbabilityModel(model),
            torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32),
            output_path,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=13,
            dynamic_axes=None,
            dynamo=False,
        )
    except (RuntimeError, ValueError, OSError) as error:
        raise ModelReleaseValidationError("PyTorch could not export the selected checkpoint to ONNX") from error
    validate_onnx_contract(output_path)


def validate_onnx_contract(path: Path) -> None:
    """Check the serialized artifact matches the strict C++ inference contract."""
    try:
        import onnx
        from onnx import TensorProto, checker
    except ImportError as error:
        raise ModelReleaseValidationError("ONNX export validation requires the onnx package") from error
    try:
        model = onnx.load_model(path)
        checker.check_model(model)
    except (OSError, ValueError, onnx.checker.ValidationError) as error:
        raise ModelReleaseValidationError("exported ONNX artifact failed ONNX validation") from error
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ModelReleaseValidationError("exported ONNX model must have exactly one input and one output")
    model_input = model.graph.input[0]
    model_output = model.graph.output[0]
    if model_input.name != INPUT_NAME or model_output.name != OUTPUT_NAME:
        raise ModelReleaseValidationError("exported ONNX tensor names do not match the inference contract")
    input_type = model_input.type.tensor_type
    output_type = model_output.type.tensor_type
    if input_type.elem_type != TensorProto.FLOAT or output_type.elem_type != TensorProto.FLOAT:
        raise ModelReleaseValidationError("exported ONNX input and output must be float32")
    input_shape = [dimension.dim_value for dimension in input_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in output_type.shape.dim]
    if input_shape != [1, 3, IMAGE_SIZE, IMAGE_SIZE] or output_shape != [1, 1]:
        raise ModelReleaseValidationError(
            "exported ONNX shapes must be 1x3x224x224 input and 1x1 output"
        )


def create_release_bundle(args: argparse.Namespace) -> Path:
    """Validate private inputs, then atomically create a new complete release bundle."""
    model_version = _nonempty(args.model_version, "model version")
    detector_id = _nonempty(args.detector_id, "detector id")
    detector_family = _nonempty(args.detector_family, "detector family")
    license_spdx = _nonempty(args.license_spdx, "license SPDX identifier")
    output_directory = args.output_dir.resolve()
    if output_directory.exists():
        raise ModelReleaseValidationError(
            f"release output directory already exists: {output_directory}; choose a new directory"
        )

    run = validate_run_metadata(load_json_object(args.run_metadata, "run metadata"))
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if checkpoint_sha256 != run.checkpoint_sha256:
        raise ModelReleaseValidationError("selected checkpoint does not match run metadata.checkpoint_sha256")
    calibration_document = load_json_object(args.calibration_artifact, "calibration artifact")
    calibration = validate_calibration_artifact(
        calibration_document,
        detector_id=detector_id,
        detector_version=model_version,
        detector_family=detector_family,
    )
    license_content = _load_license(args.license_file)
    canonical_calibration = canonical_json_bytes(calibration_document)
    calibration_sha256 = sha256_bytes(canonical_calibration)
    threshold_configuration = build_threshold_configuration(calibration, calibration_sha256)
    threshold_content = canonical_json_bytes(threshold_configuration)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_directory.name}.building-", dir=output_directory.parent
    ) as temporary_directory:
        staging_directory = Path(temporary_directory)
        onnx_path = staging_directory / ONNX_FILE_NAME
        export_onnx_checkpoint(args.checkpoint, onnx_path)
        onnx_sha256 = sha256_file(onnx_path)
        model_card = build_model_card(
            model_version=model_version,
            detector_id=detector_id,
            detector_family=detector_family,
            license_spdx=license_spdx,
            run=run,
            checkpoint_sha256=checkpoint_sha256,
            onnx_sha256=onnx_sha256,
            calibration=calibration,
            calibration_artifact_sha256=calibration_sha256,
        ).encode("utf-8")
        (staging_directory / CALIBRATION_FILE_NAME).write_bytes(canonical_calibration)
        (staging_directory / THRESHOLD_FILE_NAME).write_bytes(threshold_content)
        (staging_directory / MODEL_CARD_FILE_NAME).write_bytes(model_card)
        (staging_directory / LICENSE_FILE_NAME).write_bytes(license_content)
        artifact_digests = {
            name: sha256_file(staging_directory / name)
            for name in (
                CALIBRATION_FILE_NAME,
                LICENSE_FILE_NAME,
                MODEL_CARD_FILE_NAME,
                ONNX_FILE_NAME,
                THRESHOLD_FILE_NAME,
            )
        }
        release_manifest = build_release_manifest(
            model_version=model_version,
            detector_id=detector_id,
            detector_family=detector_family,
            artifacts=artifact_digests,
        )
        (staging_directory / RELEASE_MANIFEST_FILE_NAME).write_bytes(
            canonical_json_bytes(release_manifest)
        )
        staging_directory.rename(output_directory)
    return output_directory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="selected private .pt checkpoint")
    parser.add_argument(
        "--run-metadata",
        required=True,
        type=Path,
        help="completed private run-metadata.json from fine tuning",
    )
    parser.add_argument(
        "--calibration-artifact",
        required=True,
        type=Path,
        help="validated private calibration JSON for this exact detector release",
    )
    parser.add_argument(
        "--license-file",
        required=True,
        type=Path,
        help="operator-approved applicable model-distribution licence text",
    )
    parser.add_argument("--license-spdx", required=True, help="SPDX identifier for --license-file")
    parser.add_argument("--model-version", required=True, help="immutable detector release identifier")
    parser.add_argument(
        "--detector-id",
        default=DEFAULT_DETECTOR_ID,
        help=f"detector identifier (default: {DEFAULT_DETECTOR_ID})",
    )
    parser.add_argument(
        "--detector-family",
        default=DEFAULT_DETECTOR_FAMILY,
        help=f"independent detector-family identifier (default: {DEFAULT_DETECTOR_FAMILY})",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new private directory for the complete ONNX release bundle",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_directory = create_release_bundle(args)
    except ModelReleaseValidationError as error:
        print(f"model release export failed: {error}", file=sys.stderr)
        return 1
    print(f"private model release bundle created: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
