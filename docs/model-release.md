# Private model release procedure

Veritas Face does not commit trained weights, checkpoints, private record
manifests, portraits, face crops, prompts, or seeds. A model release is built
from those private inputs only after a fine-tuning run has selected one
checkpoint. The release bundle is then separately reviewed before it is made
available to any deployment.

This procedure applies only to the v1 fully-synthetic-portrait classifier. It
does not license or validate a claim about face swaps, identity deepfakes, or
general image authenticity.

## Required private inputs

Before exporting, retain the following private, immutable inputs:

1. The selected `mobilenetv3-small-state-dict.pt` checkpoint and its completed
   `run-metadata.json` from the fine-tuning notebook. The metadata must contain
   the selected epoch and the checkpoint SHA-256 recorded after training.
2. A calibration JSON document for the exact detector ID, model version, and
   detector family. It must use the API calibration schema and point to the
   canonical digest of a licensed, held-out calibration manifest. Do not reuse
   model-selection validation examples or the later benchmark test split for
   this purpose.
3. The applicable, operator-approved model-distribution licence text and its
   SPDX identifier. The exporter deliberately requires both rather than choosing
   a licence for the maintainer.

The private release environment needs compatible `torch`, `torchvision`, and
`onnx` packages. It needs no network access. The checkpoint is loaded with
PyTorch's `weights_only=True` mode; a checkpoint that does not exactly match
the fixed MobileNetV3-Small binary architecture is rejected.

## Export

From a repository snapshot at the same revision recorded in the run metadata,
run the exporter with private paths. `training/outputs/` is already excluded
from Git; use another access-controlled directory when appropriate.

```sh
python3 training/export_model.py \
  --checkpoint training/outputs/private-run/mobilenetv3-small-state-dict.pt \
  --run-metadata training/outputs/private-run/run-metadata.json \
  --calibration-artifact /private/calibration/release-2026.09.1.json \
  --license-file /private/licenses/model-distribution-LICENSE.txt \
  --license-spdx LICENSE-IDENTIFIER \
  --model-version release-2026.09.1 \
  --detector-id synthetic-portrait-classifier \
  --detector-family mobilenetv3-small \
  --output-dir training/outputs/release-2026.09.1
```

`--output-dir` must not already exist. The command creates the release in a
temporary sibling directory and moves it into place only after ONNX validation
and all metadata writes succeed. It never overwrites a previous release.

## Release contents

The completed directory contains:

- `portrait-classifier.onnx` — a CPU-ready model with one float32
  `normalized_nchw` input shaped `1 × 3 × 224 × 224` and one float32
  `synthetic_probability` output shaped `1 × 1`.
- `calibration.json` — a canonical copy of the exact, monotonic
  release-specific calibration artifact.
- `verdict-policy.json` — the versioned policy configuration, including the
  paired calibration checksum and the conservative `0.85`, `0.15`, and `0.20`
  thresholds enforced by the v1 API policy.
- `model-card.md` — provenance, input/output contract, calibration linkage,
  safety boundary, and limitations. It makes no unmeasured performance claim.
- `LICENSE.txt` — the explicit licence text supplied by the authorized
  operator.
- `release-manifest.json` — SHA-256 values for all five distributable files.

The release manifest does not checksum itself, avoiding a self-referential hash
cycle. Recalculate every listed digest before deployment. Its calibration
artifact must be configured with `VERITAS_FACE_CALIBRATION_PATH`; the API
applies it only when the service reports the exact paired detector ID and
version.

## Policy and evaluation boundary

The packaged verdict policy describes how calibrated evidence is interpreted:
a score at least `0.85` may support `likely_synthetic`, one at most `0.15` may
support `likely_authentic`, and a gap of `0.20` or more across independent
detector families forces `inconclusive`. Face-quality failures, an absent or
mismatched calibration artifact, unavailable inference, and invalid detector
responses always remain inconclusive. These are probabilistic evidence rules,
not a mechanism for verifying origin.

Export is not benchmarking. Keep the held-out benchmark split untouched until
the benchmark runner and publication work are complete. No release should make
accuracy, demographic, generator-coverage, robustness, or latency claims until
that evaluation has been run and reviewed.
