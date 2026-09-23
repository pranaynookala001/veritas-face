# Private held-out robustness benchmark

`training/benchmark.py` evaluates one immutable detector release against the
private **test** records only. It validates the checked-in source and split
manifests plus the private record manifest before it reads a pixel, verifies
each held-out image's SHA-256, and writes a new private JSON result. It never
opens training or validation image files.

This is an evaluation step, not an origin-verification mechanism. Scores and
aggregate metrics are probabilistic detector evidence, not proof that an image
is camera-origin or synthetic. It does not calibrate a detector, select a
threshold, or make a deployment decision.

## Required private records

Start from the validated record manifest described in
[training-data governance](training-data.md). Every record assigned to `test`
must additionally contain one of these annotations:

```json
"pose_category": "frontal"
```

Allowed values are `frontal`, `left_profile`, and `right_profile`. The held-out
set and its combined left/right-profile subgroup must each contain both
`camera_origin` and `fully_synthetic` records. This prevents a seemingly valid
profile result from being a one-class measurement. The profile label is an
operator-reviewed annotation; do not infer it from a detector score.

The shared record validator still enforces that all test records come from the
source-plan's held-out sources, preserve camera identity/capture disjointness,
and preserve text-to-image generator-family disjointness. The test split must
remain separate from calibration data and all model-selection work.

## Fixed condition matrix

Each held-out record is evaluated clean and with the following deterministic
conditions:

| Condition | Fixed v1 treatment |
| --- | --- |
| JPEG | RGB JPEG re-encode at quality 75 with 4:4:4 sampling |
| Resize | Reduce to 50%, then restore original dimensions with Lanczos |
| Crop | Centre crop retaining 80% of each dimension, then restore original dimensions with Lanczos |
| Filters | Color factor 0.5 followed by contrast factor 1.2 |
| Blur | Gaussian blur with radius 2 pixels |
| Occlusion | Grey, centred rectangle covering 25% of image area |
| Profile pose | Clean examples annotated `left_profile` or `right_profile` |

The transformations normalize decoded images to RGB and remove EXIF-dependent
behaviour. `profile_pose` is an annotated subgroup rather than a synthetic
image transform. The runner records the exact parameters in its output. A
change to this matrix requires a versioned runner change and a new result;
results from different conditions must not be presented as interchangeable.

## Scorer adapter and command

The private evaluation environment needs Pillow. Supply an explicit adapter
for the exact release and preprocessing pipeline under evaluation. The adapter
receives a temporary transformed image path, performs the same required face
selection/crop and inference path as the release, and writes **only** this JSON
object to standard output:

```json
{"synthetic_probability": 0.73, "latency_ms": 14.2}
```

`synthetic_probability` must be a finite number from 0 through 1. `latency_ms`
is optional, but if it is reported it must be finite and non-negative for every
sample in that condition. The benchmark runner measures adapter wall time as
well. It invokes the adapter without a shell and requires exactly one
`{image_path}` placeholder, so the adapter command is not a place for
untrusted record metadata or image paths.

For example, with all paths remaining in private storage:

```sh
python3 training/benchmark.py \
  --record-manifest /private/veritas/portrait-records-v1.json \
  --data-root /private/veritas/portraits \
  --scorer-command '/private/veritas/score-release --image {image_path}' \
  --scorer-identity release-2026.09.1-cpp-adapter \
  --detector-id synthetic-portrait-classifier \
  --detector-version release-2026.09.1 \
  --detector-family mobilenetv3-small \
  --source-revision EXACT_40_CHARACTER_GIT_REVISION \
  --output /private/veritas/results/benchmark-release-2026.09.1.json
```

The output path must not already exist. Temporary transformed images are
deleted after scoring; the result contains only digests, aggregate condition
metrics, and detector identifiers—never image paths, record IDs, pixels,
prompts, seeds, or per-record scores. Keep the output access-controlled and
out of Git.

## Reading results

For each condition, the result contains sample counts by label, raw-score
AUROC, mean synthetic probability by label, and adapter-wall plus optional
adapter-reported latency summaries (`mean`, `p50`, `p95`). AUROC needs both
labels and is reported without choosing an operating threshold. The aggregate
file does not establish accuracy for any unmeasured generator, population,
camera, demographic group, pose, transform severity, or deployment setting.

Do not use its held-out test records to fit calibration curves, choose a
threshold, tune the model, or iterate prompt/source selection. Review the
private results, data coverage, failure modes, and latency measurements before
the separate [publication and baseline-comparison workflow](evaluation-publication.md).
