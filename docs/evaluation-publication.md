# Aggregate evaluation publication

No evaluation results are checked into this repository. A public metric claim
must be produced only after an authorized reviewer completes the private,
held-out workflow described in [benchmarking.md](benchmarking.md). Never copy
portrait pixels, crops, record manifests, image paths, record IDs, prompts,
seeds, checkpoints, or per-record scores into a publication directory or Git.

`training/publish_evaluation.py` is a narrow publication gate. It accepts two
private aggregate benchmark outputs—one candidate release and one independent
baseline—plus the exact candidate calibration artifact and a reviewed,
aggregate-only error-analysis document. It writes a new output directory with:

- `evaluation.md` — metrics, comparison table, reviewed findings, and limits;
- `calibration-curve.svg` — the release's raw-to-calibrated score mapping;
- `condition-comparison.svg` — candidate and baseline AUROC by fixed condition;
- `latency-comparison.svg` — comparable candidate and baseline p95 latency;
- `evaluation-manifest.json` — input and artifact SHA-256 links.

The tool fails closed unless both reports use the same source revision, the
same four immutable benchmark-input digests, the same fixed condition matrix,
and matching label counts. The candidate calibration must match its exact
detector ID, version, and family. It never calibrates the baseline or turns an
AUROC into a verdict threshold.

## Required private inputs

Run the held-out benchmark once per immutable detector release, retaining each
aggregate JSON result in access-controlled storage. Use the exact same test
record manifest, data root, runner revision, fixed conditions, and adapter
environment for candidate and baseline. Use the candidate's separately held-
out calibration artifact; benchmark test records must not be used for
calibration.

Before generating anything public, create a small error-analysis JSON file.
Its input checksums bind reviewer conclusions to the precise two benchmark
outputs. It must contain at least one review finding and one additional
limitation:

```json
{
  "schema_version": "1.0",
  "candidate_benchmark_sha256": "sha256:…",
  "baseline_benchmark_sha256": "sha256:…",
  "review_scope": "private_record_level_review_without_identifiers",
  "findings": [
    {
      "condition_id": "gaussian_blur_2",
      "category": "robustness_degradation",
      "affected_count": 0,
      "summary": "Review found no recurring error pattern in this condition."
    }
  ],
  "additional_limitations": [
    "The reviewed dataset does not establish demographic fairness."
  ]
}
```

Allowed review categories are `false_positive_pattern`,
`false_negative_pattern`, `uncertainty_pattern`, `robustness_degradation`, and
`coverage_gap`. `affected_count` is a reviewer-reported triage count, not a
new detector metric. Findings must stay aggregate and omit image or record
identifiers.

## Generate and review the bundle

Keep all input and output paths outside the repository until the resulting
aggregates have been reviewed for correctness, privacy, licensing, and product
claims:

```sh
python3 training/publish_evaluation.py \
  --candidate-benchmark /private/veritas/results/candidate-2026.09.1.json \
  --baseline-benchmark /private/veritas/results/baseline-2026.09.1.json \
  --candidate-calibration /private/veritas/calibration/candidate-2026.09.1.json \
  --error-analysis /private/veritas/reviews/candidate-2026.09.1-errors.json \
  --output-dir /private/veritas/public-review/candidate-2026.09.1
```

The output directory must not already exist; this prevents a later run from
silently replacing reviewed artifacts. Inspect `evaluation.md`, render the SVG
charts, and independently recompute every digest in
`evaluation-manifest.json`. A reviewer must then decide whether the claims are
appropriately scoped before selectively adding approved public artifacts to a
documentation change.

The generated report explicitly states that scores are probabilistic evidence,
not proof of origin. Its calibration graphic is a score transformation, not a
claim of calibration quality on the held-out benchmark. The report also
retains the v1 boundary: still portraits with one dominant face and fully
synthetic text-to-image detection only, not face swaps or general image
authenticity.
