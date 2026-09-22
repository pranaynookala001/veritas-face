# Training data governance

Veritas Face trains and evaluates only on traceable camera-origin portraits and
fully synthetic text-to-image portraits. The repository holds a versioned
source catalog and split plan in `training/manifests/`; it intentionally does
not hold portrait pixels, face crops, prompt text, seeds, model checkpoints, or
materialized record lists.

The catalog records the source URL, licence terms URL, review date, and an
immutable revision for every synthetic generator. Its camera-origin source is
Wikimedia Commons portrait photography, but that collection has mixed rights:
an operator must confirm the individual file displays CC BY 4.0 or CC BY-SA
4.0 and capture the required attribution before acquiring it. This is a
selection rule, not a claim that every Commons image is eligible.

Synthetic records must be generated text-to-image without a reference image,
identity reference, face swap, or image conditioning. The catalog pins Stable
Diffusion v1.5, FLUX.1-schnell, and SDXL to their model revisions and records
their terms. Those terms and upstream availability can change; the operator is
responsible for reviewing current terms before use. Recording a licence does
not establish consent, privacy clearance, fitness for a particular deployment,
or a factual claim about a real person.

Splits are identity- and capture-group-disjoint for real portraits. Synthetic
generator families are disjoint across train, validation, and test, so the
test family is not represented in model fitting. Every materialized record
must retain a source or generation trace and a SHA-256 output digest. A
calibration artifact must point at the canonical digest of its immutable
record-level held-out manifest, not at the source catalog.

Run `python3 training/validate_manifests.py` before materializing sources and
`python3 -m unittest discover -s training/tests` after manifest edits.

## Private fine-tuning runs

[`training/fine_tune_mobilenetv3_kaggle.ipynb`](../training/fine_tune_mobilenetv3_kaggle.ipynb)
is the reproducible free-Kaggle fine-tuning path for a pretrained
MobileNetV3-Small classifier. It has no outputs checked in, makes no benchmark
claim, and must not be used to pick an operating threshold. It only fits on
`train`, selects a checkpoint by validation AUROC, and deliberately never
opens the held-out `test` records.

Before a run, make three immutable Kaggle inputs:

1. A read-only repository snapshot named for the exact Git revision.
2. A private record dataset containing a record manifest and the referenced
   portrait files.
3. A read-only copy of the official
   `MobileNet_V3_Small_Weights.IMAGENET1K_V1` checkpoint.

The notebook requires the canonical SHA-256 of the private record manifest
(as computed by the command below) and a full byte-level SHA-256 for the
pretrained checkpoint, then verifies every image byte before importing
PyTorch. Once the immutable inputs are attached, Internet access can remain
off. It records the source revision, digests, package versions, fixed seed,
preprocessing, optimiser parameters, and selected epoch in private
`run-metadata.json` output.

```sh
python3 training/fine_tune.py --record-manifest /private/path/portrait-records-v1.json
```

The private record manifest is a JSON object with this top-level shape:

```json
{
  "schema_version": "1.0",
  "manifest_version": "private-portrait-records-…",
  "source_manifest_sha256": "sha256:…",
  "split_manifest_sha256": "sha256:…",
  "records": []
}
```

Each record must include `record_id`, a safe image `relative_path`, its image
`sha256`, `label`, `source_id`, and `split`. Camera-origin records must retain
the reviewed source fields—including attribution, licence, subject group, and
capture group. Synthetic records must retain their exact generator model,
revision, family, prompt-template identifier, seed, and `output_sha256`.
`training/fine_tune.py` rejects a label/source mismatch, a synthetic generator
that differs from the catalog, duplicate content, a group that crosses splits,
or a digest mismatch. The manifest, image paths, outputs, state dictionaries,
and checkpoints are private training artifacts and remain excluded from Git.

## Private model releases

After a run selects a checkpoint, use the [private model-release procedure](model-release.md) to create a new, checksummed ONNX release bundle. The exporter binds the output to the completed run metadata and an exact detector-release calibration artifact, produces a model card, and requires an authorized operator to supply the applicable licence text and SPDX identifier. It cannot choose a licence, turn validation AUROC into a threshold, or publish an evaluation claim.

The calibration input must be a separately held-out, licence-reviewed manifest: it must not reuse the examples used for model selection or the later benchmark test split. The exported policy configuration records the API's conservative thresholds and its paired calibration checksum, but missing, invalid, or mismatched calibration still makes a deployed report `inconclusive`.

## Private held-out benchmarking

After release export, evaluate the untouched `test` records with the private
[benchmark runner](benchmarking.md). It verifies record digests and runs a
fixed JPEG, resize, crop, filter, blur, occlusion, and annotated-profile-pose
matrix through the exact release pipeline. It writes only private aggregate
results and cannot calibrate a detector or select a product threshold.
