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
