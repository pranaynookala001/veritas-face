# Portrait source and split manifests

`portrait-sources-v1.json` is the reviewed acquisition catalog for the first
training dataset. It contains no image bytes, subjects, prompts, or generated
outputs. `portrait-splits-v1.json` assigns its sources to train, validation,
and test, while requiring both labels in every split.

The real-image source requires a file-level CC BY 4.0 or CC BY-SA 4.0 review
and attribution record for every acquired image. Its split group is the person
depicted; repeated captures of the same person must remain in one split. The
synthetic sources are text-to-image only: a prompt or reference image of a real
person, image-to-image conditioning, face swapping, and identity references
are excluded. Each output must retain its model revision, family, prompt
template identifier, seed, and SHA-256 digest in private storage.

The split plan is intentionally generator-family-disjoint: Stable Diffusion
v1.5 supplies training synthetics, FLUX.1-schnell supplies validation
synthetics, and SDXL supplies held-out test synthetics. That makes an apparent
test result less likely to be a memorized detector signature from a training
generator. It does not establish performance on all generators.

Validate the catalog from the repository root:

```sh
python3 training/validate_manifests.py
python3 -m unittest discover -s training/tests
```

The validator emits canonical SHA-256 values. Pin the eventual immutable,
record-level validation manifest in calibration artifacts; do not use a source
catalog digest as if it were evidence about individual images.

Raw images, derived face crops, prompt text, seeds, checkpoints, and materialized
record manifests remain in private storage and must not be committed. Before
acquisition or publication, a maintainer must review the current licence terms,
privacy implications, and any relevant consent or jurisdictional requirements.
