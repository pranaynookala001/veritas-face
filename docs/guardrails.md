# Product guardrails

## Intended use

Veritas Face is a technical demonstration and forensic triage aid. Its verdict estimates whether a supplied portrait resembles a fully synthetic, AI-generated face versus a camera-origin portrait.

## Not evidence of identity or intent

It must not be used to decide whether a person is trustworthy, eligible for employment, enrolled in school, admitted to a service, or responsible for misconduct. It does not identify people and must not persist biometric templates.

## Required report language

- A valid C2PA credential may verify declared provenance.
- Missing, stripped, or invalid provenance does not prove authenticity or synthetic origin.
- Detector scores remain probabilities and must show quality warnings and model version.
- The system must answer `inconclusive` when image conditions or model agreement do not support a meaningful score.

## Explicit exclusions in v1

- Face swaps and identity deepfakes.
- AI retouching or partial edits to a real portrait.
- Video, live camera feeds, and multi-person scenes.
- Any demographic, identity, emotion, or intent inference.
