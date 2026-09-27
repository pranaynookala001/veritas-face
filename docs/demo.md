# Local demo notes

The portfolio images in [`docs/assets`](assets) document the running local web
experience without publishing a portrait, a crop, a model artifact, or a
performance claim.

## Capture conditions

`local-upload.png` is the initial browser state of the local Next.js app at
`http://localhost:3000`.

`local-inconclusive-report.png` is a completed local report after uploading a
generated, single-colour PNG that contains no face. The input file was used
only to drive the local capture and was not committed. The resulting report is
expected to show:

- `Assessment: Inconclusive`;
- a `no_face_detected` explanation;
- local metadata and Content Credentials evidence; and
- skipped detector/calibration evidence.

That scenario is intentionally conservative. It shows that Veritas Face does
not use a missing face, absent Content Credentials, or unavailable detector as
evidence for either origin. It also avoids placing a person's likeness or
synthetic portrait in the repository.

## Reproduce

1. Follow [the local setup guide](setup.md) to start the web app and API.
2. Open `http://localhost:3000` and capture the initial upload state.
3. Create a temporary non-portrait PNG locally (for example, a single-colour
   square) and submit it through the browser.
4. Wait for the completed evidence report and capture the report panel.
5. Delete the temporary input. Do not add it to Git.

The screenshots are UI documentation, not a model evaluation. They do not
demonstrate detector accuracy, cold-start behavior, public-service availability,
quota behavior, or an authenticity verdict. Those claims require the remaining
deployment work and an audited private model/calibration release.
