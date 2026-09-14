# Autonomous development rules

This repository is developed through reviewable, incremental work. Treat `TODO.md` as the source of truth for feature priority and the README as the public product contract.

## Every scheduled run

1. Inspect `git status`, `TODO.md`, recent commits, and CI results.
2. Choose the highest-priority unchecked task that can be finished coherently in one run.
3. Make the smallest complete implementation, including tests and documentation where behavior changes.
4. Run the checks relevant to modified components plus the root quality gate.
5. Commit and push only if all required checks pass and the commit is a real feature, tested bug fix, test suite, or documentation milestone.
6. Mark the completed task in `TODO.md`, with a concise verification note.

## Git rules

- Work only on `codex/dev`; never commit, push, or merge directly to `main`.
- Never force-push, rewrite history, bypass failing checks, or create filler commits.
- Never commit secrets, credentials, private datasets, raw uploaded images, generated face crops, or model checkpoints not cleared for redistribution.
- Use Conventional Commit messages, for example `feat(api): add report schema validation`.
- Before pushing, require a clean test result and inspect the diff for accidental files.

## Product rules

- Present scores as probabilistic evidence, not proof.
- Missing metadata/provenance is never evidence that a portrait is authentic.
- Return `inconclusive` for low-quality images, ambiguous multiple faces, missing face, unsupported media, or meaningful detector disagreement.
- Keep version and calibration information in each report.
- Preserve the explicit v1 boundary: fully synthetic portraits only, not face swaps or general image authenticity.

## Stop and report instead of guessing

Stop without a commit and ask for direction if a task requires paid services, credentials, a new external account, a proprietary dataset, a consequential product-policy decision, or three unsuccessful attempts to fix the same failing check. Stop all scheduled work when every acceptance item in `TODO.md` is complete.
