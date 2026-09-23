#!/usr/bin/env python3
"""Build a reviewable public evaluation bundle from private aggregate results.

The benchmark runner writes private, aggregate-only JSON.  This module is the
separate publication gate: it compares a candidate release with an independent
baseline on precisely the same held-out inputs, binds the candidate to its
exact calibration artifact, and requires a short human-reviewed error analysis.
It produces a Markdown report and SVG charts only; it never reads portrait
pixels, record manifests, image paths, prompts, seeds, or per-record scores.

Publication remains an operator decision.  The generated bundle deliberately
states its coverage and limitations instead of making an origin-verification
claim from detector metrics.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import html
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

from benchmark import BENCHMARK_SCHEMA_VERSION, RUNNER_VERSION, standard_conditions
from fine_tune import LABEL_TO_TARGET
from model_release import (
    GIT_REVISION_PATTERN,
    ModelReleaseValidationError,
    load_json_object,
    validate_calibration_artifact,
)


PUBLICATION_SCHEMA_VERSION = "1.0"
ERROR_ANALYSIS_SCHEMA_VERSION = "1.0"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ERROR_CATEGORIES = frozenset(
    {
        "false_positive_pattern",
        "false_negative_pattern",
        "uncertainty_pattern",
        "robustness_degradation",
        "coverage_gap",
    }
)


class EvaluationPublicationError(ValueError):
    """Raised when an evaluation bundle would be ambiguous or unsafe to publish."""


@dataclass(frozen=True)
class LatencySummary:
    count: int
    mean: float
    p50: float
    p95: float


@dataclass(frozen=True)
class ConditionMetrics:
    identifier: str
    kind: str
    parameters: Mapping[str, object]
    sample_count: int
    labels: Mapping[str, int]
    auroc: float
    mean_synthetic_probability: Mapping[str, float]
    scorer_wall: LatencySummary
    detector_reported: LatencySummary | None


@dataclass(frozen=True)
class BenchmarkReport:
    detector_id: str
    detector_version: str
    detector_family: str
    source_revision: str
    scorer_identity: str
    inputs: Mapping[str, str]
    conditions: tuple[ConditionMetrics, ...]

    @property
    def detector_label(self) -> str:
        return f"{self.detector_id}@{self.detector_version}"


@dataclass(frozen=True)
class ErrorFinding:
    condition_id: str
    category: str
    affected_count: int
    summary: str


@dataclass(frozen=True)
class ErrorAnalysis:
    findings: tuple[ErrorFinding, ...]
    additional_limitations: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationPublication:
    candidate: BenchmarkReport
    baseline: BenchmarkReport
    calibration_version: str
    calibration_validation_manifest_sha256: str
    calibration_points: tuple[tuple[float, float], ...]
    error_analysis: ErrorAnalysis
    candidate_benchmark_sha256: str
    baseline_benchmark_sha256: str
    calibration_artifact_sha256: str
    error_analysis_sha256: str


def load_benchmark_report(path: Path) -> BenchmarkReport:
    """Load and strictly validate one private aggregate benchmark report."""
    document = _load_bounded_mapping(path, "benchmark report")
    expected_keys = {
        "schema_version",
        "purpose",
        "runner",
        "source_revision",
        "detector",
        "scorer_identity",
        "inputs",
        "conditions",
        "limitations",
    }
    _exact_keys(document, expected_keys, "benchmark report")
    if document["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        raise EvaluationPublicationError("unsupported benchmark report schema version")
    if document["purpose"] != "held_out_robustness_evaluation_not_calibration_or_deployment":
        raise EvaluationPublicationError("benchmark report has an unsupported purpose")
    runner = _mapping(document["runner"], "benchmark report.runner")
    _exact_keys(runner, {"name", "version"}, "benchmark report.runner")
    if runner != {"name": "veritas-face-benchmark", "version": RUNNER_VERSION}:
        raise EvaluationPublicationError("benchmark report runner does not match this publication gate")

    source_revision = _string(document["source_revision"], "benchmark report.source_revision")
    if not GIT_REVISION_PATTERN.fullmatch(source_revision):
        raise EvaluationPublicationError("benchmark report.source_revision must be a lowercase Git revision")
    detector = _mapping(document["detector"], "benchmark report.detector")
    _exact_keys(detector, {"id", "version", "family"}, "benchmark report.detector")
    inputs = _digest_mapping(
        document["inputs"],
        {
            "record_manifest_sha256",
            "heldout_test_records_sha256",
            "source_manifest_sha256",
            "split_manifest_sha256",
        },
        "benchmark report.inputs",
    )
    conditions = _parse_conditions(document["conditions"])
    limitations = document["limitations"]
    if not isinstance(limitations, list) or not all(isinstance(item, str) and item.strip() for item in limitations):
        raise EvaluationPublicationError("benchmark report.limitations must be a non-empty string list")

    return BenchmarkReport(
        detector_id=_string(detector["id"], "benchmark report.detector.id"),
        detector_version=_string(detector["version"], "benchmark report.detector.version"),
        detector_family=_string(detector["family"], "benchmark report.detector.family"),
        source_revision=source_revision,
        scorer_identity=_string(document["scorer_identity"], "benchmark report.scorer_identity"),
        inputs=inputs,
        conditions=conditions,
    )


def load_error_analysis(
    path: Path,
    *,
    candidate: BenchmarkReport,
    candidate_benchmark_sha256: str,
    baseline_benchmark_sha256: str,
) -> ErrorAnalysis:
    """Validate the required, aggregate-only human review for a publication."""
    document = _load_bounded_mapping(path, "error analysis")
    _exact_keys(
        document,
        {
            "schema_version",
            "candidate_benchmark_sha256",
            "baseline_benchmark_sha256",
            "review_scope",
            "findings",
            "additional_limitations",
        },
        "error analysis",
    )
    if document["schema_version"] != ERROR_ANALYSIS_SCHEMA_VERSION:
        raise EvaluationPublicationError("unsupported error analysis schema version")
    if document["candidate_benchmark_sha256"] != candidate_benchmark_sha256:
        raise EvaluationPublicationError("error analysis is not bound to the candidate benchmark")
    if document["baseline_benchmark_sha256"] != baseline_benchmark_sha256:
        raise EvaluationPublicationError("error analysis is not bound to the baseline benchmark")
    if document["review_scope"] != "private_record_level_review_without_identifiers":
        raise EvaluationPublicationError("error analysis must declare the approved private review scope")

    conditions = {condition.identifier: condition for condition in candidate.conditions}
    findings_payload = document["findings"]
    if not isinstance(findings_payload, list) or not findings_payload:
        raise EvaluationPublicationError("error analysis.findings must contain at least one reviewed finding")
    findings: list[ErrorFinding] = []
    for index, raw_finding in enumerate(findings_payload):
        location = f"error analysis.findings[{index}]"
        finding = _mapping(raw_finding, location)
        _exact_keys(finding, {"condition_id", "category", "affected_count", "summary"}, location)
        condition_id = _string(finding["condition_id"], f"{location}.condition_id")
        if condition_id not in conditions:
            raise EvaluationPublicationError(f"{location}.condition_id is not a benchmark condition")
        category = _string(finding["category"], f"{location}.category")
        if category not in ERROR_CATEGORIES:
            raise EvaluationPublicationError(f"{location}.category is not an allowed review category")
        affected_count = _positive_or_zero_integer(finding["affected_count"], f"{location}.affected_count")
        if affected_count > conditions[condition_id].sample_count:
            raise EvaluationPublicationError(f"{location}.affected_count exceeds the condition sample count")
        summary = _safe_review_text(finding["summary"], f"{location}.summary")
        findings.append(ErrorFinding(condition_id, category, affected_count, summary))

    limitations_payload = document["additional_limitations"]
    if not isinstance(limitations_payload, list) or not limitations_payload:
        raise EvaluationPublicationError(
            "error analysis.additional_limitations must contain at least one limitation"
        )
    limitations = tuple(
        _safe_review_text(item, f"error analysis.additional_limitations[{index}]")
        for index, item in enumerate(limitations_payload)
    )
    return ErrorAnalysis(tuple(findings), limitations)


def build_publication(
    *,
    candidate: BenchmarkReport,
    baseline: BenchmarkReport,
    calibration_document: Mapping[str, object],
    error_analysis: ErrorAnalysis,
    candidate_benchmark_sha256: str,
    baseline_benchmark_sha256: str,
    calibration_artifact_sha256: str,
    error_analysis_sha256: str,
) -> EvaluationPublication:
    """Bind compatible reports, exact candidate calibration, and human review."""
    _validate_comparison(candidate, baseline)
    try:
        calibration = validate_calibration_artifact(
            calibration_document,
            detector_id=candidate.detector_id,
            detector_version=candidate.detector_version,
            detector_family=candidate.detector_family,
        )
    except ModelReleaseValidationError as error:
        raise EvaluationPublicationError(f"invalid candidate calibration artifact: {error}") from error
    points = _calibration_points(calibration_document, candidate)
    for value, label in (
        (candidate_benchmark_sha256, "candidate benchmark checksum"),
        (baseline_benchmark_sha256, "baseline benchmark checksum"),
        (calibration_artifact_sha256, "calibration artifact checksum"),
        (error_analysis_sha256, "error analysis checksum"),
    ):
        _sha256(value, label)
    return EvaluationPublication(
        candidate=candidate,
        baseline=baseline,
        calibration_version=calibration.calibration_version,
        calibration_validation_manifest_sha256=calibration.validation_manifest_sha256,
        calibration_points=points,
        error_analysis=error_analysis,
        candidate_benchmark_sha256=candidate_benchmark_sha256,
        baseline_benchmark_sha256=baseline_benchmark_sha256,
        calibration_artifact_sha256=calibration_artifact_sha256,
        error_analysis_sha256=error_analysis_sha256,
    )


def write_publication(output_directory: Path, publication: EvaluationPublication) -> None:
    """Create one new publication bundle atomically; existing output is preserved."""
    if output_directory.exists():
        raise EvaluationPublicationError(f"publication output already exists: {output_directory}")
    if not output_directory.name:
        raise EvaluationPublicationError("publication output directory must have a name")
    try:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}-", dir=output_directory.parent))
    except OSError as error:
        raise EvaluationPublicationError(f"could not create publication staging directory: {error}") from error

    try:
        artifacts = _render_artifacts(publication)
        checksums: dict[str, str] = {}
        for name, content in artifacts.items():
            path = staging / name
            path.write_text(content, encoding="utf-8")
            checksums[name] = _file_sha256(path)
        manifest = _publication_manifest(publication, checksums)
        manifest_path = staging / "evaluation-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(output_directory)
    except OSError as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationPublicationError(f"could not write publication output: {error}") from error
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-benchmark", required=True, type=Path)
    parser.add_argument("--baseline-benchmark", required=True, type=Path)
    parser.add_argument("--candidate-calibration", required=True, type=Path)
    parser.add_argument("--error-analysis", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for public aggregate Markdown, SVG, and manifest artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build a publication bundle after all private evaluation inputs are reviewed."""
    args = parse_args(argv)
    try:
        candidate_checksum = _file_sha256(args.candidate_benchmark)
        baseline_checksum = _file_sha256(args.baseline_benchmark)
        calibration_checksum = _file_sha256(args.candidate_calibration)
        error_analysis_checksum = _file_sha256(args.error_analysis)
        candidate = load_benchmark_report(args.candidate_benchmark)
        baseline = load_benchmark_report(args.baseline_benchmark)
        error_analysis = load_error_analysis(
            args.error_analysis,
            candidate=candidate,
            candidate_benchmark_sha256=candidate_checksum,
            baseline_benchmark_sha256=baseline_checksum,
        )
        publication = build_publication(
            candidate=candidate,
            baseline=baseline,
            calibration_document=load_json_object(args.candidate_calibration, "candidate calibration artifact"),
            error_analysis=error_analysis,
            candidate_benchmark_sha256=candidate_checksum,
            baseline_benchmark_sha256=baseline_checksum,
            calibration_artifact_sha256=calibration_checksum,
            error_analysis_sha256=error_analysis_checksum,
        )
        write_publication(args.output_dir, publication)
    except (EvaluationPublicationError, ModelReleaseValidationError) as error:
        print(f"evaluation publication failed: {error}", file=__import__("sys").stderr)
        return 1
    print(f"evaluation publication created: {args.output_dir}")
    return 0


def _parse_conditions(value: object) -> tuple[ConditionMetrics, ...]:
    if not isinstance(value, list):
        raise EvaluationPublicationError("benchmark report.conditions must be a list")
    expected = standard_conditions()
    if len(value) != len(expected):
        raise EvaluationPublicationError("benchmark report.conditions does not have the fixed v1 matrix")
    conditions: list[ConditionMetrics] = []
    for index, (raw_condition, expected_condition) in enumerate(zip(value, expected)):
        location = f"benchmark report.conditions[{index}]"
        condition = _mapping(raw_condition, location)
        _exact_keys(
            condition,
            {
                "id",
                "kind",
                "parameters",
                "sample_count",
                "labels",
                "auroc",
                "mean_synthetic_probability",
                "latency_ms",
            },
            location,
        )
        identifier = _string(condition["id"], f"{location}.id")
        if identifier != expected_condition.identifier or condition["kind"] != expected_condition.kind:
            raise EvaluationPublicationError(f"{location} does not match the fixed v1 condition matrix")
        parameters = _mapping(condition["parameters"], f"{location}.parameters")
        if dict(parameters) != dict(expected_condition.parameters):
            raise EvaluationPublicationError(f"{location}.parameters does not match the fixed v1 matrix")
        sample_count = _positive_integer(condition["sample_count"], f"{location}.sample_count")
        labels = _label_counts(condition["labels"], sample_count, f"{location}.labels")
        auroc = _probability(condition["auroc"], f"{location}.auroc")
        means = _probability_mapping(
            condition["mean_synthetic_probability"],
            set(LABEL_TO_TARGET),
            f"{location}.mean_synthetic_probability",
        )
        latency = _mapping(condition["latency_ms"], f"{location}.latency_ms")
        _exact_keys(latency, {"scorer_wall", "detector_reported"}, f"{location}.latency_ms")
        scorer_wall = _latency_summary(latency["scorer_wall"], sample_count, f"{location}.latency_ms.scorer_wall")
        detector_reported = (
            None
            if latency["detector_reported"] is None
            else _latency_summary(
                latency["detector_reported"], sample_count, f"{location}.latency_ms.detector_reported"
            )
        )
        conditions.append(
            ConditionMetrics(
                identifier,
                expected_condition.kind,
                dict(parameters),
                sample_count,
                labels,
                auroc,
                means,
                scorer_wall,
                detector_reported,
            )
        )
    return tuple(conditions)


def _validate_comparison(candidate: BenchmarkReport, baseline: BenchmarkReport) -> None:
    if candidate.detector_label == baseline.detector_label:
        raise EvaluationPublicationError("candidate and baseline must be different immutable detector releases")
    if candidate.inputs != baseline.inputs:
        raise EvaluationPublicationError("candidate and baseline must use exactly the same held-out benchmark inputs")
    if candidate.source_revision != baseline.source_revision:
        raise EvaluationPublicationError("candidate and baseline must use the same benchmark source revision")
    for candidate_condition, baseline_condition in zip(candidate.conditions, baseline.conditions):
        if (
            candidate_condition.identifier != baseline_condition.identifier
            or candidate_condition.parameters != baseline_condition.parameters
            or candidate_condition.sample_count != baseline_condition.sample_count
            or candidate_condition.labels != baseline_condition.labels
        ):
            raise EvaluationPublicationError("candidate and baseline condition coverage does not match")


def _calibration_points(
    document: Mapping[str, object], candidate: BenchmarkReport
) -> tuple[tuple[float, float], ...]:
    detectors = document.get("detectors")
    assert isinstance(detectors, list)  # validated by validate_calibration_artifact
    for detector in detectors:
        if not isinstance(detector, Mapping):
            continue
        if detector.get("id") == candidate.detector_id and detector.get("version") == candidate.detector_version:
            raw_points = detector.get("points")
            assert isinstance(raw_points, list)
            return tuple(
                (float(point["raw_probability"]), float(point["calibrated_probability"]))
                for point in raw_points
                if isinstance(point, Mapping)
            )
    raise EvaluationPublicationError("candidate calibration points could not be found")


def _render_artifacts(publication: EvaluationPublication) -> Mapping[str, str]:
    return {
        "evaluation.md": _evaluation_markdown(publication),
        "calibration-curve.svg": _calibration_svg(publication),
        "condition-comparison.svg": _comparison_svg(publication, kind="auroc"),
        "latency-comparison.svg": _comparison_svg(publication, kind="latency"),
    }


def _evaluation_markdown(publication: EvaluationPublication) -> str:
    latency_source = _latency_source(publication)
    lines = [
        "# Veritas Face held-out evaluation",
        "",
        "## Scope",
        "",
        f"This report compares candidate `{_markdown_code(publication.candidate.detector_label)}` with baseline `{_markdown_code(publication.baseline.detector_label)}` on the same reviewed held-out test records and the fixed v1 robustness matrix. Scores are probabilistic detector evidence, not proof that any image is camera-origin or fully synthetic.",
        "",
        f"- Candidate detector family: `{_markdown_code(publication.candidate.detector_family)}`",
        f"- Baseline detector family: `{_markdown_code(publication.baseline.detector_family)}`",
        f"- Benchmark source revision: `{publication.candidate.source_revision}`",
        f"- Held-out record-set digest: `{publication.candidate.inputs['heldout_test_records_sha256']}`",
        f"- Candidate calibration release: `{_markdown_code(publication.calibration_version)}`",
        f"- Calibration validation-manifest digest: `{publication.calibration_validation_manifest_sha256}`",
        "",
        "The calibration chart is a release-specific raw-to-calibrated score mapping. It is not a reliability estimate for this benchmark, and these test records were not used for calibration.",
        "",
        "## Aggregate metrics and latency",
        "",
        "![Candidate calibration mapping](calibration-curve.svg)",
        "",
        "![AUROC by fixed condition](condition-comparison.svg)",
        "",
        "![Latency by fixed condition](latency-comparison.svg)",
        "",
        f"Latency values below use `{latency_source}`. `{_latency_source_explanation(latency_source)}`",
        "",
        "| Condition | Samples | Candidate AUROC | Baseline AUROC | Delta | Candidate p95 ms | Baseline p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate_condition, baseline_condition in zip(
        publication.candidate.conditions, publication.baseline.conditions
    ):
        candidate_latency = _chosen_latency(candidate_condition, latency_source)
        baseline_latency = _chosen_latency(baseline_condition, latency_source)
        lines.append(
            "| {condition} | {samples} | {candidate:.3f} | {baseline:.3f} | {delta:+.3f} | {candidate_p95:.2f} | {baseline_p95:.2f} |".format(
                condition=candidate_condition.identifier,
                samples=candidate_condition.sample_count,
                candidate=candidate_condition.auroc,
                baseline=baseline_condition.auroc,
                delta=candidate_condition.auroc - baseline_condition.auroc,
                candidate_p95=candidate_latency.p95,
                baseline_p95=baseline_latency.p95,
            )
        )
    lines.extend(
        [
            "",
            "AUROC summarizes rank separation only; it is not an operating threshold, accuracy claim, or validation of a deployment verdict. p95 latency is reported for the exactly identified benchmark adapters and hardware context captured privately by the operator; it is not a production service-level objective.",
            "",
            "## Reviewed error analysis",
            "",
            "The following aggregate findings were written after private record-level review. They contain no record identifiers, paths, images, prompts, seeds, or individual scores. `affected_count` is an analyst-reported triage count, not a new model metric.",
            "",
            "| Condition | Review category | Affected count | Finding |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for finding in publication.error_analysis.findings:
        lines.append(
            f"| {finding.condition_id} | {finding.category.replace('_', ' ')} | {finding.affected_count} | {_markdown_cell(finding.summary)} |"
        )
    lines.extend(["", "## Limitations", ""])
    for limitation in _base_limitations(publication) + publication.error_analysis.additional_limitations:
        lines.append(f"- {_markdown_text(limitation)}")
    lines.extend(
        [
            "",
            "## Reproducibility and review links",
            "",
            "The accompanying `evaluation-manifest.json` binds this publication to checksums of the two private aggregate benchmark inputs, the exact calibration artifact, and each generated public artifact. It does not expose private benchmark inputs or record-level data.",
            "",
        ]
    )
    return "\n".join(lines)


def _calibration_svg(publication: EvaluationPublication) -> str:
    points = " ".join(
        f"{_plot_x(raw):.1f},{_plot_y(calibrated):.1f}"
        for raw, calibrated in publication.calibration_points
    )
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360" role="img" aria-labelledby="title desc">',
            "<title id=\"title\">Candidate calibration mapping</title>",
            "<desc id=\"desc\">Raw detector probability mapped to calibrated probability for the exact candidate release.</desc>",
            '<rect width="720" height="360" fill="#ffffff"/>',
            '<text x="360" y="28" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#172554">Candidate calibration mapping</text>',
            '<line x1="70" y1="300" x2="670" y2="300" stroke="#334155"/>',
            '<line x1="70" y1="300" x2="70" y2="55" stroke="#334155"/>',
            '<line x1="70" y1="300" x2="670" y2="55" stroke="#cbd5e1" stroke-dasharray="6 4"/>',
            f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="4" stroke-linejoin="round"/>',
            '<text x="370" y="340" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">raw probability</text>',
            '<text x="20" y="180" transform="rotate(-90 20 180)" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">calibrated probability</text>',
            '<text x="70" y="320" font-family="sans-serif" font-size="12" fill="#475569">0</text>',
            '<text x="660" y="320" font-family="sans-serif" font-size="12" fill="#475569">1</text>',
            '<text x="48" y="300" font-family="sans-serif" font-size="12" fill="#475569">0</text>',
            '<text x="48" y="60" font-family="sans-serif" font-size="12" fill="#475569">1</text>',
            "</svg>",
            "",
        ]
    )


def _comparison_svg(publication: EvaluationPublication, *, kind: str) -> str:
    if kind not in {"auroc", "latency"}:
        raise EvaluationPublicationError("unsupported comparison chart kind")
    latency_source = _latency_source(publication)
    candidate_values = [
        condition.auroc if kind == "auroc" else _chosen_latency(condition, latency_source).p95
        for condition in publication.candidate.conditions
    ]
    baseline_values = [
        condition.auroc if kind == "auroc" else _chosen_latency(condition, latency_source).p95
        for condition in publication.baseline.conditions
    ]
    maximum = 1.0 if kind == "auroc" else max(candidate_values + baseline_values) * 1.1
    title = "AUROC by fixed condition" if kind == "auroc" else f"p95 latency by fixed condition ({latency_source})"
    unit = "AUROC" if kind == "auroc" else "milliseconds"
    bars: list[str] = []
    labels: list[str] = []
    for index, (candidate_value, baseline_value, condition) in enumerate(
        zip(candidate_values, baseline_values, publication.candidate.conditions)
    ):
        x = 72 + index * 76
        candidate_height = 205 * candidate_value / maximum
        baseline_height = 205 * baseline_value / maximum
        bars.extend(
            [
                f'<rect x="{x}" y="{285 - candidate_height:.1f}" width="26" height="{candidate_height:.1f}" fill="#2563eb"/>',
                f'<rect x="{x + 30}" y="{285 - baseline_height:.1f}" width="26" height="{baseline_height:.1f}" fill="#f97316"/>',
            ]
        )
        labels.append(
            f'<text x="{x + 28}" y="307" text-anchor="end" transform="rotate(-45 {x + 28} 307)" font-family="sans-serif" font-size="10" fill="#334155">{html.escape(condition.identifier)}</text>'
        )
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="390" viewBox="0 0 720 390" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{html.escape(title)}</title>",
            f"<desc id=\"desc\">Blue bars are the candidate detector and orange bars are the baseline detector. Vertical scale is {html.escape(unit)}.</desc>",
            '<rect width="720" height="390" fill="#ffffff"/>',
            f'<text x="360" y="28" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#172554">{html.escape(title)}</text>',
            '<line x1="65" y1="285" x2="690" y2="285" stroke="#334155"/>',
            '<line x1="65" y1="285" x2="65" y2="60" stroke="#334155"/>',
            f'<text x="42" y="68" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">{maximum:.2f}</text>',
            '<text x="42" y="288" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">0</text>',
            *bars,
            *labels,
            '<rect x="470" y="42" width="14" height="14" fill="#2563eb"/><text x="490" y="54" font-family="sans-serif" font-size="12" fill="#334155">candidate</text>',
            '<rect x="570" y="42" width="14" height="14" fill="#f97316"/><text x="590" y="54" font-family="sans-serif" font-size="12" fill="#334155">baseline</text>',
            "</svg>",
            "",
        ]
    )


def _publication_manifest(
    publication: EvaluationPublication, artifacts: Mapping[str, str]
) -> Mapping[str, object]:
    return {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "purpose": "public_aggregate_evaluation_with_reviewed_error_analysis",
        "candidate": {
            "detector": {
                "id": publication.candidate.detector_id,
                "version": publication.candidate.detector_version,
                "family": publication.candidate.detector_family,
            },
            "benchmark_sha256": publication.candidate_benchmark_sha256,
        },
        "baseline": {
            "detector": {
                "id": publication.baseline.detector_id,
                "version": publication.baseline.detector_version,
                "family": publication.baseline.detector_family,
            },
            "benchmark_sha256": publication.baseline_benchmark_sha256,
        },
        "benchmark_inputs": dict(sorted(publication.candidate.inputs.items())),
        "source_revision": publication.candidate.source_revision,
        "calibration": {
            "version": publication.calibration_version,
            "validation_manifest_sha256": publication.calibration_validation_manifest_sha256,
            "artifact_sha256": publication.calibration_artifact_sha256,
        },
        "error_analysis": {
            "finding_count": len(publication.error_analysis.findings),
            "review_scope": "private_record_level_review_without_identifiers",
            "artifact_sha256": publication.error_analysis_sha256,
        },
        "artifacts": dict(sorted(artifacts.items())),
    }


def _latency_source(publication: EvaluationPublication) -> str:
    reports = (*publication.candidate.conditions, *publication.baseline.conditions)
    return "detector_reported" if all(condition.detector_reported is not None for condition in reports) else "scorer_wall"


def _latency_source_explanation(source: str) -> str:
    if source == "detector_reported":
        return "This is the detector adapter's reported inference latency; it excludes any unreported client or transport overhead."
    return "This is measured scorer-adapter wall time, so it includes adapter, preprocessing, and invocation overhead in the private evaluation environment."


def _chosen_latency(condition: ConditionMetrics, source: str) -> LatencySummary:
    if source == "detector_reported" and condition.detector_reported is not None:
        return condition.detector_reported
    return condition.scorer_wall


def _base_limitations(publication: EvaluationPublication) -> tuple[str, ...]:
    return (
        "The held-out set is bounded to reviewed camera-origin portraits and fully synthetic text-to-image portraits; results do not establish performance for an unmeasured population, generator, camera, or transformation severity.",
        "V1 evaluates one dominant face in a still image. It does not claim to detect face swaps, identity deepfakes, AI edits, or general image authenticity.",
        "The test records are separate from the candidate calibration records. The calibration mapping is release-specific and does not turn a detector score into proof of origin.",
        "Missing or invalid metadata and provenance are never treated as evidence that a portrait is authentic.",
        "Latency reflects the fixed benchmark adapters and private test environment, not public-demo cold starts, network transit, concurrency, or quota behavior.",
    )


def _load_bounded_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise EvaluationPublicationError(f"cannot read {label}: {error}") from error
    if len(content) > 256 * 1024:
        raise EvaluationPublicationError(f"{label} exceeds 256 KiB")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationPublicationError(f"{label} is not valid JSON") from error
    return _mapping(value, label)


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EvaluationPublicationError(f"{location} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], location: str) -> None:
    if set(value) != expected:
        raise EvaluationPublicationError(f"{location} has unexpected or missing fields")


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationPublicationError(f"{location} must be a non-empty string")
    return value.strip()


def _sha256(value: object, location: str) -> str:
    result = _string(value, location)
    if not SHA256_PATTERN.fullmatch(result):
        raise EvaluationPublicationError(f"{location} must be sha256:<64 lowercase hex>")
    return result


def _digest_mapping(value: object, expected: set[str], location: str) -> Mapping[str, str]:
    mapping = _mapping(value, location)
    _exact_keys(mapping, expected, location)
    return {key: _sha256(mapping[key], f"{location}.{key}") for key in sorted(expected)}


def _positive_integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise EvaluationPublicationError(f"{location} must be a positive integer")
    return value


def _positive_or_zero_integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvaluationPublicationError(f"{location} must be a non-negative integer")
    return value


def _probability(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvaluationPublicationError(f"{location} must be a finite probability")
    if not 0 <= float(value) <= 1:
        raise EvaluationPublicationError(f"{location} must be from 0 through 1")
    return float(value)


def _label_counts(value: object, sample_count: int, location: str) -> Mapping[str, int]:
    mapping = _mapping(value, location)
    expected = set(LABEL_TO_TARGET)
    _exact_keys(mapping, expected, location)
    counts = {label: _positive_integer(mapping[label], f"{location}.{label}") for label in sorted(expected)}
    if sum(counts.values()) != sample_count:
        raise EvaluationPublicationError(f"{location} must add up to the condition sample count")
    return counts


def _probability_mapping(value: object, expected: set[str], location: str) -> Mapping[str, float]:
    mapping = _mapping(value, location)
    _exact_keys(mapping, expected, location)
    return {key: _probability(mapping[key], f"{location}.{key}") for key in sorted(expected)}


def _latency_summary(value: object, sample_count: int, location: str) -> LatencySummary:
    mapping = _mapping(value, location)
    _exact_keys(mapping, {"count", "mean", "p50", "p95"}, location)
    count = _positive_integer(mapping["count"], f"{location}.count")
    if count != sample_count:
        raise EvaluationPublicationError(f"{location}.count must equal the condition sample count")
    values = {
        key: _nonnegative_number(mapping[key], f"{location}.{key}")
        for key in ("mean", "p50", "p95")
    }
    if values["p95"] < values["p50"]:
        raise EvaluationPublicationError(f"{location}.p95 must be at least p50")
    return LatencySummary(count, values["mean"], values["p50"], values["p95"])


def _nonnegative_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvaluationPublicationError(f"{location} must be a finite number")
    if float(value) < 0:
        raise EvaluationPublicationError(f"{location} must be non-negative")
    return float(value)


def _safe_review_text(value: object, location: str) -> str:
    text = _string(value, location)
    if len(text) > 500 or "\n" in text or "\r" in text:
        raise EvaluationPublicationError(f"{location} must be one line of at most 500 characters")
    lowered = text.lower()
    if any(marker in lowered for marker in ("record_id", "relative_path", "image_path", "prompt", "seed")):
        raise EvaluationPublicationError(f"{location} must not include private record identifiers or generation details")
    return text


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EvaluationPublicationError(f"cannot hash input: {error}") from error
    return f"sha256:{digest.hexdigest()}"


def _plot_x(value: float) -> float:
    return 70 + 600 * value


def _plot_y(value: float) -> float:
    return 300 - 245 * value


def _markdown_code(value: str) -> str:
    return value.replace("`", "")


def _markdown_text(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")


def _markdown_cell(value: str) -> str:
    return _markdown_text(value).replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
