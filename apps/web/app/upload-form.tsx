"use client";

import { ChangeEvent, FormEvent, useId, useRef, useState } from "react";

import {
  MAX_UPLOAD_BYTES,
  formatFileSize,
  validateUploadFile,
} from "../lib/upload-validation";
import {
  createAnalysisJob,
  waitForCompletedReport,
  type EvidenceReport,
} from "../lib/job-api";

type Notice = {
  kind: "error" | "info" | "success";
  message: string;
};

export function UploadForm() {
  const inputId = useId();
  const hintId = useId();
  const noticeId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [report, setReport] = useState<EvidenceReport | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [notice, setNotice] = useState<Notice>({
    kind: "info",
    message: "No portrait selected yet.",
  });

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0] ?? null;
    const result = validateUploadFile(file);

    if (!file || !result.isValid) {
      setSelectedFile(null);
      setReport(null);
      event.currentTarget.value = "";
      setNotice({ kind: "error", message: result.message });
      return;
    }

    setSelectedFile(file);
    setReport(null);
    setNotice({
      kind: "success",
      message: `${file.name} is ready for analysis (${formatFileSize(file.size)}).`,
    });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting) {
      return;
    }
    const result = validateUploadFile(selectedFile);

    if (!selectedFile || !result.isValid) {
      setSelectedFile(null);
      setReport(null);
      setNotice({ kind: "error", message: result.message });
      return;
    }

    setIsSubmitting(true);
    setReport(null);
    setNotice({ kind: "info", message: "Uploading the portrait and preparing its local evidence report…" });
    try {
      const job = await createAnalysisJob(selectedFile);
      setNotice({ kind: "info", message: "Upload accepted. Checking local metadata, Content Credentials, and face-quality gates…" });
      const completedReport = await waitForCompletedReport(job.job_id);
      setReport(completedReport);
      setNotice({
        kind: "success",
        message: "The local evidence report is ready. It remains inconclusive until synthetic-detector evidence is available.",
      });
    } catch (error) {
      setNotice({
        kind: "error",
        message: error instanceof Error ? error.message : "The analysis service could not complete this request.",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  function clearSelection() {
    setSelectedFile(null);
    setReport(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    setNotice({ kind: "info", message: "Portrait selection cleared." });
  }

  return (
    <>
      <form className="upload-form" onSubmit={handleSubmit} noValidate>
      <div className="upload-form__heading">
        <p className="eyebrow">START WITH A STILL PORTRAIT</p>
        <h1>Upload a portrait for an evidence-led assessment.</h1>
        <p>
          Choose one JPEG, PNG, or WebP image. This first check only confirms that the
          file is eligible for analysis—it does not make an authenticity claim.
        </p>
      </div>

      <div className="upload-form__field">
        <input
          ref={inputRef}
          id={inputId}
          className="file-picker__input"
          name="portrait"
          type="file"
          accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
          aria-describedby={`${hintId} ${noticeId}`}
          onChange={handleFileChange}
          disabled={isSubmitting}
        />
        <label className="file-picker" htmlFor={inputId}>
          <span className="file-picker__icon" aria-hidden="true">+</span>
          <span className="file-picker__copy">
            <strong>Choose a portrait</strong>
            <span>JPEG, PNG, or WebP · maximum {formatFileSize(MAX_UPLOAD_BYTES)}</span>
          </span>
        </label>
        <p className="field-hint" id={hintId}>
          Files are checked again by the service before they are stored or analyzed.
        </p>
      </div>

      <div
        className={`upload-notice upload-notice--${notice.kind}`}
        id={noticeId}
        role={notice.kind === "error" ? "alert" : "status"}
        aria-live={notice.kind === "error" ? "assertive" : "polite"}
        aria-atomic="true"
      >
        <span className="upload-notice__marker" aria-hidden="true" />
        <span>{notice.message}</span>
      </div>

      {selectedFile ? (
        <div className="selected-file">
          <div>
            <p className="selected-file__name">{selectedFile.name}</p>
            <p>{formatFileSize(selectedFile.size)} · format accepted</p>
          </div>
          <button className="text-button" type="button" onClick={clearSelection} disabled={isSubmitting}>
            Remove
          </button>
        </div>
      ) : null}

      <button className="primary-button" type="submit" disabled={!selectedFile || isSubmitting}>
        {isSubmitting ? "Preparing evidence report…" : "Create evidence report"}
      </button>
      </form>

      {report ? <EvidenceReportPanel report={report} /> : null}
    </>
  );
}

function EvidenceReportPanel({ report }: { report: EvidenceReport }) {
  return (
    <section className="report-panel" aria-labelledby="report-heading">
      <p className="eyebrow">COMPLETED LOCAL EVIDENCE REPORT</p>
      <h2 id="report-heading">Assessment: {formatVerdict(report.verdict)}</h2>
      <p className="report-panel__summary">
        This is a local eligibility report, not a proof of origin. It remains inconclusive because
        no synthetic-portrait detector score is available yet.
      </p>

      <div className="report-panel__section">
        <h3>Why the assessment is inconclusive</h3>
        <ul className="reason-list">
          {report.reasons.map((reason) => <li key={reason}>{formatReason(reason)}</li>)}
        </ul>
      </div>

      <div className="report-panel__section">
        <h3>Evidence available locally</h3>
        <dl className="evidence-list">
          {report.evidence.map((item) => (
            <div key={`${item.source}-${item.status}`} className="evidence-list__item">
              <dt>{formatLabel(item.source)} <span>{formatLabel(item.status)}</span></dt>
              <dd>{item.detail}</dd>
            </div>
          ))}
        </dl>
      </div>

      <p className="report-panel__versions">
        Report {report.report_version} · calibration {report.calibration_version}
      </p>
    </section>
  );
}

function formatVerdict(verdict: EvidenceReport["verdict"]) {
  return verdict.split("_").map(capitalize).join(" ");
}

function formatReason(reason: string) {
  const labels: Record<string, string> = {
    no_face_detected: "No face was detected for a portrait assessment.",
    multiple_faces_ambiguous: "Multiple similarly sized faces make the primary subject ambiguous.",
    face_too_small: "The selected face is too small for a meaningful assessment.",
    face_too_blurry: "The selected face is too blurry for a meaningful assessment.",
    face_too_dark: "The selected face is too dark for a meaningful assessment.",
    face_too_bright: "The selected face is too bright for a meaningful assessment.",
    no_synthetic_detector_score: "No synthetic-portrait detector score is available in this local evidence report.",
  };
  return labels[reason] ?? formatLabel(reason);
}

function formatLabel(value: string) {
  return value.split("_").map(capitalize).join(" ");
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
