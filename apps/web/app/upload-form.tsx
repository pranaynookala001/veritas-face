"use client";

import { ChangeEvent, FormEvent, useId, useRef, useState } from "react";

import {
  MAX_UPLOAD_BYTES,
  formatFileSize,
  validateUploadFile,
  type UploadFile,
} from "../lib/upload-validation";

type Notice = {
  kind: "error" | "info" | "success";
  message: string;
};

export function UploadForm() {
  const inputId = useId();
  const hintId = useId();
  const noticeId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<UploadFile | null>(null);
  const [notice, setNotice] = useState<Notice>({
    kind: "info",
    message: "No portrait selected yet.",
  });

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0] ?? null;
    const result = validateUploadFile(file);

    if (!file || !result.isValid) {
      setSelectedFile(null);
      event.currentTarget.value = "";
      setNotice({ kind: "error", message: result.message });
      return;
    }

    setSelectedFile(file);
    setNotice({
      kind: "success",
      message: `${file.name} is ready for analysis (${formatFileSize(file.size)}).`,
    });
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const result = validateUploadFile(selectedFile);

    if (!result.isValid) {
      setSelectedFile(null);
      setNotice({ kind: "error", message: result.message });
      return;
    }

    setNotice({
      kind: "success",
      message: "Your portrait passed the local checks. Upload processing will be available once the job API is connected.",
    });
  }

  function clearSelection() {
    setSelectedFile(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    setNotice({ kind: "info", message: "Portrait selection cleared." });
  }

  return (
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
          <button className="text-button" type="button" onClick={clearSelection}>
            Remove
          </button>
        </div>
      ) : null}

      <button className="primary-button" type="submit" disabled={!selectedFile}>
        Validate selected portrait
      </button>
    </form>
  );
}
