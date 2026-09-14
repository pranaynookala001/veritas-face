import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_UPLOAD_BYTES,
  formatFileSize,
  validateUploadFile,
} from "../lib/upload-validation.js";

test("accepts supported image metadata", () => {
  const result = validateUploadFile({
    name: "portrait.WEBP",
    type: "image/webp",
    size: 2_048,
  });

  assert.deepEqual(result, { isValid: true, message: "Image is eligible for upload." });
});

test("allows a supported extension when the browser omits a MIME type", () => {
  const result = validateUploadFile({ name: "portrait.jpeg", type: "", size: 2_048 });

  assert.equal(result.isValid, true);
});

test("rejects a missing, empty, oversized, or unsupported file", () => {
  assert.match(validateUploadFile(null).message, /Choose a portrait/);
  assert.match(
    validateUploadFile({ name: "empty.png", type: "image/png", size: 0 }).message,
    /non-empty/,
  );
  assert.match(
    validateUploadFile({ name: "large.png", type: "image/png", size: MAX_UPLOAD_BYTES + 1 }).message,
    /10 MiB/,
  );
  assert.match(
    validateUploadFile({ name: "portrait.gif", type: "image/gif", size: 2_048 }).message,
    /JPEG, PNG, or WebP/,
  );
});

test("rejects an unsupported MIME type even when the extension is allowed", () => {
  const result = validateUploadFile({ name: "portrait.png", type: "image/gif", size: 2_048 });

  assert.equal(result.isValid, false);
  assert.match(result.message, /JPEG, PNG, or WebP/);
});

test("formats byte counts for visible file metadata", () => {
  assert.equal(formatFileSize(0), "0 bytes");
  assert.equal(formatFileSize(2_048), "2 KiB");
  assert.equal(formatFileSize(MAX_UPLOAD_BYTES), "10 MiB");
});
