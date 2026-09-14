export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

const ALLOWED_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const ALLOWED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp"]);

/**
 * @typedef {{ name: string, size: number, type: string }} UploadFile
 * @typedef {{ isValid: true, message: string } | { isValid: false, message: string }} UploadValidationResult
 */

/**
 * Validate browser-provided file metadata before an upload is attempted.
 * The API must inspect the content independently before processing it.
 *
 * @param {UploadFile | null} file
 * @returns {UploadValidationResult}
 */
export function validateUploadFile(file) {
  if (!file) {
    return { isValid: false, message: "Choose a portrait before continuing." };
  }

  if (!Number.isSafeInteger(file.size) || file.size <= 0) {
    return { isValid: false, message: "Choose a non-empty image file." };
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      isValid: false,
      message: `Choose an image no larger than ${formatFileSize(MAX_UPLOAD_BYTES)}.`,
    };
  }

  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_EXTENSIONS.has(extension)) {
    return { isValid: false, message: "Choose a JPEG, PNG, or WebP image." };
  }

  if (file.type && !ALLOWED_MIME_TYPES.has(file.type.toLowerCase())) {
    return { isValid: false, message: "Choose a JPEG, PNG, or WebP image." };
  }

  return { isValid: true, message: "Image is eligible for upload." };
}

/**
 * @param {number} bytes
 * @returns {string}
 */
export function formatFileSize(bytes) {
  if (bytes === 0) {
    return "0 bytes";
  }

  const units = ["bytes", "KiB", "MiB"];
  const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** unitIndex;
  const precision = Number.isInteger(value) || unitIndex === 0 || value >= 10 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[unitIndex]}`;
}
