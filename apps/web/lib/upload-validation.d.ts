export type UploadFile = {
  name: string;
  size: number;
  type: string;
};

export type UploadValidationResult =
  | { isValid: true; message: string }
  | { isValid: false; message: string };

export const MAX_UPLOAD_BYTES: number;

export function validateUploadFile(file: UploadFile | null): UploadValidationResult;
export function formatFileSize(bytes: number): string;
