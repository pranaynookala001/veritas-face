export type Evidence = {
  source: string;
  status: string;
  detail: string;
  version?: string;
  score?: number;
};

export type EvidenceReport = {
  report_version: string;
  calibration_version: string;
  verdict: "likely_synthetic" | "likely_authentic" | "inconclusive";
  confidence: number | null;
  reasons: string[];
  evidence: Evidence[];
  model_versions: Record<string, string>;
};

export type AnalysisJob = {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed" | "expired";
  expires_at: string;
  report?: EvidenceReport;
};

export class JobApiError extends Error {
  status: number;
}

export const DEFAULT_API_BASE_URL: string;

export function createAnalysisJob(file: File, options?: { apiBaseUrl?: string; fetchImpl?: typeof fetch }): Promise<AnalysisJob>;
export function getAnalysisJob(jobId: string, options?: { apiBaseUrl?: string; fetchImpl?: typeof fetch }): Promise<AnalysisJob>;
export function waitForCompletedReport(
  jobId: string,
  options?: {
    apiBaseUrl?: string;
    attempts?: number;
    delay?: (milliseconds: number) => Promise<void>;
    fetchImpl?: typeof fetch;
    intervalMs?: number;
  },
): Promise<EvidenceReport>;
