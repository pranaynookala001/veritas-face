const configuredApiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();

export const DEFAULT_API_BASE_URL = configuredApiBaseUrl || "http://localhost:8000";

export class JobApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "JobApiError";
    this.status = status;
  }
}

/**
 * Submit a validated browser File without exposing its name in later API responses.
 *
 * @param {File} file
 * @param {{ apiBaseUrl?: string, fetchImpl?: typeof fetch }} [options]
 */
export async function createAnalysisJob(file, options = {}) {
  const formData = new FormData();
  formData.append("portrait", file, file.name);
  return requestJson("/v1/jobs", {
    method: "POST",
    body: formData,
  }, options);
}

/**
 * Fetch public job state and a completed mock report when the worker has finished.
 *
 * @param {string} jobId
 * @param {{ apiBaseUrl?: string, fetchImpl?: typeof fetch }} [options]
 */
export async function getAnalysisJob(jobId, options = {}) {
  return requestJson(`/v1/jobs/${encodeURIComponent(jobId)}`, { method: "GET" }, options);
}

/**
 * Poll the temporary job resource until its report is available or reaches a terminal error.
 *
 * @param {string} jobId
 * @param {{ apiBaseUrl?: string, attempts?: number, delay?: (milliseconds: number) => Promise<void>, fetchImpl?: typeof fetch, intervalMs?: number }} [options]
 */
export async function waitForCompletedReport(jobId, options = {}) {
  const attempts = options.attempts ?? 24;
  const intervalMs = options.intervalMs ?? 250;
  const delay = options.delay ?? sleep;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const job = await getAnalysisJob(jobId, options);
    if (job.status === "completed" && job.report) {
      return job.report;
    }
    if (job.status === "failed") {
      throw new JobApiError("The local report could not be completed. Please try another eligible portrait.", 409);
    }
    if (job.status === "expired") {
      throw new JobApiError("The temporary analysis job expired before its report was available.", 410);
    }
    if (attempt < attempts - 1) {
      await delay(intervalMs);
    }
  }

  throw new JobApiError("The report is taking longer than expected. Please try again shortly.", 408);
}

async function requestJson(path, init, options) {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl(path, options.apiBaseUrl), init);
  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new JobApiError(
      payload?.error?.message || "The analysis service could not complete this request.",
      response.status,
    );
  }
  return payload;
}

function apiUrl(path, apiBaseUrl = DEFAULT_API_BASE_URL) {
  return `${apiBaseUrl.replace(/\/$/, "")}${path}`;
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
