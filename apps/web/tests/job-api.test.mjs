import assert from "node:assert/strict";
import test from "node:test";

import {
  JobApiError,
  createAnalysisJob,
  waitForCompletedReport,
} from "../lib/job-api.js";

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
  };
}

test("submits a portrait as multipart form data and returns the queued receipt", async () => {
  let request;
  const receipt = await createAnalysisJob(
    new File(["portrait bytes"], "portrait.png", { type: "image/png" }),
    {
      apiBaseUrl: "http://api.test/",
      fetchImpl: async (url, init) => {
        request = { url, init };
        return jsonResponse({ job_id: "job-1", status: "queued", expires_at: "2026-09-20T00:00:00Z" }, 202);
      },
    },
  );

  assert.equal(request.url, "http://api.test/v1/jobs");
  assert.equal(request.init.method, "POST");
  assert.equal(request.init.body.get("portrait").name, "portrait.png");
  assert.equal(receipt.status, "queued");
});

test("waits for a completed report while the local worker is still processing", async () => {
  let calls = 0;
  const report = await waitForCompletedReport("job-1", {
    attempts: 2,
    delay: async () => {},
    fetchImpl: async () => {
      calls += 1;
      return jsonResponse(
        calls === 1
          ? { job_id: "job-1", status: "queued", expires_at: "2026-09-20T00:00:00Z" }
          : {
              job_id: "job-1",
              status: "completed",
              expires_at: "2026-09-20T00:00:00Z",
              report: {
                report_version: "local-evidence-v4",
                calibration_version: "not_available",
                verdict: "inconclusive",
                confidence: null,
                reasons: ["no_face_detected"],
                evidence: [],
                model_versions: {},
              },
            },
      );
    },
  });

  assert.equal(calls, 2);
  assert.equal(report.verdict, "inconclusive");
});

test("surfaces a terminal worker failure instead of waiting indefinitely", async () => {
  await assert.rejects(
    waitForCompletedReport("job-1", {
      fetchImpl: async () => jsonResponse({ job_id: "job-1", status: "failed", expires_at: "2026-09-20T00:00:00Z" }),
    }),
    (error) => error instanceof JobApiError && error.status === 409,
  );
});
