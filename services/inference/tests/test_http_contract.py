#!/usr/bin/env python3
"""Black-box contract tests for the dependency-free inference HTTP service."""

from __future__ import annotations

import http.client
import json
import re
import selectors
import subprocess
import sys
import unittest

if len(sys.argv) != 2:
    raise SystemExit("usage: test_http_contract.py PATH_TO_VERITAS_INFERENCE")

INFERENCE_BINARY = sys.argv.pop(1)


class InferenceHttpContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.process = subprocess.Popen(
            [INFERENCE_BINARY, "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert cls.process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(cls.process.stdout, selectors.EVENT_READ)
        events = selector.select(timeout=5)
        if not events:
            cls.process.terminate()
            stderr = cls.process.stderr.read() if cls.process.stderr is not None else ""
            raise RuntimeError(f"Inference service did not start within five seconds: {stderr}")
        startup_line = cls.process.stdout.readline()
        match = re.fullmatch(r"veritas-inference listening on http://127\.0\.0\.1:(\d+)\n", startup_line)
        if match is None:
            cls.process.terminate()
            stderr = cls.process.stderr.read() if cls.process.stderr is not None else ""
            raise RuntimeError(f"Unexpected inference-service startup output: {startup_line!r}; {stderr}")
        cls.port = int(match.group(1))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=5)

    def request(self, method: str, path: str) -> tuple[int, dict[str, str], dict]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(method, path)
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        headers = dict(response.getheaders())
        status = response.status
        connection.close()
        return status, headers, body

    def test_health_and_healthz_report_liveness(self) -> None:
        expected = {
            "status": "ok",
            "service": "veritas-face-inference",
            "version": "0.1.0",
        }
        for path in ("/health", "/healthz"):
            status, headers, payload = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertEqual(payload, expected)
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertEqual(headers["Content-Type"], "application/json")

    def test_model_info_distinguishes_a_live_service_from_an_unloaded_model(self) -> None:
        status, _, payload = self.request("GET", "/v1/model-info")

        self.assertEqual(status, 200)
        self.assertEqual(payload["service"], "veritas-face-inference")
        self.assertEqual(payload["version"], "0.1.0")
        self.assertEqual(
            payload["model"],
            {
                "id": "synthetic-portrait-classifier",
                "version": "not_loaded",
                "status": "unavailable",
            },
        )
        self.assertEqual(
            payload["input"],
            {
                "color_space": "RGB",
                "width": 224,
                "height": 224,
                "channels": 3,
                "layout": "HWC",
                "value_range": "0_to_255",
                "normalization": "not_configured",
            },
        )

    def test_unknown_and_non_get_requests_are_explicit_json_errors(self) -> None:
        status, _, payload = self.request("GET", "/not-a-route")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")

        status, headers, payload = self.request("POST", "/health")
        self.assertEqual(status, 405)
        self.assertEqual(headers["Allow"], "GET")
        self.assertEqual(payload["error"]["code"], "method_not_allowed")

    def test_inference_contract_is_unavailable_until_a_model_is_loaded(self) -> None:
        status, _, payload = self.request("POST", "/v1/infer")
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "model_unavailable")

        status, headers, payload = self.request("GET", "/v1/infer")
        self.assertEqual(status, 405)
        self.assertEqual(headers["Allow"], "POST")
        self.assertEqual(payload["error"]["code"], "method_not_allowed")


if __name__ == "__main__":
    unittest.main()
