#!/usr/bin/env python3
"""Exercise the public C++ inference path against Python ONNX Runtime."""

from __future__ import annotations

import base64
import http.client
import json
import re
import selectors
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper

if len(sys.argv) != 2:
    raise SystemExit("usage: test_preprocessing_parity.py PATH_TO_VERITAS_INFERENCE")

INFERENCE_BINARY = sys.argv.pop(1)
WIDTH = 224
HEIGHT = 224
CHANNELS = 3
MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STANDARD_DEVIATIONS = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_probability_model(path: Path) -> None:
    """Create a small, non-production model with a known one-value output."""

    model_input = helper.make_tensor_value_info(
        "normalized_nchw", TensorProto.FLOAT, [1, CHANNELS, HEIGHT, WIDTH]
    )
    model_output = helper.make_tensor_value_info("synthetic_probability", TensorProto.FLOAT, [1, 1])
    channel_weights = helper.make_tensor(
        "channel_weights",
        TensorProto.FLOAT,
        [CHANNELS, 1],
        [0.55, -0.40, 0.30],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean",
                ["normalized_nchw"],
                ["channel_means"],
                axes=[2, 3],
                keepdims=0,
            ),
            helper.make_node("MatMul", ["channel_means", "channel_weights"], ["weighted_mean"]),
            helper.make_node("Sigmoid", ["weighted_mean"], ["synthetic_probability"]),
        ],
        "veritas-face-preprocessing-parity",
        [model_input],
        [model_output],
        [channel_weights],
    )
    model = helper.make_model(
        graph,
        producer_name="veritas-face-tests",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    # Keep the fixture compatible with the pinned C++ runtime in CI.
    model.ir_version = 9
    checker.check_model(model)
    onnx.save_model(model, path)


def make_rgb_pixels() -> bytes:
    pixels = bytearray(WIDTH * HEIGHT * CHANNELS)
    for row in range(HEIGHT):
        for column in range(WIDTH):
            offset = (row * WIDTH + column) * CHANNELS
            pixels[offset] = (column * 3 + row * 5 + 17) % 256
            pixels[offset + 1] = (column * 11 + row * 7 + 29) % 256
            pixels[offset + 2] = (column * 13 + row * 19 + 43) % 256
    return bytes(pixels)


def python_probability(model_path: Path, pixels: bytes) -> float:
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).astype(np.float32)
    normalized_hwc = (rgb / 255.0 - MEANS) / STANDARD_DEVIATIONS
    normalized_nchw = np.transpose(normalized_hwc, (2, 0, 1))[np.newaxis, ...]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return float(session.run(None, {"normalized_nchw": normalized_nchw})[0].item())


class InferencePreprocessingParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="veritas-face-onnx-")
        cls.model_path = Path(cls.temporary_directory.name) / "parity-probability.onnx"
        make_probability_model(cls.model_path)
        cls.process = subprocess.Popen(
            [
                INFERENCE_BINARY,
                "--port",
                "0",
                "--model",
                str(cls.model_path),
                "--model-id",
                "preprocessing-parity-fixture",
                "--model-version",
                "fixture-v1",
                "--threads",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert cls.process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(cls.process.stdout, selectors.EVENT_READ)
        events = selector.select(timeout=10)
        if not events:
            cls.process.terminate()
            stderr = cls.process.stderr.read() if cls.process.stderr is not None else ""
            raise RuntimeError(f"Inference service did not start within ten seconds: {stderr}")
        startup_line = cls.process.stdout.readline()
        match = re.fullmatch(r"veritas-inference listening on http://127\.0\.0\.1:(\d+)\n", startup_line)
        if match is None:
            cls.process.terminate()
            stderr = cls.process.stderr.read() if cls.process.stderr is not None else ""
            raise RuntimeError(
                "Unexpected inference-service startup output: "
                f"{startup_line!r}; exit={cls.process.poll()!r}; stderr={stderr}"
            )
        cls.port = int(match.group(1))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=5)
        cls.temporary_directory.cleanup()

    def request(self, body: object) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        serialized = body if isinstance(body, str) else json.dumps(body)
        connection.request(
            "POST",
            "/v1/infer",
            body=serialized,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        status = response.status
        connection.close()
        return status, payload

    def valid_payload(self, pixels: bytes) -> dict:
        return {
            "face_crop": {
                "color_space": "RGB",
                "width": WIDTH,
                "height": HEIGHT,
                "channels": CHANNELS,
                "layout": "HWC",
                "value_range": "0_to_255",
                "pixels_base64": base64.b64encode(pixels).decode("ascii"),
            }
        }

    def test_cpu_onnx_output_matches_python_preprocessing_and_runtime(self) -> None:
        pixels = make_rgb_pixels()
        expected_probability = python_probability(self.model_path, pixels)

        status, payload = self.request(self.valid_payload(pixels))

        self.assertEqual(status, 200)
        self.assertAlmostEqual(payload["synthetic_probability"], expected_probability, places=6)
        self.assertEqual(
            payload["detector"],
            {"id": "preprocessing-parity-fixture", "version": "fixture-v1"},
        )
        self.assertIsInstance(payload["latency_ms"], float)
        self.assertGreaterEqual(payload["latency_ms"], 0.0)

    def test_model_info_reports_a_ready_cpu_model_and_fixed_preprocessing_contract(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("GET", "/v1/model-info")
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(
            payload["model"],
            {
                "id": "preprocessing-parity-fixture",
                "version": "fixture-v1",
                "status": "ready",
                "runtime": "onnxruntime-cpu",
            },
        )
        self.assertEqual(
            payload["input"],
            {
                "color_space": "RGB",
                "width": WIDTH,
                "height": HEIGHT,
                "channels": CHANNELS,
                "layout": "HWC",
                "value_range": "0_to_255",
                "normalization": "imagenet_rgb_v1",
            },
        )

    def test_invalid_payloads_are_rejected_without_running_the_model(self) -> None:
        status, payload = self.request("{not JSON")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_inference_request")

        invalid_shape = self.valid_payload(make_rgb_pixels())
        invalid_shape["face_crop"]["width"] = WIDTH - 1
        status, payload = self.request(invalid_shape)
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "invalid_face_crop")

        invalid_length = self.valid_payload(b"\x00")
        status, payload = self.request(invalid_length)
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "invalid_face_crop")


if __name__ == "__main__":
    unittest.main()
