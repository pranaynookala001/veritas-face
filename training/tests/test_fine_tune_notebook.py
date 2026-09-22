import json
from pathlib import Path
import unittest


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "fine_tune_mobilenetv3_kaggle.ipynb"


class FineTuneNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.code = "\n".join(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_notebook_is_unexecuted_and_uses_the_pinned_lightweight_backbone(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        compile(self.code, str(NOTEBOOK_PATH), "exec")
        self.assertIn("MobileNet_V3_Small_Weights.IMAGENET1K_V1", self.code)
        self.assertIn("mobilenet_v3_small(weights=None)", self.code)
        self.assertIn("weights_sha256", self.code)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_notebook_validates_private_data_and_never_loads_held_out_test_records(self) -> None:
        self.assertIn("validate_record_manifest", self.code)
        self.assertIn("verify_record_files", self.code)
        self.assertIn("record_manifest_sha256", self.code)
        self.assertIn("train_records", self.code)
        self.assertIn("validation_records", self.code)
        self.assertNotIn("test_records", self.code)
        self.assertNotIn("test_loader", self.code)

    def test_notebook_records_determinism_and_does_not_select_a_product_threshold(self) -> None:
        self.assertIn("torch.use_deterministic_algorithms(True)", self.code)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", self.code)
        self.assertIn("build_run_metadata", self.code)
        self.assertNotIn("selected_threshold", self.code)


if __name__ == "__main__":
    unittest.main()
