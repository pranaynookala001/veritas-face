import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_manifests import (  # noqa: E402
    ManifestValidationError,
    canonical_sha256,
    load_json,
    validate_manifests,
)


MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


class TrainingManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = load_json(MANIFEST_DIR / "portrait-sources-v1.json")
        self.splits = load_json(MANIFEST_DIR / "portrait-splits-v1.json")

    def test_checked_in_manifests_are_traceable_and_hold_out_generator_families(self) -> None:
        summary = validate_manifests(self.sources, self.splits)

        self.assertEqual(summary.source_count, 4)
        self.assertEqual(
            summary.synthetic_families,
            {
                "latent_diffusion": "train",
                "latent_diffusion_xl": "test",
                "rectified_flow": "validation",
            },
        )
        self.assertRegex(summary.source_sha256, r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(summary.split_sha256, r"^sha256:[0-9a-f]{64}$")

    def test_generator_family_cannot_cross_split_boundaries(self) -> None:
        invalid_splits = copy.deepcopy(self.splits)
        invalid_splits["splits"]["validation"]["source_ids"].append(
            "stable-diffusion-v1-5-text-to-image"
        )

        with self.assertRaisesRegex(ManifestValidationError, "appears in both"):
            validate_manifests(self.sources, invalid_splits)

    def test_synthetic_sources_reject_reference_images_or_unpinned_revisions(self) -> None:
        invalid_sources = copy.deepcopy(self.sources)
        generator = invalid_sources["sources"][1]["generator"]
        generator["image_conditioning_allowed"] = True
        generator["revision"] = "not-pinned"

        with self.assertRaisesRegex(ManifestValidationError, "revision"):
            validate_manifests(invalid_sources, self.splits)

    def test_digest_is_canonical_across_json_key_ordering(self) -> None:
        self.assertEqual(
            canonical_sha256({"a": [1, 2], "b": True}),
            canonical_sha256({"b": True, "a": [1, 2]}),
        )


if __name__ == "__main__":
    unittest.main()
