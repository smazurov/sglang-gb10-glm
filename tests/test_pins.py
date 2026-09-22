"""Immutable pin assertions, and Dockerfile/profile pin-drift guard.

The Dockerfile ARG defaults must mirror profile.yaml exactly; CI passes only
release metadata, never pins — so these defaults are the build's actual pins.
"""
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILE = yaml.safe_load((ROOT / "profile.yaml").read_text())
DOCKERFILE = (ROOT / "docker" / "Dockerfile").read_text()


class PinTests(unittest.TestCase):
    def test_immutable_pins(self):
        self.assertEqual(PROFILE["profile"], "glm-flash-v2")
        self.assertEqual(PROFILE["image"]["repository"], "ghcr.io/smazurov/sglang-gb10-glm")
        self.assertEqual(
            PROFILE["base"]["image"],
            "lmsysorg/sglang@sha256:df8461b8099014daccc1dd548517578f110e601f6e893c55c23a83c2521bed53",
        )
        sglang = PROFILE["sglang"]
        self.assertEqual(sglang["head"], "d6fabb74b45d4fb92796cfb6740810b4811b018e")
        self.assertEqual(sglang["upstream_tree"], "ece17616d5a531bd37cfe476de07a30e6ada0ecb")
        self.assertEqual(sglang["patched_tree"], "4c582984e40ace4a203e369237954df26368bff0")
        checkpoint = PROFILE["checkpoint"]
        self.assertEqual(checkpoint["repository"], "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark")
        self.assertEqual(checkpoint["revision"], "53e77dbb04fa9dd68725daa899ec92eabf8a872c")
        self.assertEqual(PROFILE["expected_deps"], {
            "torch": "2.13.0+cu130", "tilelang": "0.1.12",
            "flashinfer-python": "0.6.18", "sglang-kernel": "0.4.6.post1",
            "flashinfer-cubin": "0.6.18", "flashinfer-jit-cache": "0.6.18+cu130",
            "cuda-tile": "1.6.0rc5", "nvidia-cutlass-dsl": "4.6.2",
        })

    def test_five_patches_with_hash_contract(self):
        patches = PROFILE["patches"]
        self.assertEqual(len(patches), 5)
        import hashlib
        for p in patches:
            with self.subTest(patch=p["file"]):
                data = (ROOT / "patches" / p["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), p["sha256"])
                self.assertTrue(p["markers"], "markers are rebase diagnostics; keep them")

    def test_dockerfile_arg_defaults_mirror_profile(self):
        args = dict(re.findall(r"^ARG (\w+)=([^\s]*)$", DOCKERFILE, re.M))
        self.assertEqual(args["BASE_IMAGE"], PROFILE["base"]["image"])
        self.assertEqual(args["SGLANG_HEAD"], PROFILE["sglang"]["head"])
        self.assertEqual(args["SGLANG_UPSTREAM_TREE"], PROFILE["sglang"]["upstream_tree"])
        self.assertEqual(args["SGLANG_PATCHED_TREE"], PROFILE["sglang"]["patched_tree"])
        self.assertEqual(args["MODEL_REPOSITORY"], PROFILE["checkpoint"]["repository"])
        self.assertEqual(args["MODEL_REVISION"], PROFILE["checkpoint"]["revision"])

    def test_dockerfile_applies_series_in_profile_order(self):
        positions = []
        for p in PROFILE["patches"]:
            needle = f"git apply --index ${{GLM_OVERLAY}}/patches/{p['file']}"
            self.assertIn(needle, DOCKERFILE)
            positions.append(DOCKERFILE.index(needle))
        self.assertEqual(positions, sorted(positions))

    def test_dockerfile_asserts_both_tree_hashes(self):
        self.assertIn("test \"$(git rev-parse 'HEAD^{tree}')\" = \"${SGLANG_UPSTREAM_TREE}\"", DOCKERFILE)
        self.assertIn("test \"$(git write-tree)\" = \"${SGLANG_PATCHED_TREE}\"", DOCKERFILE)


if __name__ == "__main__":
    unittest.main()
