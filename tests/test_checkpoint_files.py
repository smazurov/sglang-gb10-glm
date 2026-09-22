"""Vendored checkpoint processor-interface files vs the sha256 contract.

Five originals are asserted directly against contracts/processor-metadata.json.
config.json is special: the contract refers to the NORMALIZED config (the
metadata producer's deterministic output — what the gate mounts over the
original at gate time), so the test regenerates it and pins the result to
profile.yaml's metadata.normalized_sha256 and the contract hash. This makes
the T1 gate's inputs fully verifiable in the free static tier.
"""
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoint"
CONTRACT = json.loads((ROOT / "contracts" / "processor-metadata.json").read_text())
PROFILE = yaml.safe_load((ROOT / "profile.yaml").read_text())
ORIGINAL_CONFIG_SHA = "e1c0246a44ebefb5fd6383fb57aebbf7ac69ff6e7b23e989c0571b279a0eca23"


class CheckpointFilesTests(unittest.TestCase):
    def test_vendored_files_match_contract(self):
        self.assertEqual(sorted(CONTRACT), sorted([
            "chat_template.jinja", "config.json", "generation_config.json",
            "processor_config.json", "tokenizer.json", "tokenizer_config.json"]))
        for name, digest in CONTRACT.items():
            if name == "config.json":
                continue  # contract refers to the normalized config; see below
            with self.subTest(file=name):
                data = (CHECKPOINT / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_original_config_pinned_and_normalized_output_matches_contract(self):
        original = CHECKPOINT / "config.json"
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), ORIGINAL_CONFIG_SHA)
        result = subprocess.run(
            [sys.executable, str(ROOT / "gates" / "normalize_modelopt.py")],
            input=original.read_bytes(), capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        normalized = result.stdout
        self.assertEqual(hashlib.sha256(normalized).hexdigest(), CONTRACT["config.json"])
        self.assertEqual(hashlib.sha256(normalized).hexdigest(),
                         PROFILE["checkpoint"]["metadata"]["normalized_sha256"])
        # The normalization is value-level: everything outside quantization
        # must pass through untouched, the original must NOT carry KV metadata
        # (the role's fail-closed guard), and the normalized output must.
        original_json = json.loads(original.read_bytes())
        self.assertNotIn("kv_cache_quant_algo", original_json["quantization_config"])
        normalized_json = json.loads(normalized)
        self.assertEqual(normalized_json["quantization_config"]["quant_algo"], "MIXED_PRECISION")
        self.assertIn("kv_cache_quant_algo", normalized_json["quantization_config"])

    def test_license_is_mit_and_attribution_carried(self):
        license_text = (CHECKPOINT / "LICENSE").read_text()
        self.assertIn("MIT License", license_text)
        self.assertIn("Z.AI", license_text)


if __name__ == "__main__":
    unittest.main()
