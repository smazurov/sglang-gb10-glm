"""Offline checks for the opt-in GLM wheel/dependency contract.

The manifest/lock/baseline boundary assertions hold; the
installer is the Dockerfile, so installer-facing assertions are Dockerfile
assertions.
"""
import json
import unittest
from pathlib import Path

from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
DOCKERFILE = (ROOT / "docker" / "Dockerfile").read_text()


class DependencyContractTests(unittest.TestCase):
    def test_published_wheels_and_exact_baseline_boundary(self):
        manifest = json.loads((CONTRACTS / "dependencies.json").read_text())
        baseline = json.loads((CONTRACTS / "dependency-baseline.json").read_text())
        self.assertEqual(len(manifest["wheels"]), 5)
        self.assertEqual({w["name"] for w in manifest["wheels"]},
                         {"transformers", "tokenizers", "torchcodec", "nvshmem4py-cu13", "cython"})
        self.assertEqual(len(baseline["conflicts"]), 13)
        self.assertEqual(baseline["base"], manifest["base"])
        self.assertEqual({r["requirement"] for r in manifest["intentional_overrides"]},
                         {"transformers==5.12.1", "tokenizers==0.22.2"})

    def test_lock_is_exactly_the_manifest(self):
        # The offline install consumes requirements.lock with --require-hashes;
        # it must be exactly the manifest entries (order included), or the
        # before-audit would reject the mismatch at build time anyway. Check
        # it here so the failure surfaces in the free static tier.
        manifest = json.loads((CONTRACTS / "dependencies.json").read_text())
        expected = [f"{canonicalize_name(w['name'])}=={w['version']} --hash=sha256:{w['sha256']}"
                    for w in manifest["wheels"]]
        lock = (CONTRACTS / "requirements.lock").read_text().splitlines()
        self.assertEqual(lock, expected)

    def test_install_is_offline_hash_locked_and_ordered(self):
        # before-audit < pip install < tree replacement < after-audit; the
        # audits are networkless; the install is resolver-free with hashes.
        self.assertIn("--network=none", DOCKERFILE)
        self.assertIn("--no-index", DOCKERFILE)
        self.assertIn("--only-binary=:all:", DOCKERFILE)
        self.assertIn("--no-deps", DOCKERFILE)
        self.assertIn("--require-hashes", DOCKERFILE)
        for token in ("dependency-overlay.py before", "-m pip install",
                      "git apply --index", "dependency-overlay.py after"):
            self.assertIn(token, DOCKERFILE)
        positions = [DOCKERFILE.index(token) for token in
                     ("dependency-overlay.py before", "-m pip install",
                      "git apply --index", "dependency-overlay.py after")]
        self.assertEqual(positions, sorted(positions))

    def test_receipts_kept_wheels_dropped_after_audit(self):
        self.assertIn("rm -rf ${GLM_OVERLAY}/wheels", DOCKERFILE)
        self.assertIn("dependency-overlay.py after ${GLM_OVERLAY}", DOCKERFILE)


if __name__ == "__main__":
    unittest.main()
