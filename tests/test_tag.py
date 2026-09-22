"""Content-addressed tag: determinism and sensitivity."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tag as tag_mod  # noqa: E402


class TagTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(tag_mod.sha8(ROOT), tag_mod.sha8(ROOT))

    def test_covers_the_defining_file_sets(self):
        entries = tag_mod.manifest(ROOT)
        self.assertIn("profile.yaml", entries)
        self.assertIn("docker/Dockerfile", entries)
        self.assertTrue(any(k.startswith("patches/") for k in entries))
        self.assertTrue(any(k.startswith("gates/") for k in entries))
        self.assertTrue(any(k.startswith("contracts/") for k in entries))
        # CI/docs/tests do not define the image; they must not rotate the tag.
        self.assertFalse(any(k.startswith(".github/") or k.startswith("tests/") for k in entries))

    def test_sensitive_to_every_included_file(self):
        targets = ["profile.yaml", "docker/Dockerfile",
                   "gates/dependency-overlay.py", "contracts/requirements.lock"]
        with tempfile.TemporaryDirectory() as tmp:
            for name in targets:
                with self.subTest(target=name):
                    copy = Path(tmp) / "repo"
                    if copy.exists():
                        shutil.rmtree(copy)
                    shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", ".git", "wheels"))
                    baseline = tag_mod.sha8(copy)[0]
                    target = copy / name
                    data = bytearray(target.read_bytes())
                    data[0] = (data[0] + 1) % 256
                    target.write_bytes(bytes(data))
                    self.assertNotEqual(tag_mod.sha8(copy)[0], baseline)


if __name__ == "__main__":
    unittest.main()
