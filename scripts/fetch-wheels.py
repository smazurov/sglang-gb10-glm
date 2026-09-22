#!/usr/bin/env python3
"""Download the exact pinned wheels into wheels/ for the offline image build.

Reuses dependency-overlay.validate_wheels for verification: every downloaded
artifact must match contracts/dependencies.json (URL shape, sha256, wheel
tags, dist-info identity) and wheels/ must contain exactly the manifest set
afterwards. A corrupt download is a hard failure, never a re-download.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("dependency_overlay", ROOT / "gates" / "dependency-overlay.py")
DEPS = module_from_spec(SPEC)
SPEC.loader.exec_module(DEPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=ROOT / "wheels")
    parser.add_argument("--local", action="store_true",
                        help="macOS/foreign-arch dry run: verify hashes only; the full "
                             "audit (platform tags, dist-info) runs in-image and on CI")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "contracts" / "dependencies.json").read_text())
    args.dest.mkdir(parents=True, exist_ok=True)
    for wheel in manifest["wheels"]:
        target = args.dest / wheel["filename"]
        if target.exists():
            data = target.read_bytes()
            if hashlib.sha256(data).hexdigest() == wheel["sha256"]:
                print(f"cache hit ok: {wheel['filename']}")
                continue
            raise SystemExit(f"cached wheel hash mismatch; investigate rather than redownload: {target}")
        print(f"downloading {wheel['filename']}")
        with urllib.request.urlopen(wheel["url"]) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != wheel["sha256"]:
            raise SystemExit(f"downloaded wheel hash mismatch: {wheel['filename']}")
        target.write_bytes(data)

    # Fail closed: exactly the manifest set, verified end to end.
    if args.local:
        for wheel in manifest["wheels"]:
            data = (args.dest / wheel["filename"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == wheel["sha256"], wheel["filename"]
        names = {p.name for p in args.dest.iterdir()}
        assert names == {w["filename"] for w in manifest["wheels"]}, names
        print("WHEELS_VERIFIED_LOCAL", len(manifest["wheels"]), flush=True)
    else:
        DEPS.validate_wheels(manifest, args.dest, ROOT / "contracts" / "requirements.lock")
        print("WHEELS_VERIFIED", len(manifest["wheels"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
