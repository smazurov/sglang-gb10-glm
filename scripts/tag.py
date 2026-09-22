#!/usr/bin/env python3
"""Content-addressed image tag.

The hash covers every file that defines the image: profile.yaml,
docker/Dockerfile, patches/, gates/, contracts/. Nothing else (workflows,
tests, docs) participates; changing any included file always produces a new
tag and never silently reuses a stale image. Deterministic across machines:
sha256 over a sorted (path, sha256) manifest.
"""
import hashlib
import json
import sys
from pathlib import Path

INCLUDED = ["profile.yaml", "docker/Dockerfile", "patches", "gates", "contracts"]


def manifest(root: Path) -> dict:
    entries = {}
    for name in INCLUDED:
        path = root / name
        if path.is_file():
            files = [path]
        else:
            files = sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        for f in files:
            entries[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return entries


def sha8(root: Path) -> tuple[str, dict]:
    entries = manifest(root)
    blob = json.dumps(entries, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8], entries


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path.cwd()
    flags = sys.argv[1:]
    if "--sha8" in flags:
        print(sha8(root)[0])
        return
    if "--tag" in flags:
        import yaml
        profile = yaml.safe_load((root / "profile.yaml").read_text())
        print(f"{profile['image']['tag_prefix']}-{sha8(root)[0]}")
        return
    sha8_value, entries = sha8(root)
    print(json.dumps({"sha8": sha8_value, "files": entries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
