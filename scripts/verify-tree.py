#!/usr/bin/env python3
"""Verify the pinned SGLang tree identity outside the image build.

Shallow-fetches sglang.patched_tree pins from profile.yaml, applies the patch
series with `git apply --index`, and checks both tree hashes. This is the
same assertion the Dockerfile performs, runnable on any host (CI static tier,
operator workstation) without pulling the base image.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], **kwargs) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=Path, help="keep the applied tree at this path instead of a temp dir")
    args = parser.parse_args()

    profile = yaml.safe_load((ROOT / "profile.yaml").read_text())
    sglang = profile["sglang"]
    patches = [ROOT / "patches" / p["file"] for p in profile["patches"]]

    workdir = args.keep if args.keep else Path(tempfile.mkdtemp(prefix="glm-tree-"))
    keep = bool(args.keep)
    if keep:
        workdir.mkdir(parents=True, exist_ok=True)
    try:
        run(["git", "init", "-q", str(workdir)])
        run(["git", "remote", "add", "origin", sglang["repository"]], cwd=workdir)
        run(["git", "fetch", "-q", "--depth=1", "origin", sglang["head"]], cwd=workdir)
        run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=workdir)
        head = run(["git", "rev-parse", "HEAD"], cwd=workdir)
        upstream_tree = run(["git", "rev-parse", "HEAD^{tree}"], cwd=workdir)
        assert head == sglang["head"], (head, sglang["head"])
        assert upstream_tree == sglang["upstream_tree"], (upstream_tree, sglang["upstream_tree"])
        for patch in patches:
            run(["git", "apply", "--index", str(patch)], cwd=workdir)
        patched_tree = run(["git", "write-tree"], cwd=workdir)
        assert patched_tree == sglang["patched_tree"], (patched_tree, sglang["patched_tree"])
        print("TREE_VERIFY_PASS", head, upstream_tree, patched_tree, flush=True)
    except Exception:
        if not keep:
            subprocess.run(["rm", "-rf", str(workdir)], check=False)
        raise


if __name__ == "__main__":
    main()
