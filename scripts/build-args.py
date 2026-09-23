#!/usr/bin/env python3
"""Emit --build-arg flags for every pin in profile.yaml.

profile.yaml is the single source of truth for the image recipe. The
Dockerfile declares these ARGs with NO defaults so a build that does not
supply them fails closed (empty BASE_IMAGE cannot resolve) rather than
silently building against a stale hardcoded default. Both CI and local
builds go through this helper, so there is exactly one place a pin lives.

Usage:
    docker buildx build $(python3 scripts/build-args.py) -f docker/Dockerfile .
    python3 scripts/build-args.py --check   # verify coverage vs the Dockerfile
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# profile.yaml location -> Dockerfile ARG name. Every entry here is a pin that
# the Dockerfile declares without a default and therefore MUST be supplied.
PIN_MAP = {
    "BASE_IMAGE": ("base", "image"),
    "SGLANG_HEAD": ("sglang", "head"),
    "SGLANG_UPSTREAM_TREE": ("sglang", "upstream_tree"),
    "SGLANG_PATCHED_TREE": ("sglang", "patched_tree"),
    "SGLANG_REPOSITORY": ("sglang", "repository"),
    "MODEL_REPOSITORY": ("checkpoint", "repository"),
    "MODEL_REVISION": ("checkpoint", "revision"),
}


def pins(profile_path: Path = ROOT / "profile.yaml") -> dict[str, str]:
    profile = yaml.safe_load(profile_path.read_text())
    out: dict[str, str] = {}
    for arg, keys in PIN_MAP.items():
        node = profile
        for k in keys:
            assert isinstance(node, dict) and k in node, f"profile.yaml missing {'.'.join(keys)} (for {arg})"
            node = node[k]
        assert isinstance(node, str) and node, f"profile.yaml {'.'.join(keys)} is empty (for {arg})"
        out[arg] = node
    return out


def dockerfile_required_args(dockerfile: Path = ROOT / "docker" / "Dockerfile") -> set[str]:
    """ARG names declared with no default — these must be passed at build time."""
    required = set()
    for line in dockerfile.read_text().splitlines():
        m = re.match(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", line)
        if m:
            required.add(m.group(1))
    return required


def main(argv: list[str]) -> int:
    if "--check" in argv:
        provided = set(pins())
        required = dockerfile_required_args()
        missing = required - provided
        extra = provided - required
        assert not missing, f"Dockerfile requires ARGs not emitted by build-args.py: {sorted(missing)}"
        assert not extra, f"build-args.py emits ARGs no longer required by the Dockerfile: {sorted(extra)}"
        print(f"BUILD_ARGS_CHECK_OK {len(provided)} pins cover all required Dockerfile ARGs", flush=True)
        return 0
    print(" ".join(f"--build-arg {k}={v}" for k, v in pins().items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
