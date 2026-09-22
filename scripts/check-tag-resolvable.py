#!/usr/bin/env python3
"""Check that a registry tag resolves ALL the way down: the manifest and
every child of an index must be retrievable.

Exists because GHCR's package API lists every manifest (including an index's
children) as a "version"; deleting an untagged-looking child orphans the
index and every tag on it, which `docker manifest inspect` cannot detect —
it only sees the index. Pulls then fail with `manifest unknown` while every
shallow check passes. This is the fail-closed gate against that state.
"""
import json
import sys
import urllib.request

ACCEPT = "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json"


def api(url: str, token: str, method: str = "GET") -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {token}", "Accept": ACCEPT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def main() -> int:
    ref = sys.argv[1]  # e.g. ghcr.io/owner/repo:tag
    repo, _, tag = ref.partition(":")
    tok_url = (f"https://ghcr.io/token?scope=repository:{repo.removeprefix('ghcr.io/')}:pull"
               f"&service=ghcr.io")
    with urllib.request.urlopen(tok_url, timeout=30) as r:
        token = json.load(r)["token"]
    status, body = api(f"https://ghcr.io/v2/{repo.removeprefix('ghcr.io/')}/manifests/{tag}", token)
    if status != 200:
        print(f"tag {ref} does not resolve: HTTP {status}", file=sys.stderr)
        return 1
    top = json.loads(body)
    children = top.get("manifests", [])
    for child in children:
        cstatus, _ = api(
            f"https://ghcr.io/v2/{repo.removeprefix('ghcr.io/')}/manifests/{child['digest']}",
            token, method="HEAD")
        if cstatus != 200:
            print(f"index child {child['digest']} does not resolve: HTTP {cstatus}",
                  file=sys.stderr)
            return 1
    print(f"resolvable: {ref} ({len(children)} children checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
