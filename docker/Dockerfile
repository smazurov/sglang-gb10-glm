# syntax=docker/dockerfile:1
# sglang-gb10-glm — patched SGLang for GLM-5.3-Flash on DGX Spark (GB10/SM121, ARM64).
#
# All pins are single-sourced in profile.yaml and passed in as --build-arg by
# scripts/build-args.py. The ARGs below are declared WITHOUT defaults so a
# build that omits them fails closed (empty BASE_IMAGE cannot resolve) rather
# than silently building against a stale hardcoded value. tests/test_pins.py
# asserts build-args.py covers exactly the Dockerfile's required ARGs and that
# each value matches profile.yaml. CI passes release metadata
# (IMAGE_SOURCE/IMAGE_REVISION/IMAGE_VERSION) in addition to the pins.
#
# Build (context = repo root, after scripts/fetch-wheels.py):
#   docker buildx build --push \
#     $(python3 scripts/build-args.py) \
#     --build-arg IMAGE_SOURCE=https://github.com/smazurov/sglang-gb10-glm \
#     --build-arg IMAGE_REVISION=$(git rev-parse HEAD) \
#     -f docker/Dockerfile .
#
# The image bakes its own acceptance gates into /opt/glm-gates so any host
# with the image plus the pinned checkpoint snapshot can run validation
# without this repo (see validation/commands.md).

# Base is the GHCR transport mirror of the pinned upstream arm64 manifest
# (byte-identical; digest asserted equal to upstream by mirror-base.yml).
# Values come from profile.yaml via scripts/build-args.py — no defaults here.
ARG BASE_IMAGE
ARG SGLANG_HEAD
ARG SGLANG_UPSTREAM_TREE
ARG SGLANG_PATCHED_TREE
ARG MODEL_REPOSITORY
ARG MODEL_REVISION

FROM ${BASE_IMAGE}

# Re-declared stage-scoped (no defaults): pre-FROM ARGs are only visible to
# FROM interpolation. Empty re-declaration inherits the global value.
ARG SGLANG_HEAD
ARG SGLANG_UPSTREAM_TREE
ARG SGLANG_PATCHED_TREE
ARG MODEL_REPOSITORY
ARG MODEL_REVISION

ENV GLM_OVERLAY=/opt/glm-overlay \
    GLM_GATES=/opt/glm-gates \
    SGLANG_SOURCE=/sgl-workspace/sglang \
    GLM_PY=/opt/sglang/bin/python3

COPY profile.yaml ${GLM_OVERLAY}/profile.yaml
COPY patches/ ${GLM_OVERLAY}/patches/
COPY contracts/dependencies.json ${GLM_OVERLAY}/manifest.json
COPY contracts/requirements.lock ${GLM_OVERLAY}/requirements.lock
COPY contracts/dependency-baseline.json ${GLM_OVERLAY}/baseline.json
COPY contracts/processor-metadata.json ${GLM_GATES}/processor-metadata.json
COPY gates/ ${GLM_GATES}/
COPY wheels/ ${GLM_OVERLAY}/wheels/

# Fail-closed dependency contract BEFORE anything else: the pinned tree
# requires exactly these compiled-package versions; the digest-pinned base
# must already ship them. Also verifies every patch file's sha256 against
# profile.yaml before a single hunk is applied.
RUN set -eux; \
    ${GLM_PY} - <<'PY'
import hashlib
import yaml
from importlib.metadata import version
from pathlib import Path
from packaging.specifiers import SpecifierSet
from packaging.version import Version

profile = yaml.safe_load(Path('/opt/glm-overlay/profile.yaml').read_text())
installed = {d: version(d) for d in profile['expected_deps']}
print('dep inventory (base)', installed, flush=True)
mism = {d: {'installed': installed[d], 'required': w} for d, w in profile['expected_deps'].items()
        if not SpecifierSet(w if any(c in w for c in '<>=!~') else '==' + w).contains(Version(installed[d]), prereleases=True)}
assert not mism, ('dependency version mismatch against the pinned tree requirements', mism)
import torch
print('torch cuda runtime', torch.version.cuda, flush=True)
for p in profile['patches']:
    data = Path('/opt/glm-overlay/patches', p['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == p['sha256'], ('patch sha256 drift', p['file'])
print('patch sha256 contract OK', flush=True)
PY

# Offline wheel contract audit + install. No network, no resolver, no source
# distributions, no dependency builds: validate exact published wheels and the
# immutable inherited-conflict baseline first, then install with --require-hashes.
RUN --network=none set -eux; \
    ${GLM_PY} ${GLM_GATES}/dependency-overlay.py before ${GLM_OVERLAY} && \
    ${GLM_PY} -m pip install --no-index --find-links=${GLM_OVERLAY}/wheels \
        --only-binary=:all: --no-deps --require-hashes \
        -r ${GLM_OVERLAY}/requirements.lock

# Replace the base image's SGLang source with the pinned upstream commit plus
# the overlay patches. Commit, upstream tree, patch shas and the resulting
# patched tree are all asserted; the import-identity check makes it impossible
# to ship an image whose runtime source is not exactly the intended tree.
ARG SGLANG_REPOSITORY
RUN set -eux; \
    git init -q /tmp/sglang-src; \
    cd /tmp/sglang-src; \
    git remote add origin ${SGLANG_REPOSITORY}; \
    git fetch -q --depth=1 origin ${SGLANG_HEAD}; \
    git checkout -q FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${SGLANG_HEAD}"; \
    test "$(git rev-parse 'HEAD^{tree}')" = "${SGLANG_UPSTREAM_TREE}"; \
    git apply --index ${GLM_OVERLAY}/patches/sglang-glm53-integration.patch; \
    git apply --index ${GLM_OVERLAY}/patches/sglang-glm53-spark-mxfp8.patch; \
    git apply --index ${GLM_OVERLAY}/patches/sglang-35609-batch-is-full.patch; \
    git apply --index ${GLM_OVERLAY}/patches/sglang-33391-mm-pickle-storage.patch; \
    git apply --index ${GLM_OVERLAY}/patches/sglang-glm53-ssd-hicache.patch; \
    test "$(git write-tree)" = "${SGLANG_PATCHED_TREE}"; \
    rm -rf ${SGLANG_SOURCE}; \
    mkdir -p ${SGLANG_SOURCE}; \
    cp -a /tmp/sglang-src/. ${SGLANG_SOURCE}/; \
    rm -rf /tmp/sglang-src; \
    cd ${SGLANG_SOURCE}/python; \
    ${GLM_PY} -m compileall -q sglang; \
    ${GLM_PY} - <<'PY'
import sglang
from sglang.srt.layers.quantization import fp8 as _fp8
from sglang.srt.managers import scheduler as _sched
import sglang.kernels.ops.attention.dsa.tilelang_kernel as _tlk

src = '/sgl-workspace/sglang/python/sglang'
mods = [sglang, _fp8, _sched, _tlk]
failures = [(m.__name__, m.__file__) for m in mods if not (m.__file__ or '').startswith(src)]
assert not failures, ('overlay source not on the import path', failures)
print('overlay import identity OK', flush=True)
PY

# Audit actual requirements, including the replaced SGLang tree. Keep receipts;
# drop the wheels (contracts keep their hashes; the image does not need them).
RUN --network=none set -eux; \
    ${GLM_PY} ${GLM_GATES}/dependency-overlay.py after ${GLM_OVERLAY} \
        --source ${SGLANG_SOURCE}/python/pyproject.toml && \
    rm -rf ${GLM_OVERLAY}/wheels

ARG IMAGE_SOURCE=unknown
ARG IMAGE_REVISION=unknown
ARG IMAGE_VERSION=dev
LABEL org.opencontainers.image.title="sglang-gb10-glm" \
      org.opencontainers.image.description="Patched SGLang for GLM-5.3-Flash on DGX Spark GB10 (SM121, ARM64)" \
      org.opencontainers.image.source=${IMAGE_SOURCE} \
      org.opencontainers.image.revision=${IMAGE_REVISION} \
      org.opencontainers.image.version=${IMAGE_VERSION} \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.hardware.target="gb10-sm121" \
      ai.model.repository=${MODEL_REPOSITORY} \
      ai.model.revision=${MODEL_REVISION} \
      ai.sglang.repository=${SGLANG_REPOSITORY} \
      ai.sglang.head=${SGLANG_HEAD} \
      ai.sglang.upstream-tree=${SGLANG_UPSTREAM_TREE} \
      ai.sglang.patched-tree=${SGLANG_PATCHED_TREE} \
      ai.glm-overlay.profile="glm-flash-v2"
