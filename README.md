# sglang-gb10-glm

Patched SGLang image for serving **GLM-5.3-Flash** across **two DGX Sparks**
(GB10/SM121, ARM64, `--tp 2`), published to
`ghcr.io/smazurov/sglang-gb10-glm`. This repo owns *image maintenance* (the
pinned dev base, pinned upstream SGLang + carried patches, wheel overrides)
so the serving cluster does not have to. It shares no code with the serving
side: it publishes image tags; the checkpoint is referenced only by
`(repository, revision)` pin plus a sha256 file contract, with the six
MIT-licensed interface files vendored (`checkpoint/`) so CI can run the
processor gate without weights. Cluster credentials, hostnames, and topology
are deliberately absent from this public repo.

## Current pins (accepted baseline, seeded 2026-09-18)

| | |
|---|---|
| Base | `ghcr.io/smazurov/sglang-base@sha256:df8461b8…` — byte-identical transport mirror of `docker.io/lmsysorg/sglang` (nightly-dev-cu13-20260909-db272201, ARM64; mirror-base.yml asserts manifest sha256 equality) |
| SGLang head | `d6fabb74b45d4fb92796cfb6740810b4811b018e` |
| Upstream tree | `ece17616d5a531bd37cfe476de07a30e6ada0ecb` |
| Patched tree | `4c582984e40ace4a203e369237954df26368bff0` |
| Checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark@53e77dbb…` (reference only) |
| Wheels | transformers 5.16.1, tokenizers 0.23.1, torchcodec 0.16.0, nvshmem4py-cu13 0.3.1, cython 3.3.0 |

All pins and their provenance: [`profile.yaml`](profile.yaml).

## Layout

```
profile.yaml      single source of truth: every pin + provenance
patches/          the five carried patches (sha256-pinned, applied in order)
checkpoint/       the checkpoint's six MIT-licensed interface files (CI gate inputs; no weights)
contracts/        wheel manifest/lock, dependency baseline, processor file contract
gates/            baked into the image: dependency audit, CPU processor gate,
                  GPU kernel gate, normalized-metadata producer
docker/Dockerfile the build: fail-closed pin verification + gates
scripts/          tag.py (content tag), fetch-wheels.py, verify-tree.py
tests/            offline contract tests (free CI tier)
validation/       commands.md — cluster-side pre-test drift check + T2/T3 (ad hoc, GPU needs a Spark)
```

## Validation story

| tier | where | what | gates |
|---|---|---|---|
| T0 | CI, free `ubuntu-24.04-arm` | patch series applies at pin, exact tree hashes, contract tests, tag determinism | merge |
| build | CI, `ubuntu-24.04-arm` + `jlumbroso/free-disk-space` | image built LOCALLY with in-build gates: dep contract, offline hash-locked wheel install, tree identity, compileall, import identity, dependency before/after audits, then the CPU processor gate vs the vendored checkpoint interface files — push happens only after the gate passes | publish |
| T2 | cluster (ad hoc, GPU) | CUDA-kernel/numerical/IPC gates | acceptance |
| T3 | serving repo (ad hoc, over SSH) | serving acceptance per the serving repo's guide | accepted tuple |

Publish ≠ accepted: CI publishes candidates after build + pre-test; only
recorded T2/T3 receipts change the accepted tuple in the serving repo's
guide. The gate scripts are baked into the image, so the cluster commands
need no code from this repo. Cluster validation is run ad hoc by the
operator over SSH on the cluster head — never in CI (no GPUs, no cluster
secrets here) — using only `validation/commands.md` and the published image.

## Flows

- **Upstream SGLang moves / patch rebases:** edit `profile.yaml` pins (+ the
  Dockerfile ARG defaults; tests assert they match). PR → CI rebuilds with a
  new content tag → validate on the cluster (ad hoc, over SSH) → the serving
  repo adopts by pinning the new tag. Nothing auto-follows upstream.
- **Checkpoint updates:** bump `checkpoint.revision` (+ `file_contract`
  hashes). Same flow. Checkpoint-coupled validation proves the new pairing.
- **Content tag:** `scripts/tag.py --sha8` covers profile.yaml, the
  Dockerfile, patches/, gates/, contracts/ — any change ⇒ new tag; unchanged
  inputs skip the build (tag-exists check in CI).
- **Base-pin changes** additionally require re-running the `mirror-base`
  workflow (dispatch-only) before a build can succeed: it re-mirrors the new
  base byte-identically into GHCR (`skopeo copy --preserve-digests`, with a
  sha256-of-raw-manifest assert) and is what keeps both legs fast — pulls
  come from GHCR (same cloud as the runners) and pushes cross-repo-mount the
  shared base blobs (~11 s observed vs a full 14 GiB re-upload).
- **The push is gated:** the build job builds `--load` (unpushed), runs the
  CPU processor pre-test against the vendored checkpoint files, and pushes by
  digest only on `GLM_PROCESSOR_PASS`; named tags assemble after that. GHCR
  never garbage-collects untagged manifests, so a failed candidate leaves no
  registry residue.

## Building locally

```bash
python3 scripts/fetch-wheels.py          # downloads + verifies pinned wheels
python3 scripts/verify-tree.py           # patch series vs exact tree hashes
docker buildx build -f docker/Dockerfile .   # context = repo root
```

Current candidate: `glm-flash-v2-8bb31a67` (manifest
`sha256:83c67f46…`) — CI pre-test `GLM_PROCESSOR_PASS` receipt recorded
2026-09-22; cluster T2/T3 receipts pending.