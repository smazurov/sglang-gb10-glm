# sglang-gb10-glm

Patched SGLang images for **GLM-5.3-Flash on DGX Spark** (GB10/SM121, ARM64),
published to `ghcr.io/smazurov/sglang-gb10-glm`. This repo owns *image
maintenance* (upstream SGLang moves, carried patches, wheel overrides) so the
serving cluster does not have to.

**Sister repo:** `the serving org/the serving repo` (private, two-node DGX Spark cluster) — owns
serving (sparkrun recipes), the deployment/upgrade guide
(`stack/GLM-SGLANG.md`), and the accepted image tuple. The two repos share no
code: the serving repo consumes published image tags; this repo references the
checkpoint only by `(repository, revision)` pin plus a sha256 file contract.
No weights or checkpoint files ever live here.

## Current pins (accepted baseline, seeded 2026-09-18)

| | |
|---|---|
| Base | `lmsysorg/sglang@sha256:df8461b8…` (nightly-dev-cu13-20260909-db272201, ARM64) |
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
contracts/        wheel manifest/lock, dependency baseline, processor file contract
gates/            baked into the image: dependency audit, CPU processor gate,
                  GPU kernel gate, normalized-metadata producer
docker/Dockerfile the build: fail-closed pin verification + gates
scripts/          tag.py (content tag), fetch-wheels.py, verify-tree.py
tests/            offline contract tests (free CI tier)
validation/       commands.md — cluster-side T1/T2/T3 validation (ad hoc, GPU needs a Spark)
```

## Validation story

| tier | where | what | gates |
|---|---|---|---|
| T0 | CI, free `ubuntu-24.04-arm` | patch series applies at pin, exact tree hashes, contract tests, tag determinism | merge |
| T0-build | CI, `ubuntu-24.04-arm` + `jlumbroso/free-disk-space` | image build with in-build gates: dep contract, offline hash-locked wheel install, tree identity, compileall, import identity, dependency before/after audits | publish |
| T1 | cluster (ad hoc) | CPU processor gate vs pinned checkpoint | acceptance |
| T2 | cluster (ad hoc, GPU) | CUDA-kernel/numerical/IPC gates | acceptance |
| T3 | the serving repo (ad hoc) | serving acceptance per the the serving repo's guide | accepted tuple |

Publish ≠ accepted: CI publishes candidates; only recorded T1–T3 receipts
change the accepted tuple in the serving repo's guide.

## Flows

- **Upstream SGLang moves / patch rebases:** edit `profile.yaml` pins (+ the
  Dockerfile ARG defaults; tests assert they match). PR → CI rebuilds with a
  new content tag → validate on the cluster → the serving repo adopts by pinning the
  new tag. Nothing auto-follows upstream.
- **Checkpoint updates:** bump `checkpoint.revision` (+ `file_contract`
  hashes). Same flow. Checkpoint-coupled validation proves the new pairing.
- **Content tag:** `scripts/tag.py --sha8` covers profile.yaml, the
  Dockerfile, patches/, gates/, contracts/ — any change ⇒ new tag; unchanged
  inputs skip the build (tag-exists check in CI).

## Building locally

```bash
python3 scripts/fetch-wheels.py          # downloads + verifies pinned wheels
python3 scripts/verify-tree.py           # patch series vs exact tree hashes
docker buildx build -f docker/Dockerfile .   # context = repo root
```

The first CI run is the shakedown: it must reproduce the accepted trees
(tree-hash equality against the accepted images) and record the actual
runner disk layout.
