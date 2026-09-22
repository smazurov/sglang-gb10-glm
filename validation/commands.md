# Cluster validation — how to test a published candidate

CI (public infra) validates image-internal properties and the CPU processor
pre-test (run inside the build job against the local image, before anything
is pushed), then publishes the candidate. Nothing unvalidated lands in the
registry; a `workflow_dispatch` with `force_t1` re-runs the gate against an
already-published stable tag without rebuilding. The cluster session is for
what CI structurally cannot do: the GPU gates, the serving acceptance, and
the pre-test re-run against the real snapshot as a drift check. The image is
self-validating: its gates are baked into `/opt/glm-gates`, so these commands
need the image and the cluster's HF cache, not this repository.

Run on the head unless stated otherwise. Record every receipt (command
output, image digest, checkpoint revision) in the the serving repo's guide's accepted
tuple or as a comment on the pin-change PR that produced the candidate.

Set up once per session:

```bash
IMAGE=ghcr.io/smazurov/sglang-gb10-glm:<tag>          # candidate tag
docker pull "$IMAGE"                                 # head has internet; nodes get it via the fabric registry
```

## pre-test — CPU processor gate (runs in CI now)

CI runs this gate against the vendored checkpoint interface files
(`checkpoint/`, MIT — no weights). The same command re-run on the cluster
against the **real snapshot** is a drift check: it proves the head's
cached files still match the contract the CI-validated candidate was
gated on.

Verifies the pinned checkpoint's small files against the baked sha256
contract, then runs real `AutoProcessor`/`get_processor` paths, pixel/grid
accounting, save/reload, and the tokenizer-only negative control. Fails
closed on any checkpoint metadata drift.

```bash
SNAP=/srv/hf-cache/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4-Spark/snapshots/53e77dbb04fa9dd68725daa899ec92eabf8a872c
CFG=/srv/hf-cache/runtime-configs/modelopt-flat-config.json

# Generate the normalized ModelOpt metadata from the checkpoint config (the
# producer is baked into the image — the cluster runs no repo code for this).
mkdir -p "$(dirname "$CFG")"
docker run --rm -i "$IMAGE" /opt/sglang/bin/python3 /opt/glm-gates/normalize_modelopt.py \
  < "$SNAP/config.json" > "$CFG"

# Offline gate: runc, no network, 4 GiB / 4 CPUs, checkpoint RO, normalized
# config mounted over the snapshot's config.json.
docker run --rm --pull=never --runtime=runc --network=none \
  --memory=4g --cpus=4 \
  --entrypoint /opt/sglang/bin/python3 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /srv/hf-cache:/cache/huggingface:ro \
  -v "$CFG:$SNAP/config.json:ro" \
  "$IMAGE" /opt/glm-gates/gate-processor.py \
  /cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4-Spark/snapshots/53e77dbb04fa9dd68725daa899ec92eabf8a872c
```

Expected token: `GLM_PROCESSOR_PASS`. Record the printed shapes/hash map.

## T2 — GPU / CUDA-kernel gate

Requires a GB10. Compiles and runs the fp8 DSA partial kernel at the deployed
shapes (the zero-rope-tail case that would otherwise crash ~8 minutes into a
cold boot), plus numerical/routes/raw-layout and native IPC checks and the
KDA pytest suite (zero collection is a failure).

```bash
docker run --rm --gpus all "$IMAGE" \
  /opt/sglang/bin/python3 /opt/glm-gates/gate-pr33391.py
```

Expected token: `PASS` in the final output line. Never run T2 while the
serving pair is resident; drain both ranks first (see the the serving repo's guide).

## T3 — serving acceptance

Run through the the serving repo sparkrun lane (recipe pins the image; `--tp 2 -H
the cluster head node,the cluster worker node`), then follow the the serving repo's guide's runtime-acceptance
section: text/multimodal round-trips with exact assertions, capacity from
`/get_server_info`, memory observation during working vision. That guide owns
the accepted tuple; this repo only ever publishes candidates.

## Adoption

A candidate is adopted only after T1+T2+T3 receipts exist for it. Then the
the serving repo's guide's accepted tuple records: image tag + digest, checkpoint
revision, and the receipts. Updating that tuple is the only way the serving repo
changes behavior; a GHCR publish alone proves nothing about serving.
