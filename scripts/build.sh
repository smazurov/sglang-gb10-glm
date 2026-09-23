#!/usr/bin/env bash
# Local build wrapper: single-sources every pin from profile.yaml via
# scripts/build-args.py, mirroring what CI does. Run from the repo root after
# scripts/fetch-wheels.py has populated wheels/.
#
#   scripts/build.sh                 # build locally, load into docker (tag: candidate)
#   scripts/build.sh --push          # build and push
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-ghcr.io/smazurov/sglang-gb10-glm}"
PY="${PYTHON:-python3}"

# Fail closed if the pin coverage ever drifts from the Dockerfile's required ARGs.
"$PY" scripts/build-args.py --check

LOAD_OR_PUSH="--load"
[ "${1:-}" = "--push" ] && LOAD_OR_PUSH="--push"

exec docker buildx build \
  --platform linux/arm64 \
  $LOAD_OR_PUSH -t "$IMAGE:candidate" \
  -f docker/Dockerfile \
  $("$PY" scripts/build-args.py) \
  --build-arg IMAGE_SOURCE="https://github.com/smazurov/sglang-gb10-glm" \
  --build-arg IMAGE_REVISION="$(git rev-parse HEAD)" \
  --build-arg IMAGE_VERSION="local" \
  .
