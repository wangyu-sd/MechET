#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dockerfile="$repo_dir/containers/taiji-ailab-cuda118/Dockerfile"
build_context="$repo_dir/containers/taiji-ailab-cuda118"
base_image=${MECHET_AILAB_BASE_IMAGE:-mirrors.tencent.com/whaleywang/metabo@sha256:48ffa174a1f1a4f2518ad4fc327b8bb0e20b1984fee71487dbe2e3fce04bb287}
target_image=${MECHET_AILAB_TARGET_IMAGE:-mirrors.tencent.com/whaleywang/metabo:taiji6-cuda118-compat-v1}

case "$base_image" in
  mirrors.tencent.com/*) ;;
  *)
    echo "[meteor-image][error] base image must come from mirrors.tencent.com: $base_image" >&2
    exit 2
    ;;
esac

case "$target_image" in
  mirrors.tencent.com/*) ;;
  *)
    echo "[meteor-image][error] target image must use mirrors.tencent.com: $target_image" >&2
    exit 3
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "[meteor-image][error] docker is not installed on this build host" >&2
  exit 4
fi

echo "[meteor-image] base=$base_image"
echo "[meteor-image] target=$target_image"
echo "[meteor-image] dockerfile=$dockerfile"

exec docker buildx build \
  --platform linux/amd64 \
  --pull \
  --provenance=false \
  --build-arg "BASE_IMAGE=$base_image" \
  --file "$dockerfile" \
  --tag "$target_image" \
  --push \
  --progress=plain \
  "$build_context"
