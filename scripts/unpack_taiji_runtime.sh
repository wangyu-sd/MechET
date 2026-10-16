#!/usr/bin/env bash
# Only base-image gzip/tar/coreutils are needed; no external zstd bootstrap.
set -Eeuo pipefail
if (( $# != 3 )); then
  echo 'usage: unpack_taiji_runtime.sh ARCHIVE_GZ SHA256 DESTINATION' >&2
  exit 2
fi
printf '%s  %s\n' "$2" "$1" | sha256sum --check --strict
tar -xzf "$1" -C "$3"
