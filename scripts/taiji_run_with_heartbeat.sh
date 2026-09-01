#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# == 0 )); then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

label=${TAIJI_HEARTBEAT_LABEL:-meteor-task}
interval=${TAIJI_HEARTBEAT_SECONDS:-60}
mirror_pid1=${TAIJI_MIRROR_PID1_STDOUT:-0}
if [[ ! "$interval" =~ ^[1-9][0-9]*$ ]] || (( interval > 60 )); then
  echo "TAIJI_HEARTBEAT_SECONDS must be an integer in [1, 60]" >&2
  exit 2
fi
if [[ "$mirror_pid1" != 0 && "$mirror_pid1" != 1 ]]; then
  echo "TAIJI_MIRROR_PID1_STDOUT must be 0 or 1" >&2
  exit 2
fi

emit() {
  local message=$1
  printf '%s\n' "$message"
  if [[ "$mirror_pid1" == 1 && -e /proc/1/fd/1 && -w /proc/1/fd/1 ]]; then
    # Taiji may rotate PID 1's log pipe during a long job. A transient mirror
    # failure must not terminate the heartbeat subshell under `set -e`.
    printf '%s\n' "$message" > /proc/1/fd/1 || true
  fi
}

export PYTHONUNBUFFERED=1
emit "[TAIJI][launch] label=$label pid=$$ time=$(date --iso-8601=seconds) command=$*"

"$@" &
child_pid=$!

forward_signal() {
  local signal=$1
  emit "[TAIJI][signal] label=$label signal=$signal child_pid=$child_pid time=$(date --iso-8601=seconds)"
  kill "-$signal" "$child_pid" 2>/dev/null || kill -s "$signal" "$child_pid" 2>/dev/null || true
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

(
  while kill -0 "$child_pid" 2>/dev/null; do
    emit "[TAIJI][heartbeat] label=$label child_pid=$child_pid time=$(date --iso-8601=seconds)"
    sleep "$interval"
  done
) &
heartbeat_pid=$!

set +e
wait "$child_pid"
status=$?
set -e
kill "$heartbeat_pid" 2>/dev/null || true
wait "$heartbeat_pid" 2>/dev/null || true
emit "[TAIJI][exit] label=$label child_pid=$child_pid status=$status time=$(date --iso-8601=seconds)"
exit "$status"
