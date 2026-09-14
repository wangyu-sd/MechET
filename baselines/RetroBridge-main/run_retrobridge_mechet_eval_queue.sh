#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/estar/pxy/mechet/baselines/RetroBridge-main

"$ROOT/run_retrobridge_mechet_eval.sh" uspto31k
"$ROOT/run_retrobridge_mechet_eval.sh" flower
