#!/usr/bin/env bash
set -Eeuo pipefail

shared=/aaa/fionafyang/buddy1/whaleywang/MechET
mountpoint -q /aaa/fionafyang/buddy1
source "$shared/configs/runtime/shared_host_storage.env"
launch="$(cd -- "$(dirname -- "$0")" && pwd -P)"
[[ "$launch" == /aaa/fionafyang/buddy1/* ]]
cd "$launch"
sha256sum --check --quiet inputs.sha256

python=/root/miniconda3/envs/meteor/bin/python
contract="$(cat inputs/runtime_contract.path)"
registration="$launch/inputs/confirmation_registration.json"
registration_sha="$(sha256sum "$registration" | cut -d ' ' -f 1)"
panel="$launch/inputs/panel.json"
runtime="$("$python" -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_root"])' "$contract")"
site="$("$python" -S -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="$launch/code:$runtime/runtime:$runtime/rdkit:$site"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 USE_FLAX=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$launch/code"

frozen=(--runtime-contract "$contract" --panel "$panel"
        --confirmation-registration "$registration"
        --confirmation-registration-sha256 "$registration_sha")
runner=("$python" -S -u -m scripts.run_pr117_reference "${frozen[@]}" --workers 6 --backend hf)
trained=()
cbyd_policy="$launch/inputs/cbyd_policy.json"
if [[ -f "$launch/inputs/trained_checkpoint.path" ]]; then
    trained=(--trained-checkpoint "$(cat "$launch/inputs/trained_checkpoint.path")"
             --trained-checkpoint-sha256 "$(cat "$launch/inputs/trained_checkpoint.sha256")")
    runner=("$python" -S -u -m scripts.pr117_actor_transfer "${frozen[@]}" "${trained[@]}")
fi
if [[ -f "$cbyd_policy" ]]; then
    [[ ${#trained[@]} == 0 ]]
    policy_sha="$(sha256sum "$cbyd_policy" | cut -d ' ' -f 1)"
    awk -v digest="$policy_sha" '$1 == digest && $2 == "inputs/cbyd_policy.json" { found=1 } END { exit !found }' "$launch/inputs.sha256"
    runner=("$python" -S -u -m scripts.pr117_cbyd "${frozen[@]}" --workers 6
            --cbyd-policy "$cbyd_policy" --cbyd-policy-sha256 "$policy_sha")
fi
public=()
result_dir="$launch/results"
public_registration="$launch/inputs/public_registration.json"
if [[ -f "$public_registration" ]]; then
    [[ ${#trained[@]} == 0 && ! -f "$cbyd_policy" ]]
    for name in public_registration.json batch_plan.json public_batch_index.txt public_results_root.path; do
        digest="$(sha256sum "$launch/inputs/$name" | cut -d ' ' -f 1)"
        grep -Fxq -- "$digest  inputs/$name" "$launch/inputs.sha256"
    done
    results_root="$(cat "$launch/inputs/public_results_root.path")"
    batch="$(cat "$launch/inputs/public_batch_index.txt")"
    [[ "$batch" =~ ^[1-9][0-9]*$ && "$results_root" == /aaa/fionafyang/buddy1/* ]]
    result_dir="$results_root/$(printf 'batch-%02d' "$batch")"
    public=(--runtime-contract "$contract" --panel "$panel" --public-registration "$public_registration"
            --public-registration-sha256 "$(sha256sum "$public_registration" | cut -d ' ' -f 1)"
            --batch-plan "$launch/inputs/batch_plan.json"
            --batch-plan-sha256 "$(sha256sum "$launch/inputs/batch_plan.json" | cut -d ' ' -f 1)"
            --batch-index "$batch")
fi
case "${1:-}" in
    --validate)
        [[ $# == 1 ]]
        if (( ${#public[@]} )); then
            exec "$python" -S -u -m scripts.pr117_public_batches validate "${public[@]}"
        fi
        exec "${runner[@]}" --validate
        ;;
    --finalize)
        [[ $# == 1 ]]
        if (( ${#public[@]} )); then
            printf '%s\n' 'Use scripts.pr117_public_batches merge with the registered results root.' >&2
            exit 2
        fi
        if [[ -f "$cbyd_policy" ]]; then
            printf '%s\n' 'Use scripts.pr117_cbyd_score with the separate literature evaluator inputs.' >&2
            exit 2
        fi
        if (( ${#trained[@]} )); then
            parent_results="$(cat "$launch/inputs/parent_results.path")"
            if [[ ! -s "$parent_results/screen_result.json" || ! -s "$launch/results/screen_result.json" ]]; then
                printf '%s\n' 'Paired scoring requires the completed parent and trained results.' >&2
                exit 1
            fi
            test ! -e "$launch/paired_actor_results.json"
            staging="$(mktemp -d -- "$launch/.paired-actor-XXXXXXXX")"
            trap 'rm -rf -- "$staging"' EXIT
            "$python" -S -m scripts.pr117_actor_transfer_score "${frozen[@]}" "${trained[@]}" \
                --parent-results "$parent_results" --trained-results "$launch/results" \
                --output "$staging/paired_actor_results.json"
            ln -T -- "$staging/paired_actor_results.json" "$launch/paired_actor_results.json"
            exit 0
        fi
        exec "$python" -S -m scripts.pr117_confirmation_finalize "$launch" --analysis-code "$launch/code" "${frozen[@]}"
        ;;
    '') [[ $# == 0 ]] ;;
    *) printf 'Usage: %s [--validate|--finalize]\n' "$0" >&2; exit 2 ;;
esac

test ! -e "$result_dir"
"$python" -S - "$launch" <<'PY'
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

launch = Path(sys.argv[1])
raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
rows = [tuple(map(str.strip, line.split(",", 2))) for line in raw.splitlines()]
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible:
    rows = [row for row in rows if row[0] == visible or row[1].startswith(visible)]
if len(rows) != 1 or "A100" not in rows[0][2]:
    raise SystemExit("Expected exactly one scheduler-visible A100")
ready = dict(hostname=socket.gethostname(), gpu_uuid=rows[0][1], gpu_model=rows[0][2])
pending = launch / "worker_ready.pending"
with pending.open("x") as out:
    out.write(json.dumps(ready) + "\n")
os.link(pending, launch / "worker_ready.json")
pending.unlink()
print(json.dumps(ready))
PY

for _ in $(seq 1 360); do
    if [[ -s "$launch/allocation_receipt.json" ]]; then break; fi
    sleep 2
done
test -s "$launch/allocation_receipt.json"
"$python" -S - "$launch" <<'PY'
import json
from pathlib import Path
import sys

launch = Path(sys.argv[1])
deployment = json.loads((launch / "inputs/deployment.json").read_bytes())
ready = json.loads((launch / "worker_ready.json").read_bytes())
allocation = json.loads((launch / "allocation_receipt.json").read_bytes())
if (allocation.get("task_flag") != deployment["task_flag"] or not allocation.get("instance_id")
        or any(allocation.get(key) != ready[key] for key in ("hostname", "gpu_uuid"))):
    raise SystemExit("Allocation receipt and registered worker identity disagree")
PY
if (( ${#public[@]} )); then
    exec "$python" -S -u -m scripts.pr117_public_batches run "${public[@]}" --results-root "$results_root" \
        --allocation-receipt "$launch/allocation_receipt.json"
fi
exec "${runner[@]}" --allocation-receipt "$launch/allocation_receipt.json" --output "$launch/results"
