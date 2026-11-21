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
case "${1:-}" in
    --validate)
        [[ $# == 1 ]]
        exec "${runner[@]}" --validate
        ;;
    --finalize)
        [[ $# == 1 ]]
        exec "$python" -S -m scripts.pr117_confirmation_finalize "$launch" --analysis-code "$launch/code" "${frozen[@]}"
        ;;
    '') [[ $# == 0 ]] ;;
    *) printf 'Usage: %s [--validate|--finalize]\n' "$0" >&2; exit 2 ;;
esac

test ! -e "$launch/results"
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
exec "${runner[@]}" --allocation-receipt "$launch/allocation_receipt.json" --output "$launch/results"
