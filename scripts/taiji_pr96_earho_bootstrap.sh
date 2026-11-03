#!/usr/bin/env bash
set -Eeuo pipefail
if (( $# != 3 )); then
  echo 'usage: earho-bootstrap ENTRY_ARCHIVE ENTRY_SHA CONTRACT' >&2
  exit 2
fi
entry_archive=$1
entry_sha=$2
contract=$3
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
earho_node=$(mktemp -d "$TMPDIR/meteor-pr96-earho.XXXXXX")
mkdir "$earho_node/code" "$earho_node/entry" "$earho_node/runtime" "$earho_node/rdkit"
python - "$entry_archive" "$entry_sha" "$contract" <<'PY'
import hashlib,json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[3]).read_text())
paths={sys.argv[1]:sys.argv[2],c['reference_code_archive']:c['reference_code_archive_sha256'],
       c['runtime_archive']:c['runtime_archive_sha256'],**c['files_sha256']}
for name,expected in paths.items():
    h=hashlib.sha256()
    with Path(name).open('rb') as reader:
        for block in iter(lambda:reader.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('EARHO_INPUT_HASH_CHANGED:'+name)
    print(json.dumps({'event':'earho-input-verified','path':name}),flush=True)
PY
export MECHET_EARHO_VERIFIED_CONTRACT_SHA256=$(sha256sum "$contract" | cut -d' ' -f1)
readarray -t earho_archives < <(python - "$contract" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
print(c['reference_code_archive']);print(c['runtime_archive']);print(c['world_size'])
PY
)
tar -xzf "${earho_archives[0]}" -C "$earho_node/code"
tar -xzf "${earho_archives[1]}" -C "$earho_node/runtime"
tar -xzf "$entry_archive" -C "$earho_node/entry"
python -m pip install --no-deps --no-index --target "$earho_node/rdkit" /aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
export MECHET_PR96_REFERENCE_CODE_ROOT="$earho_node/code"
export PYTHONPATH="$earho_node/runtime:$earho_node/rdkit:$earho_node/code/src:$earho_node/code"
export HF_HUB_CACHE="$earho_node/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - "$contract" <<'PY'
import json,os,shutil,sys,torch,transformers,peft,triton
from pathlib import Path
from mechet.pr96_eval_runtime import sha
from mechet.qwen35_protocol import OFFICIAL_REVISION,validate_runtime_versions
c=json.loads(Path(sys.argv[1]).read_text())
validate_runtime_versions(torch_version=torch.__version__,cuda_version=torch.version.cuda,triton_version=triton.__version__)
base=json.loads(Path(c['base_manifest']).read_text())
if base['revision']!=OFFICIAL_REVISION:raise ValueError('EARHO_BASE_REVISION_CHANGED')
local=Path(os.environ['HF_HUB_CACHE'])/'models--Qwen--Qwen3.5-0.8B'/'snapshots'/OFFICIAL_REVISION
local.mkdir(parents=True)
for name,record in base['files'].items():
    shutil.copy2(Path(base['snapshot'])/name,local/name)
    if sha(local/name)!=record['sha256']:raise ValueError('EARHO_BASE_FILE_CHANGED:'+name)
    print(json.dumps({'event':'earho-base-staged','file':name}),flush=True)
print(json.dumps({'event':'earho-runtime-ready','torch':torch.__version__,'cuda':torch.version.cuda,
    'transformers':transformers.__version__,'peft':peft.__version__,'triton':triton.__version__,
    'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}),flush=True)
PY
cd "$earho_node/code"
exec python -u -m torch.distributed.run --standalone --nproc_per_node="${earho_archives[2]}" \
  "$earho_node/entry/scripts/train_pr96_earho.py" --contract "$contract"
