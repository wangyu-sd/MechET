#!/usr/bin/env bash
set -Eeuo pipefail
if (( $# != 3 )); then
  echo 'usage: test-bootstrap ENTRY_ARCHIVE ENTRY_SHA TEST_CONTRACT' >&2
  exit 2
fi
entry_archive=$1
entry_sha=$2
contract=$3
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
pr96_test_node=$(mktemp -d "$TMPDIR/meteor-pr96-test.XXXXXX")
export PR96_NODE_CACHE="$pr96_test_node"
mkdir "$pr96_test_node/code" "$pr96_test_node/entry" "$pr96_test_node/runtime" "$pr96_test_node/rdkit"
python - "$entry_archive" "$entry_sha" "$contract" <<'PY'
import hashlib,json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[3]).read_text())
paths={sys.argv[1]:sys.argv[2],c['reference_code_archive']:c['reference_code_archive_sha256'],
       c['runtime_archive']:c['runtime_archive_sha256'],**c['files_sha256']}
for p,digest in paths.items():
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=digest:raise ValueError('PR96_TEST_ARCHIVE_OR_INPUT_HASH_CHANGED:'+p)
    print(json.dumps({'phase':'test-input-hash-verified','path':p,'sha256':digest}),flush=True)
PY
readarray -t pr96_test_archives < <(python - "$contract" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
print(c['reference_code_archive']);print(c['runtime_archive'])
PY
)
tar -xzf "${pr96_test_archives[0]}" -C "$pr96_test_node/code"
tar -xzf "${pr96_test_archives[1]}" -C "$pr96_test_node/runtime"
tar -xzf "$entry_archive" -C "$pr96_test_node/entry"
python -m pip install --no-deps --no-index --target "$pr96_test_node/rdkit" /aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
export MECHET_PR96_REFERENCE_CODE_ROOT="$pr96_test_node/code"
export PYTHONPATH="$pr96_test_node/runtime:$pr96_test_node/rdkit:$pr96_test_node/code/src:$pr96_test_node/code"
export HF_HUB_CACHE="$pr96_test_node/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - "$contract" <<'PY'
import json,os,shutil,sys,torch,transformers,peft,triton
from pathlib import Path
from mechet.pr96_eval_runtime import sha
from mechet.qwen35_protocol import OFFICIAL_REVISION,validate_runtime_versions
c=json.loads(Path(sys.argv[1]).read_text())
validate_runtime_versions(torch_version=torch.__version__,cuda_version=torch.version.cuda,triton_version=triton.__version__)
base=json.loads(Path(c['base_manifest']).read_text())
if base['revision']!=OFFICIAL_REVISION:raise ValueError('PR96_TEST_BASE_REVISION_CHANGED')
local=Path(os.environ['HF_HUB_CACHE'])/'models--Qwen--Qwen3.5-0.8B'/'snapshots'/OFFICIAL_REVISION
local.mkdir(parents=True)
for name,record in base['files'].items():
    dest=local/name
    shutil.copy2(Path(base['snapshot'])/name,dest)
    if sha(dest)!=record['sha256']:raise ValueError('PR96_TEST_BASE_FILE_CHANGED:'+name)
    print(json.dumps({'phase':'test-base-staged','file':name}),flush=True)
print(json.dumps({'phase':'test-runtime-ready','torch':torch.__version__,'cuda':torch.version.cuda,
    'transformers':transformers.__version__,'peft':peft.__version__,'triton':triton.__version__,
    'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}),flush=True)
PY
cd "$pr96_test_node/code"
exec python -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$pr96_test_node/entry/scripts/evaluate_pr96_test.py" --contract "$contract"
