#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr81-system-one-20261004
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
raw="$shared/data/raw/mech_uspto_31k/data"
old="$shared/data/mech_uspto_31k_full_endpoint_rxnmapper"
output="${MECHET_EQU_OUTPUT:?MECHET_EQU_OUTPUT is required}"
localretro="${MECHET_EQU_LOCALRETRO:?MECHET_EQU_LOCALRETRO is required}"
limit="${MECHET_EQU_LIMIT:-0}"
old_cache_sha=5850784796a9590248e269a676d7f2ffbeb4d908f33bbbf600c0542f22f6412d

case "$limit" in
  0|50) ;;
  *) printf 'unsupported MECHET_EQU_LIMIT=%s\n' "$limit" >&2; exit 2 ;;
esac

cd "$repo"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
runtime=$(mktemp -d /tmp/mechet_equ_proxy_rxnmapper.XXXXXX)
tar -xzf "$shared/artifacts/rxnmapper_pydeps_20260824.tar.gz" -C "$runtime"
export PYTHONPATH="$runtime/rxnmapper_pydeps:$repo/src:$repo"
python=/root/miniconda3/envs/meteor/bin/python

"$python" - <<'PY'
import importlib.metadata as m
import torch
print({'phase': 'mapping_runtime', 'rxnmapper': m.version('rxnmapper'),
       'transformers': m.version('transformers'),
       'cuda_devices': torch.cuda.device_count(),
       'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
      flush=True)
assert m.version('rxnmapper') == '0.4.2'
assert m.version('transformers') == '4.57.1'
assert torch.cuda.device_count() == 1
assert 'A100' in torch.cuda.get_device_name(0).upper()
PY

test -s "$old/manifest.json"
test -s "$old/rxnmapper_cache.jsonl"
test "$(sha256sum "$old/rxnmapper_cache.jsonl" | cut -d ' ' -f1)" = "$old_cache_sha"
test ! -e "$output/manifest.json"
test ! -e "$localretro/manifest.json"
"$python" - "$raw" "$old" "$shared/outputs/agent/system_one_pr81_endpoint_product_min_equ_audit_v2_20261005.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

raw,old,audit=map(Path,sys.argv[1:])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()
assert digest(audit)=='5b1df74f1b20c656c78e8397b2686c78eb9a423e2ef199a60a0ed75076927df4'
m=json.loads((old/'manifest.json').read_text())
a=json.loads(audit.read_text())
for split,stem,count in [('train','train',24959),('valid','val',3120),('test','test',3120)]:
    path=raw/f'{stem}-00000-of-00001.parquet'
    assert digest(path)==m['source_files'][split]['sha256']==a['splits'][split]['raw_sha256']
    assert m['splits'][split]['rows']==a['splits'][split]['counts']['reactions']==count
print({'phase':'frozen_source_gate','rows':{'train':24959,'valid':3120,'test':3120}},flush=True)
PY

printf '[equ-proxy] building limit=%s output=%s\n' "$limit" "$output"
"$python" -u scripts/build_mech_uspto31k_rxnmapper_baseline.py \
  --hf-root "$raw" \
  --product-field rxn_prod_equ \
  --reuse-mapping-cache "$old/rxnmapper_cache.jsonl" \
  --reuse-cache-sha256 "$old_cache_sha" \
  --output-dir "$output" \
  --localretro-dir "$localretro" \
  --batch-size 64 \
  --limit-reactions "$limit"

"$python" - "$output/manifest.json" "$limit" <<'PY'
import json
from pathlib import Path
import sys
m=json.loads(Path(sys.argv[1]).read_text())
limit=int(sys.argv[2])
expected={s: (limit if limit else n) for s,n in
          [('train',24959),('valid',3120),('test',3120)]}
assert m['artifact_type']=='mech_uspto_31k_full_hf_endpoint_rxnmapper_equ_proxy'
assert m['product_source_field']=='rxn_prod_equ'
assert m['executor_filtering'] is False
assert {s:m['splits'][s]['rows'] for s in expected}==expected
print({'phase':'validated_equ_proxy_mapping','rows':expected,
       'mapping_runtime':m['mapping_runtime']},flush=True)
PY
