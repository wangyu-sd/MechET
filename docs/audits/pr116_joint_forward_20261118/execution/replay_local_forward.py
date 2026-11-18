"""Run the production scorer on saved TRAIN receipts with the local FlowER 30k checkpoint."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

from mechet.route_joint_forward import group_preferences
from mechet.route_joint_phase import attach_joint_evidence

root=Path(__file__).resolve().parents[3]
diag=Path(__file__).resolve().parent
local=root/'artifacts/pr116_joint_20261118/local_forward_30k'
opts=json.loads((local/'bound_runtime.json').read_bytes())
opts['teacher_protocol']['normalization']='full_mapped_dense_H_project_stereo_mask_isotopes_v2'
opts.update(lambda_f=.1,lambda_d=.1,teacher_local_rank=0,cache_dir=str(local/'projected_replay_cache'))
audit=json.loads((diag/'inventory/saved_receipt_audit.json').read_bytes())
source=(root/audit['source_path']).read_bytes()
if hashlib.sha256(source).hexdigest()!=audit['source_sha256']:
    raise ValueError('saved TRAIN receipt changed')
base=json.loads(source)
results=[]
for mode in ('first','replay'):
    group=deepcopy(base)
    stats=attach_joint_evidence([group],dict(joint_forward=opts))
    pairs=group_preferences(group,.25)
    result=dict(mode=mode,stats=stats,pairs=len(pairs))
    results.append(result)
    print(json.dumps(result),flush=True)
if results[-1]['stats']['global_executed_queries']!=0:
    raise ValueError('replay executed a query instead of reusing the score')
(local/'production_replay_projected.json').write_text(json.dumps(dict(
    source_sha256=audit['source_sha256'],protocol=opts['teacher_protocol'],rounds=results),indent=2)+'\n')
