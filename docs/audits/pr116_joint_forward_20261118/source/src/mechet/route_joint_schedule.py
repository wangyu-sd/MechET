"""Frozen PR114 schedule and common initial policy validation for PR116 arms."""
import json
from pathlib import Path

from .external_route_data import file_sha, iter_records, signature

VALIDATION_COUNTS = {0: 512, 8: 256, 16: 256, 32: 512, 64: 512}
TRAIN_SHA = '497a6dc0d97b29f29002a1fd16a83fd08243335fa0ad745f7e7dc85c9839acef'
VALID_SHA = '664b05acc2c6ee1e66ba613b911ad54e0de98abba75b7f0ac5a85c8fa615e7f3'
ACTOR_SHA = '2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453'

def validate(cfg):
    spec=cfg['step_budget']
    opts=cfg['joint_forward']
    arms={'R':(0.,0.),'JR':(.1,0.),'J':(.1,.1)}
    if arms.get(spec['arm'])!=(opts['lambda_f'],opts['lambda_d']):
        raise ValueError('joint arm coefficients changed')
    if (cfg['parent_sha256']!=ACTOR_SHA or cfg['products_per_round']!=32
            or cfg['epochs']!=1 or cfg['candidates']!=4 or cfg['validation_targets']!=512
            or cfg['refit']['matched_continuations']!=1 or cfg['refit']['max_forks_per_tree']!=2
            or cfg['acceleration']['refit_lanes']!=dict(credit_forks_per_tree=2,exploration_forks_per_tree=0)
            or opts['margin']!=.25 or opts['beta_d']!=.1 or cfg.get('initial_baseline')
            or cfg.get('acceleration_continuation') or file_sha(spec['roots'])!=TRAIN_SHA
            or file_sha(spec['valid'])!=VALID_SHA):
        raise ValueError('PR116 frozen protocol changed')
    if spec['arm']!='R':
        protocol=opts['teacher_protocol']
        if not protocol or file_sha(opts['teacher_script'])!=protocol['script_sha256']:
            raise ValueError('forward replay teacher identity changed')

def population(cfg,byid):
    spec=cfg['step_budget']
    roots=list(iter_records(spec['roots']))[:64*32]
    val=list(iter_records(spec['valid']))
    if (len(roots)!=2048 or len({r['target'] for r in roots})!=2048 or
            len(val)!=512 or len({r['target'] for r in val})!=512 or
            any(byid[row['task_id']]['target']!=row['target'] for row in roots)
            or {v['target'] for v in val}.intersection(r['target'] for r in roots)):
        raise ValueError('invalid step budget population')
    return [r['task_id'] for r in roots],val

def common_baseline(cfg,rows,root):
    """Import only the completed frozen native G4 runs with their original evidence binding."""
    from .external_route_validation import summarize_validation
    from .route_training_contract import atomic_json
    spec=cfg['step_budget']
    path=Path(spec['baseline_contract'])
    parent=json.loads(path.read_bytes())
    required=('parent_sha256','dataset_manifest_sha256','world_size','gpu_name','inference_backend',
              'products_per_round','validation_protocol','validation_targets','route_endpoint_contract',
              'public_observation_contract','policy_precision','max_context','max_action_tokens')
    if (any(parent[k]!=cfg[k] for k in required) or file_sha(path)!=spec['baseline_sha256']
            or file_sha(parent['clean_valid_file'])!=VALID_SHA):
        raise ValueError('incompatible common step-0 evaluation')
    src=Path(parent['output_dir'])/'validation'/'checkpoint-000000'
    row_ids=[r['task_id'] for r in rows]
    original=json.loads((src/'summary.json').read_bytes())
    outcomes=json.loads((src/'root_outcomes.json').read_bytes())
    computed=summarize_validation(row_ids,outcomes,4)
    binding=dict(contract_sha256=file_sha(path),actor_sha256=ACTOR_SHA,checkpoint_round=0,
                 protocol_sha256=signature(parent['validation_protocol']),task_ids_sha256=signature(row_ids),
                 denominator=512,fractions=[0.])
    if (binding!=original['binding'] or not computed['complete']
            or any(original[k]!=v for k,v in computed.items())):
        raise ValueError('initial actor evidence mismatch')
    dst=Path(root)/'validation'/'checkpoint-000000'
    dst.mkdir(parents=True,exist_ok=True)
    report=dict(original,source_binding=binding,source_directory=str(src),
                binding=dict(binding,contract_sha256=cfg['_sha']))
    atomic_json(dst/'root_outcomes.json',outcomes)
    atomic_json(dst/'summary.json',report)
    best=dst.parent/'best_checkpoint.json'
    if not best.exists():atomic_json(best,report)
