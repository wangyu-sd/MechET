"""Read-only original256-episode reward replay and centered-advantage ledger.

This is an artifact audit, not an executor re-run or a chemistry performance
evaluation. Hidden references stay inside the private shared evidence folder.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib,json,math,os
from pathlib import Path
import sys,time,types
import tempfile

ROOT=Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
OUT=Path('/aaa/fionafyang/buddy1/whaleywang/datasets/mechet_route_rl_efficiency_audit_20261111')
SUB=ROOT/'outputs/submissions/external_route_stage2_dr_grpo_20261106_03'
RUN=ROOT/'outputs/agent/external_stage2_target_only_route_dr_grpo_v2_20261106_03'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic(path,data):
 descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.writing',dir=path.parent)
 temporary=Path(name)
 try:
  with os.fdopen(descriptor,'w') as writer:
   json.dump(data,writer,indent=2);writer.write('\n');writer.flush();os.fsync(writer.fileno())
  os.link(temporary,path)  # Atomic no-clobber publication, even with competing writers.
 finally:temporary.unlink()

def main(output):
 output=Path(output).resolve()
 if not output.is_relative_to(Path('/aaa/fionafyang/buddy1')):raise ValueError('SHARED_STORAGE_REQUIRED')
 if output.exists():raise FileExistsError(output)
 output.parent.mkdir(parents=True,exist_ok=True)
 os.umask(0o077)
 cfg=json.loads((SUB/'contract.json').read_text());code=Path(cfg['code_root'])
 paths={str(code/n):v for n,v in cfg['code_files_sha256'].items()}
 for p,v in cfg['files_sha256'].items():
  if p.endswith(('references.sqlite','stock.txt','manifest.json')):paths[p]=v
 def verify(item):
  p,w=item
  if sha(p)!=w:raise ValueError('FROZEN_BASELINE_INPUT_CHANGED:'+p)
 with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(verify,paths.items()))
 sys.path[:]=[p for p in sys.path if p]+[str(code),str(code/'src')]
 package=types.ModuleType('mechet');package.__path__=[str(code/'src/mechet')];sys.modules['mechet']=package
 from mechet.external_route_rl import HiddenRouteReward,route_group_advantages
 from mechet.route_policy_objective import policy_sequences
 stock=set((Path(cfg['dataset'])/'stock.txt').read_text().splitlines())
 summary=dict(scope='original_first_round_saved256_reward_replay_and_signal_ledger',
  status='RUNNING',source_contract=str(SUB/'contract.json'),contract_sha256=sha(SUB/'contract.json'),
  auditor_script_sha256=sha(__file__),
  source_root=str(code),source_files_sha256=paths,actor_sha256=cfg['parent_sha256'],
  chemistry_executions=0,optimizer_updates=0,group_ledger=[],mismatches=[],
  label_interpretation='Original frozen supervision labels; reference support is not experimental feasibility',
  executor_validity='not independently reexecuted; original stored reports only')
 statuses=Counter();histogram=Counter();counts=Counter();artifacts={}
 with HiddenRouteReward(Path(cfg['dataset'])/'references.sqlite',stock,
                        partial_weight=cfg['partial_reward_weight']) as oracle:
  for rank in range(4):
   directory=RUN/'round-000000'/('rank%02d'%rank)
   path=directory/'rewarded_episodes.json';binding_path=directory/'binding.json'
   artifacts[str(path)]=sha(path);artifacts[str(binding_path)]=sha(binding_path)
   binding=json.loads(binding_path.read_text());groups=json.loads(path.read_text())
   if binding['contract_sha256']!=summary['contract_sha256'] or len(groups)!=16:
    raise ValueError('ORIGINAL_RANK_BINDING_OR_DENOMINATOR_CHANGED')
   for index,group in enumerate(groups):
    if len(group)!=4 or any(r['task_id']!=binding['task_ids'][index] or
       r['behavior_version']!=binding['behavior_version'] for r in group):
     raise ValueError('ORIGINAL_G4_ROOT_IDENTITY_CHANGED')
    rewards=[r['reward'] for r in group];advantages=route_group_advantages(rewards)
    informative=advantages is not None and any(a!=0 for a in advantages)
    generated=policy_tokens=nonzero_tokens=0
    for position,row in enumerate(group):
     fresh=oracle.score(row['task_id'],row)
     if fresh!=row['reward_evidence'] or fresh['reward']!=row['reward']:
      summary['mismatches'].append(dict(rank=rank,group=index,episode=row['episode_id'],
       source=str(path),recorded=row['reward_evidence'],reconstructed=fresh))
     tokens=sum(len(r.get('generated_token_ids',[])) for r in row['records'])
     if tokens!=row['generated_tokens']:raise ValueError('GENERATED_TOKEN_TOTAL_MISMATCH:'+row['episode_id'])
     generated+=tokens;counts['episodes']+=1;statuses[row['status']]+=1
     counts['model_decision_records']+=len(row['records']);counts['completed_reactions']+=len(row['reactions'])
     counts['stock_solved']+=row['status']=='stock_solved'
     counts['reference_supported_closed']+=fresh['verified_stock_closed']
     counts['unsupported_closed']+=fresh['reward_status']=='unsupported_closed'
     histogram[str(row['reward'])]+=1
     if advantages is not None:
      n=sum(len(r) for _,r,_ in policy_sequences(row));policy_tokens+=n
      if advantages[position]!=0:nonzero_tokens+=n
    counts['groups']+=1;counts['informative_groups']+=informative
    counts['censored_groups']+=advantages is None
    counts['generated_tokens']+=generated;counts['policy_tokens']+=policy_tokens
    counts['policy_tokens_in_informative_groups']+=policy_tokens if informative else 0
    counts['policy_tokens_with_nonzero_advantage']+=nonzero_tokens
    summary['group_ledger'].append(dict(rank=rank,task_id=group[0]['task_id'],G=4,
     rewards=rewards,advantages=advantages,usable=advantages is not None,
     informative=informative,generated_tokens=generated,policy_tokens=policy_tokens,
     nonzero_advantage_policy_tokens=nonzero_tokens,statuses=[r['status'] for r in group]))
 summary.update(counts=dict(counts),terminal_statuses=dict(statuses),reward_histogram=dict(histogram),
  raw_artifacts_sha256=artifacts,completed_unix=time.time())
 witness_path=SUB/'first_optimizer_update_witness.json';witness_raw=witness_path.read_bytes()
 artifacts[str(witness_path)]=hashlib.sha256(witness_raw).hexdigest()
 witness=json.loads(witness_raw)
 expected={'targets':'groups','sampled_route_episodes':'episodes','model_decision_requests':'model_decision_records',
  'completed_reaction_programs':'completed_reactions','native_stock_solved':'stock_solved',
  'verified_stock_solved':'reference_supported_closed','effective_groups':'informative_groups','censored_groups':'censored_groups'}
 summary['witness_agreement']={k:counts[v]==witness[k] for k,v in expected.items()}
 summary['informative_group_fraction']=counts['informative_groups']/counts['groups']
 summary['nonzero_advantage_policy_token_fraction']=counts['policy_tokens_with_nonzero_advantage']/counts['policy_tokens']
 summary['cold_first_round_sampling_wall_seconds']=witness['sampling_wall_seconds']
 summary['steady_state_throughput']='NOT_MEASURED'
 summary['status']='PASS' if not summary['mismatches'] and all(summary['witness_agreement'].values()) else 'FAIL'
 atomic(output,summary)
 print(json.dumps({k:v for k,v in summary.items() if k not in ('source_files_sha256','group_ledger','raw_artifacts_sha256','mismatches')}),flush=True)
 if summary['status']!='PASS':raise ValueError('ORIGINAL_REWARD_REPLAY_MISMATCH')

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 main(parser.parse_args().output)
