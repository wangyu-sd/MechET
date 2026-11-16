"""Replay saved TRAIN actions across the material boundary, without generation.

This is deterministic adapter verification, not an estimate of policy quality.
All private paths are supplied by the operator; no hidden reference is provided
to RouteEpisode or to the material resolver.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from mechet.external_route_rl import TargetOnlyRouteEpisode, HiddenRouteReward
from mechet.route_materials import PROTIC_MATERIAL
from mechet.route_training_contract import atomic_json


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def replay(contract, cases, output):
    cfg=json.loads(Path(contract).read_text())
    dataset=Path(cfg['dataset'])
    stock=set((dataset/'stock.txt').read_text().splitlines())
    examples=[x for x in json.loads(Path(cases).read_text())
              if x['kind']=='contracted_reference_edge']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    results=[]
    for index,case in enumerate(examples):
        source=Path(case['source'])
        old=json.loads(source.read_text())[case['group_index']][case['episode_index']]
        local=dict(cfg,behavior_version=old['behavior_version'],stock_material_contract=PROTIC_MATERIAL)
        ep=TargetOnlyRouteEpisode(local,old['public_root'],stock,old['episode_id'],old['seed'])
        # The archived audit numbers decisions from one; request a0 is first.
        prefix=old['records'][:case['first_finish']['decision']]
        for receipt in prefix:
            # Every original observation, action, acceptance/error and mapped
            # before/after state must agree. Only stock handoff may change.
            ep.accept(deepcopy(receipt))
        edge=next(r for r in ep.reactions if r['occurrence']==case['first_finish']['occurrence'])
        old_edge=next(r for r in old['reactions'] if r['occurrence']==edge['occurrence'])
        assert edge['final_state']==old_edge['final_state']
        assert edge['actions']==old_edge['actions']
        aliases=[m for m in edge['endpoint_materials'] if m['rule']!='exact']
        result=dict(case_index=index,task_id=case['task_id'],source_sha256=sha(source),
            original_prefix_decisions=len(prefix),mapped_states_and_actions_unchanged=True,
            endpoint_materials=edge['endpoint_materials'],aliases=aliases,
            new_status=ep.status,new_pending_products=([ep.current['product']] if ep.current else [])+
                [p['product'] for p in ep.pending],original_total_decisions=len(old['records']))
        if ep.status is not None:
            report=ep.finish_report();report['task_id']=case['task_id']
            with HiddenRouteReward(dataset/'references.sqlite',stock,material_contract=PROTIC_MATERIAL) as oracle:
                result['new_reward_evidence']=oracle.score(case['task_id'],report,include_progress=True)
            from scripts.eval_pr98_route_policy import validate_route_certificate,replay_complete_route
            if ep.status=='stock_solved':
                validate_route_certificate(report,stock)
                replay_complete_route(report)
                result['certificate_and_full_action_replay']='PASS'
            atomic_json(output/f'case-{index:02d}.private.report.json',report)
        results.append(result)
    summary=dict(scope='saved_TRAIN_actions_adapter_replay_only_no_model_generation',
        material_contract=PROTIC_MATERIAL,contract_sha256=sha(contract),cases_sha256=sha(cases),
        cases=len(results),results=results,
        warning='Selected regression witnesses, not an experimental closure rate or general stability audit.')
    atomic_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=replay(args.contract,args.cases,args.output)
    print(json.dumps(dict(cases=result['cases'],summary=str(args.output/'summary.json'))))
