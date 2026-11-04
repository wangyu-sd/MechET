"""Resolve recorded GT imports and verify EVERY new fixed-binding witness.

This is offline artifact/software verification, not a new model experiment.
Original files remain immutable; the complete corrected population is emitted.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import representation_match as matcher
from action_state import before_action
import run_representation_audit as audit

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'results_02'
OUT=HERE/'results_final'
GT=HERE.parents[2]/'data/flower_inverse_tool_sft_action_delta_v1/test.jsonl'
RULES={}


def verify_event(job):
    r,witnesses=job
    event=matcher.Event(r['state_before'],r['state_after'])
    checked=0
    for proof in witnesses:
        rule=RULES[proof['template_id']]
        binding={int(k):int(v) for k,v in proof['template_to_state_atom_map'].items()}
        assert set(binding)==set(rule.q[0]),r['event_id']
        assert len(set(binding.values()))==len(binding),r['event_id']
        side=proof['replay_start_side']
        view=event.view(side,proof['replay_start_representation']!='aromatic')
        for i,x in binding.items():
            assert x in view.query_atoms(rule.q[side][i],i in rule.original[side]),r['event_id']
        for (i,j),q in rule.b[side].items():
            bond=view.bonds.get(tuple(sorted((binding[i],binding[j]))))
            assert bond is not None and q.Match(bond),r['event_id']
        replay=view.replay(rule,side,binding)
        assert replay is not None,r['event_id']
        assert replay['normalized_full_state_sha256']==proof['normalized_full_state_sha256'],r['event_id']
        checked+=1
    return checked


def main():
    start=time.monotonic()
    OUT.mkdir(exist_ok=False)
    source_summary=json.loads((SOURCE/'summary.json').read_text())
    assert source_summary['status']=='complete'
    manifest=source_summary['manifest']
    for p,h in manifest['source_hashes'].items():
        assert audit.sha(p)==h,p
    # Store the exact numerical-runtime code before any future edit.
    runtime={}
    for name in ('representation_match.py','run_representation_audit.py'):
        p=HERE/name
        assert audit.sha(p)==manifest['local_code_sha256'][str(p)]
        runtime[str(p)]={'sha256':audit.sha(p),'source':p.read_text()}
    audit.save(OUT/'runtime_source_snapshot.json',runtime)
    assert audit.sha(GT)=='70a06f01a08490d4b5da602056a8f8e2f978128d3da3224d1601186529aeee3f'
    imports={}
    with GT.open() as f:
        for line in f:
            r=json.loads(line)
            for s in r['metadata']['trace_plan']['steps']:
                if s.get('imports'):
                    imports['gt:'+r['id']+':'+str(s['step_index'])]=s
    assert len(imports)==72
    results={r['event_id']:r for r in audit.rows(SOURCE/'event_matches.jsonl.gz')}
    used={w['template_id'] for r in results.values() for w in r['replay_witnesses']}
    catalog_versions={}
    for r in audit.rows(SOURCE/'replay_template_catalog.jsonl.gz'):
        if r['template_id'] in used:
            RULES[r['template_id']]=matcher.compile_rule(r['reaction_smarts'])
            catalog_versions[r['template_id']]={x['catalog_version'] for x in r['source_refs']}
    assert set(RULES)==used
    jobs=[]
    corrected=[]
    for r in audit.rows(audit.BASE/'event_inputs.jsonl.gz'):
        result=results[r['event_id']]
        if r['event_id'] in imports:
            s=imports[r['event_id']]
            assert (s['state_before'],s['state_after'])==(r['state_before'],r['state_after'])
            resolved=before_action(s['state_before'],s['imports'])
            event=matcher.Event(resolved,s['state_after'])
            # Preserve the recorded free-H radicals, rather than inventing H2.
            # All 72 are outside this CLOSED-SHELL representation repair.
            assert event.radical_changed
            assert result['versions']==[] and result['legacy_versions']==[]
            result['pre_import_preparation_error']=result.pop('repair_error')
            result['repair_status']='radical_change_outside_requested_repair'
            result['input_adapter']='recorded_step_imports_before_action_v1'
            corrected.append(dict(event_id=r['event_id'],state_before_recorded=s['state_before'],
                                  recorded_imports=s['imports'],state_before_action=resolved,
                                  state_after=s['state_after'],source_gt=str(GT),
                                  source_gt_sha256=audit.sha(GT) if not corrected else corrected[0]['source_gt_sha256']))
        if result.get('error'):
            result['legacy_error']=result.pop('error')
        result['status']='author_transition_compatible' if result['versions'] else (
            'unsupported_radical_transition' if result['repair_status']=='radical_change_outside_requested_repair'
            else 'no_certified_author_transition')
        for w in result['replay_witnesses']:
            assert set(w['versions'])<=catalog_versions[w['template_id']]
        if result['replay_witnesses']:
            jobs.append((r,result['replay_witnesses']))
    assert len(corrected)==72
    print(audit.dumps(dict(phase='verify_all_new_replay_certificates',events=len(jobs),
                           witnesses=sum(len(x[1]) for x in jobs),gt_import_states_resolved=72)),flush=True)
    with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('fork')) as pool:
        checked=sum(pool.map(verify_event,jobs,chunksize=32))
    assert checked==sum(len(x[1]) for x in jobs)
    with gzip.open(OUT/'gt_recorded_import_corrections.jsonl.gz','wt') as f:
        for r in corrected:
            f.write(audit.dumps(r)+'\n')
    with gzip.open(OUT/'event_matches.jsonl.gz','wt',compresslevel=3) as f:
        for r in results.values():
            f.write(audit.dumps(r)+'\n')
    manifest=dict(manifest,first_pass_results=str(SOURCE),source_input_states=str(audit.BASE/'event_inputs.jsonl.gz'),
        gt_import_corrections=str(OUT/'gt_recorded_import_corrections.jsonl.gz'),
        gt_import_corrections_sha256=audit.sha(OUT/'gt_recorded_import_corrections.jsonl.gz'),
        gt_import_corrections_count=72,gt_import_corrections_change_coverage=False,
        replay_catalog=str(SOURCE/'replay_template_catalog.jsonl.gz'),
        all_new_replay_witnesses_verified=checked)
    final=audit.summarize(OUT,manifest,source_summary['elapsed_seconds']+time.monotonic()-start)
    audit.save(OUT/'progress.json',dict(status='complete',completed=189233,total=189233,percent=100))
    unit=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    (OUT/'unit_tests.log').write_text(unit.stdout+unit.stderr)
    assert unit.returncode==0 and 'Ran 20 tests' in unit.stderr
    gz=subprocess.run(['gzip','-t',str(OUT/'event_matches.jsonl.gz'),str(OUT/'gt_recorded_import_corrections.jsonl.gz')],capture_output=True,text=True)
    assert gz.returncode==0,gz.stderr
    verification=dict(status='passed',standalone_tests=20,events=189233,
        every_new_fixed_binding_certificate_verified=True,new_replay_events=len(jobs),new_replay_witnesses=checked,
        gt_action_states_resolved=72,coverage_changed_by_import_resolution=False,
        source_dataset_sha256=audit.sha(GT),
        current_code_sha256={str(p):audit.sha(p) for p in HERE.glob('*.py')},
        artifact_sha256={str(p):audit.sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='verification.json'},
        main_repository_suite=dict(status='collection_errors',tests=[
          'tests/test_mech_uspto31k_causal_import.py: missing mechet.a7_rescue',
          'tests/test_nl_history_beam.py: missing mechet.in_place_grounded_flow']))
    audit.save(OUT/'verification.json',verification)
    print((OUT/'report.txt').read_text(),flush=True)
    print(audit.dumps(dict(verification='passed',new_replay_events=len(jobs),new_replay_witnesses=checked,unit_tests=20)),flush=True)


if __name__=='__main__':
    main()
