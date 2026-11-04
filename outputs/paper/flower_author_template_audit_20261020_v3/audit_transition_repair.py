"""Full-population audit; no model calls, no mutable historical results."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

from rdkit import Chem,RDLogger
import transition_repair as matcher

HERE=Path(__file__).resolve().parent
V2=matcher.V2
BASE=V2/'results_final'
INPUTS=V2.parent/'flower_author_template_audit_20261019_v1/results_v2/event_inputs.jsonl.gz'
CATALOG=V2/'results_02/replay_template_catalog.jsonl.gz'
LINKS=V2.parent/'electron_event_templates_20261019_v1/results/program_event_links.jsonl.gz'
VERSIONS=('Reaction_templates','Reaction_templates_20250216')
RULES={};GROUPS={};VERSION_GROUPS={}


def rows(p):
    with gzip.open(p,'rt') as f:
        yield from (json.loads(line) for line in f)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(p,r):
    p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')


def coverage(rows):
    c=defaultdict(Counter)
    for r in rows:
        for group in (r['population'],*([r['run_id']] if 'run_id' in r else [])):
            out=c[group];out['events']+=1
            for v in VERSIONS:
                out['v2:'+v]+=v in r['legacy_v2_versions']
                out['single:'+v]+=v in r['versions']
                out['single_or_composed:'+v]+=v in set(r['versions'])|set(r['composition_versions'])
            out['status:'+r.get('repair_v3_status','unspecified')]+=1
    return {k:dict(v) for k,v in c.items()}


def load_rules():
    for row in rows(CATALOG):
        rule=matcher.compile_rule(row['reaction_smarts'])
        rule.refs=row['source_refs']
        RULES[row['template_id']]=rule
    GROUPS.update(matcher.group_rules(RULES.values()))
    for version in VERSIONS:
        VERSION_GROUPS[version]=matcher.group_rules(r for r in RULES.values()
                                                   if any(x['catalog_version']==version for x in r.refs))


def verify_proof(event,proof):
    rule=RULES[proof['template_id']]
    binding={int(k):int(v) for k,v in proof['template_to_state_atom_map'].items()}
    assert set(binding)==set(rule.q[0]) and len(set(binding.values()))==len(binding)
    side=proof['replay_start_side']
    view=event.view(side,proof['replay_start_representation']!='aromatic')
    for i,x in binding.items():assert x in view.query_atoms(rule.q[side][i],i in rule.original[side])
    for (i,j),q in rule.b[side].items():
        b=view.bonds.get(tuple(sorted((binding[i],binding[j]))))
        assert b is not None and q.Match(b)
    actual=view.replay(rule,side,binding)
    assert actual and actual['normalized_full_state_sha256']==proof['normalized_full_state_sha256']
    assert set(proof['versions'])<={r['catalog_version'] for r in rule.refs}


def normalized_state(s,ids):
    return matcher.legacy.normalized(Chem.AddHs(matcher.legacy.parse(s)),ids)


def process_one(job):
    original,old=job
    result=dict(old)
    result.update(legacy_v2_versions=list(old['versions']),direct_v3_witnesses=[],
                  composition_versions=[],composition_witnesses=[],v3_guarded=[],v3_errors=[])
    missing=set(VERSIONS)-set(old['versions'])
    if not missing:
        result['repair_v3_status']='v2_both_versions_reused'
        return result
    try:
        event=matcher.Event(original['state_before'],original['state_after'])
        for rule in event.candidate_rules(GROUPS):
            versions={r['catalog_version'] for r in rule.refs}&missing
            if not versions:continue
            try:proof=event.match(rule)
            except matcher.legacy.MatchWorkLimit as exc:
                result['v3_guarded'].append(str(exc));continue
            if proof:
                proof.update(template_id=hashlib.sha256(rule.raw.encode()).hexdigest(),versions=sorted(versions))
                verify_proof(event,proof)
                result['direct_v3_witnesses'].append(proof)
                missing-=versions
                if not missing:break
        result['versions']=sorted(set(VERSIONS)-missing)
        for version in sorted(missing):
            try:composition=matcher.match_composition(original['state_before'],original['state_after'],VERSION_GROUPS[version])
            except matcher.legacy.MatchWorkLimit as exc:
                result['v3_guarded'].append(str(exc));continue
            if composition:
                # Independent fixed-binding replay of EVERY component and
                # exact chain boundary checks against ORIGINAL complete states.
                previous=normalized_state(original['state_before'],event.original_maps)
                for step in composition['steps']:
                    assert normalized_state(step['state_before'],event.original_maps)==previous
                    step['versions']=[version]
                    verify_proof(matcher.Event(step['state_before'],step['state_after']),step)
                    previous=normalized_state(step['state_after'],event.original_maps)
                assert previous==event.keys[0]
                composition['versions']=[version]
                result['composition_versions'].append(version)
                result['composition_witnesses'].append(composition)
        result['repair_v3_status']=('new_single_rule_replay' if result['direct_v3_witnesses'] else
                                   'new_composition_only' if result['composition_versions'] else
                                   'inconclusive_guard' if result['v3_guarded'] else 'no_additional_match')
    except Exception as exc:
        # This must be zero in a completed delivery, never counted as a
        # chemical negative. Assertion errors also surface to the driver.
        if isinstance(exc,AssertionError):raise
        result['v3_errors'].append(type(exc).__name__+': '+str(exc))
        result['repair_v3_status']='processing_error'
    assert set(old['versions'])<=set(result['versions'])
    return result


def process_chunk(chunk):
    # RDKit messages here are failed hypothetical template bindings. Actual
    # per-event exceptions remain explicit in v3_errors and fail delivery.
    RDLogger.DisableLog('rdApp.error')
    return [process_one(job) for job in chunk]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=HERE/'results')
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(exist_ok=False)
    start=time.monotonic()
    old_summary=json.loads((BASE/'summary.json').read_text())
    old_verify=json.loads((BASE/'verification.json').read_text())
    assert old_summary['status']=='complete' and old_verify['status']=='passed'
    source_paths=[INPUTS,CATALOG,BASE/'event_matches.jsonl.gz',BASE/'gt_recorded_import_corrections.jsonl.gz',LINKS,
                  V2/'representation_match.py',matcher.legacy.H_SOURCE,HERE/'transition_repair.py',Path(__file__)]
    hashes={str(p):sha(p) for p in source_paths}
    for p,h in old_summary['manifest']['source_hashes'].items():assert sha(p)==h,p
    for p,h in old_verify['artifact_sha256'].items():assert sha(p)==h,p
    assert sha(matcher.legacy.H_SOURCE)==old_summary['manifest']['author_h_expansion_sha256']
    assert sha(V2/'representation_match.py')==old_verify['current_code_sha256'][str(V2/'representation_match.py')]
    save(out/'manifest.json',dict(source_sha256=hashes,events=189233,workers=args.workers,
                                single_and_composition_separate=True,no_new_model_calls=True))
    old={r['event_id']:r for r in rows(BASE/'event_matches.jsonl.gz')}
    corrections={r['event_id']:r for r in rows(BASE/'gt_recorded_import_corrections.jsonl.gz')}
    load_rules()
    jobs=[];ids=set()
    for row in rows(INPUTS):
        eid=row['event_id'];assert eid not in ids;ids.add(eid)
        if eid in corrections:
            fix=corrections[eid]
            assert row['state_before']==fix['state_before_recorded']
            row['state_before']=fix['state_before_action']
        jobs.append((row,old[eid]))
    assert ids==set(old) and len(ids)==189233
    chunks=[jobs[i:i+32] for i in range(0,len(jobs),32)]
    results=[];last=0
    with gzip.open(out/'event_matches.jsonl.gz','wt',compresslevel=3) as f:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            for chunk in pool.map(process_chunk,chunks,chunksize=1):
                for row in chunk:
                    results.append(row);f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                now=time.monotonic()
                if now-last>10 or len(results)==189233:
                    progress=dict(status='running',completed=len(results),events=189233,
                                  percentage=100*len(results)/189233,elapsed_seconds=now-start)
                    save(out/'progress.json',progress);print(json.dumps(progress),flush=True);last=now
    counts=coverage(results)
    lookup={r['event_id']:r for r in results}
    paths=defaultdict(Counter)
    for link in rows(LINKS):
        rs=[lookup[i] for i in link['event_ids']];group=paths[link['run_id']];group['programs']+=1
        for v in VERSIONS:
            group['single:'+v]+=bool(rs) and all(v in r['versions'] for r in rs)
            group['single_or_composed:'+v]+=bool(rs) and all(v in set(r['versions'])|set(r['composition_versions']) for r in rs)
    assert sum(x['programs'] for x in paths.values())==37922
    errors=[r for r in results if r['v3_errors']]
    guarded=[r for r in results if r['v3_guarded']]
    proof_count=sum(len(r['direct_v3_witnesses'])+sum(len(c['steps']) for c in r['composition_witnesses']) for r in results)
    summary=dict(status='complete' if not errors else 'completed_with_processing_errors',events=len(results),
                 coverage=counts,program_paths={k:dict(v) for k,v in paths.items()},
                 new_fixed_binding_witnesses_verified=proof_count,processing_errors=len(errors),
                 inconclusive_guard_events=len(guarded),elapsed_seconds=time.monotonic()-start,
                 interpretation='Author local-transition compatibility, not endpoint accuracy or chemical feasibility. Composition is not single-rule matching.')
    save(out/'summary.json',summary)
    tests=[]
    for folder in (V2,HERE):
        proc=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(folder),'-p','test_*.py','-v'],capture_output=True,text=True)
        tests.append(dict(folder=str(folder),returncode=proc.returncode))
        (out/(folder.name+'_unit_tests.log')).write_text(proc.stdout+proc.stderr)
    subprocess.run(['gzip','-t',str(out/'event_matches.jsonl.gz')],check=True)
    save(out/'verification.json',dict(status='passed' if not errors and all(t['returncode']==0 for t in tests) else 'failed',
         tests=tests,all_new_witnesses_verified=proof_count,full_event_population=len(results),
         processing_errors=len(errors),guarded_events=len(guarded),
         artifact_sha256={str(p):sha(p) for p in out.iterdir() if p.is_file()}))
    save(out/'progress.json',dict(status=summary['status'],completed=len(results),events=189233,percentage=100,
                                elapsed_seconds=time.monotonic()-start))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    assert not errors


if __name__=='__main__':main()
