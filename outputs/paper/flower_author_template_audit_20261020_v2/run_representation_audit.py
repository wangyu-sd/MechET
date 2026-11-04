"""Full-population incremental representation repair; no new model sampling.

Every old result is retained unchanged. Only missing VERSION witnesses need
new replay. Denominators are all 189233 events, not the retry subset.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

from rdkit import Chem
import representation_match as matcher

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'flower_author_template_audit_20261019_v1/results_v2'
VERSIONS=('Reaction_templates','Reaction_templates_20250216')
GROUPS=None
PROGRAM_LINKS=HERE.parent/'electron_event_templates_20261019_v1/results/program_event_links.jsonl.gz'


def program_links():
    return rows(PROGRAM_LINKS)


def dumps(r):
    return json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def save(p,r):
    p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with gzip.open(p,'rt') as f:
        yield from (json.loads(line) for line in f)


def load_catalog(out):
    raw={}
    for r in rows(BASE/'compiled_author_templates.jsonl.gz'):
        raw[r['reaction_smarts']]=list(r['source_refs'])
    for r in rows(BASE/'unsupported_author_templates.jsonl.gz'):
        s=r['expanded_smarts']
        ref={k:v for k,v in r.items() if k!='unsupported_reason'}
        raw.setdefault(s,[]).append(ref)
    groups=defaultdict(list)
    errors=[]
    with gzip.open(out/'replay_template_catalog.jsonl.gz','wt',compresslevel=3) as f:
        for s,refs in raw.items():
            rid=hashlib.sha256(s.encode()).hexdigest()
            try:
                r=matcher.compile_rule(s)
                r.refs=refs
                groups[(r.sigma_key,r.typed_key)].append(r)
                f.write(dumps(dict(template_id=rid,reaction_smarts=s,
                                   explicit_hydrogen_smarts=r.expanded,source_refs=refs))+'\n')
            except Exception as exc:
                errors.append(dict(template_id=rid,reaction_smarts=s,source_refs=refs,
                                   error=type(exc).__name__+': '+str(exc)))
    with gzip.open(out/'unsupported_templates.jsonl.gz','wt') as f:
        for r in errors:
            f.write(dumps(r)+'\n')
    for rules in groups.values():
        rules.sort(key=lambda r:(len(r.q[0]),len(r.raw),r.raw))
    return groups,dict(raw_unique=len(raw),replay_supported_unique=sum(map(len,groups.values())),
                       unsupported_unique=len(errors),errors=dict(Counter(r['error'].split('\n')[0] for r in errors)))


def process_one(job):
    r,old=job
    old_versions=set(old['versions'])
    missing=set(VERSIONS)-old_versions
    out={k:v for k,v in old.items() if k not in ('bindings','matching_template_ids')}
    out.update(legacy_versions=sorted(old_versions),legacy_status=old['status'],new_replay_versions=[],
               replay_witnesses=[],guarded_replay_templates=[])
    if not missing:
        out['repair_status']='legacy_both_versions_reused'
        return out
    try:
        event=matcher.Event(r['state_before'],r['state_after'])
        if event.radical_changed:
            out['repair_status']='radical_change_outside_requested_repair'
            return out
        if event.unchanged:
            out['repair_status']='no_event'
            return out
        for rule in event.candidate_rules(GROUPS):
            supported={ref['catalog_version'] for ref in rule.refs}
            if not missing & supported:
                continue
            rid=hashlib.sha256(rule.raw.encode()).hexdigest()
            try:
                proof=event.match(rule)
            except matcher.MatchWorkLimit:
                out['guarded_replay_templates'].append(rid)
                continue
            if proof:
                added=missing & supported
                out['replay_witnesses'].append(dict(template_id=rid,versions=sorted(added),**proof))
                missing-=added
                if not missing:
                    break
        out['new_replay_versions']=sorted((set(VERSIONS)-missing)-old_versions)
        out['versions']=sorted(old_versions|set(out['new_replay_versions']))
        out['repair_status']='new_full_state_replay_match' if out['new_replay_versions'] else (
            'inconclusive_work_guard' if out['guarded_replay_templates'] else 'no_additional_match')
        if out['versions']:
            out['status']='author_transition_compatible'
    except Exception as exc:
        out['repair_status']='representation_processing_error'
        out['repair_error']=type(exc).__name__+': '+str(exc)
    return out


def process_chunk(chunk):
    return [process_one(r) for r in chunk]


def summarize(out,manifest,elapsed):
    c=defaultdict(Counter)
    byrun=defaultdict(Counter)
    results={}
    for r in rows(out/'event_matches.jsonl.gz'):
        if r['event_id'] in results:
            raise ValueError('duplicate event result')
        results[r['event_id']]=r
        pop=r['population']
        count=c[pop]
        count['events']+=1
        count['old_either']+=bool(r['legacy_versions'])
        count['new_either']+=bool(r['versions'])
        count[r['repair_status']]+=1
        for v in VERSIONS:
            count['old:'+v]+=v in r['legacy_versions']
            count['new:'+v]+=v in r['versions']
        if 'run_id' in r:
            b=byrun[r['run_id']]
            b['events']+=1
            for v in VERSIONS:
                b['old:'+v]+=v in r['legacy_versions']
                b['new:'+v]+=v in r['versions']
        assert set(r['legacy_versions'])<=set(r['versions'])
    assert len(results)==189233
    assert c['saved_model_event']['events']==71868
    assert c['flower_strict_test_gt_event']['events']==117365
    paths=defaultdict(Counter)
    for r in program_links():
        events=[results[i] for i in r['event_ids']]
        p=paths[r['run_id']]
        p['programs']+=1
        p['event_occurrences']+=len(events)
        for v in VERSIONS:
            p['all_events:'+v]+=bool(events) and all(v in e['versions'] for e in events)
    assert sum(v['programs'] for v in paths.values())==37922
    assert sum(v['event_occurrences'] for v in paths.values())==124808
    gt=defaultdict(list)
    for r in results.values():
        if 'reaction_id' in r:
            gt[r['reaction_id']].append(r)
    assert len(gt)==28967
    summary=dict(status='complete',manifest=manifest,elapsed_seconds=elapsed,
        populations={k:dict(v) for k,v in c.items()},by_run={k:dict(v) for k,v in byrun.items()},
        program_paths={k:dict(v) for k,v in paths.items()},
        gt_reaction_paths={v:sum(all(v in r['versions'] for r in events) for events in gt.values()) for v in VERSIONS},
        no_old_positive_removed=True,
        interpretation='Author local-transition compatibility: legacy strict joint-state match OR new complete mapped-state template-replay certificate. Not kinetic feasibility or endpoint accuracy.',
        limitations=['radical changes and unbalanced author templates remain unsupported',
                     'no reaction-level reagent/exclusion or mechanistic stage-sequence certification',
                     'one witness per previously missing catalog version; not exhaustive template multiplicity',
                     'all old inputs/outputs immutable; no model/executor/data changes'])
    save(out/'summary.json',summary)
    lines=['Representation repair: complete (same 189233-event population)',
           'Current author template revision (Reaction_templates):']
    for pop,v in c.items():
        n=v['events'];a=v['old:'+VERSIONS[0]];b=v['new:'+VERSIONS[0]]
        lines.append(f'{pop}: {a}/{n} ({100*a/n:.3f}%) -> {b}/{n} ({100*b/n:.3f}%); added {b-a}')
    for run,v in byrun.items():
        n=v['events'];a=v['old:'+VERSIONS[0]];b=v['new:'+VERSIONS[0]]
        lines.append(f'{run}: {a}/{n} ({100*a/n:.3f}%) -> {b}/{n} ({100*b/n:.3f}%)')
    lines+=['All additions require full-state replay equality with original maps, charge, H, isotope and stereo preserved.',
            'This is a MATCHER repair, not an improvement in model predictions or endpoint accuracy.']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    global GROUPS
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    # Authenticate immutable v1 artifacts before reusing any positives.
    validation=json.loads((BASE/'verification.json').read_text())
    paths=[BASE/'event_inputs.jsonl.gz',BASE/'event_matches.jsonl.gz',
           BASE/'compiled_author_templates.jsonl.gz',BASE/'unsupported_author_templates.jsonl.gz']
    for path in paths:
        assert sha(path)==validation['artifact_sha256'][str(path)]
    old_summary=json.loads((BASE/'summary.json').read_text())
    assert sha(PROGRAM_LINKS)==old_summary['source_hashes'][str(PROGRAM_LINKS)]
    paths.append(PROGRAM_LINKS)
    manifest=dict(source_hashes={str(p):sha(p) for p in paths},
        rdkit_version=Chem.rdBase.rdkitVersion,author_h_expansion_source=str(matcher.H_SOURCE),
        author_h_expansion_sha256=sha(matcher.H_SOURCE),
        original_denominator=189233,old_positive_reuse=True,
        local_code_sha256={str(p):sha(p) for p in HERE.glob('*.py')})
    save(a.out/'manifest.json',manifest)
    GROUPS,cat=load_catalog(a.out)
    save(a.out/'catalog_summary.json',cat)
    print(dumps(dict(phase='catalog_ready',**cat)),flush=True)
    old={r['event_id']:r for r in rows(BASE/'event_matches.jsonl.gz')}
    jobs=[]
    for r in rows(BASE/'event_inputs.jsonl.gz'):
        jobs.append(({k:r[k] for k in ('event_id','state_before','state_after')},old[r['event_id']]))
    assert len(jobs)==len(old)==189233
    chunks=[jobs[i:i+32] for i in range(0,len(jobs),32)]
    done=0
    last=time.monotonic()
    with gzip.open(a.out/'event_matches.jsonl.gz','wt',compresslevel=3) as f:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            for records in pool.map(process_chunk,chunks,chunksize=1):
                for r in records:
                    f.write(dumps(r)+'\n')
                done+=len(records)
                if done%256==0 or done==len(jobs):
                    f.flush()
                    progress=dict(status='running',completed=done,total=len(jobs),percent=100*done/len(jobs),elapsed_seconds=time.monotonic()-start)
                    save(a.out/'progress.json',progress)
                    if time.monotonic()-last>=30 or done==len(jobs):
                        print(dumps(progress),flush=True)
                        last=time.monotonic()
    result=summarize(a.out,manifest,time.monotonic()-start)
    save(a.out/'progress.json',dict(status='complete',completed=done,total=len(jobs),percent=100,elapsed_seconds=result['elapsed_seconds']))
    print((a.out/'report.txt').read_text(),flush=True)


if __name__=='__main__':
    main()
