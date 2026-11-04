"""Reporting adapter; never changes witnesses, version membership or counts."""
from collections import Counter,defaultdict
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

from rdkit import rdBase
import audit_transition_repair as audit


def final_row(row):
    out=dict(row)
    out['legacy_v2_status']=out.pop('status')
    out['legacy_v2_repair_status']=out.pop('repair_status')
    if 'Reaction_templates' in out['versions']:
        out['status']='author_single_template_compatible_current'
    elif 'Reaction_templates' in out['composition_versions']:
        out['status']='author_composition_compatible_current'
    elif out['versions'] or out['composition_versions']:
        out['status']='dated_author_version_only'
    elif out['v3_guarded']:
        out['status']='inconclusive_work_guard'
    else:
        out['status']='no_certified_author_transition'
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=audit.HERE/'results')
    parser.add_argument('--out',type=Path,default=audit.HERE/'results_final')
    args=parser.parse_args();source=args.source.resolve();out=args.out.resolve()
    raw_summary=json.loads((source/'summary.json').read_text())
    raw_verify=json.loads((source/'verification.json').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    assert raw_summary['status']=='complete' and raw_verify['status']=='passed'
    for p,h in manifest['source_sha256'].items():assert audit.sha(p)==h,p
    for p,h in raw_verify['artifact_sha256'].items():
        # The first-pass runner wrote final progress AFTER its hash. Numerical
        # files must still match. Final delivery hashes progress after sealing.
        if Path(p).name!='progress.json':assert audit.sha(p)==h,p
    out.mkdir(exist_ok=False)
    rs=[final_row(r) for r in audit.rows(source/'event_matches.jsonl.gz')]
    assert len(rs)==189233 and len({r['event_id'] for r in rs})==189233
    assert audit.coverage(rs)==raw_summary['coverage']
    with gzip.open(out/'event_matches.jsonl.gz','wt',compresslevel=3) as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    gt=defaultdict(list)
    for r in rs:
        if r['population']=='flower_strict_test_gt_event':gt[r['reaction_id']].append(r)
    assert len(gt)==28967
    gt_paths={}
    for v in audit.VERSIONS:
        gt_paths[v]=dict(reactions=len(gt),single_template_each_event=sum(all(v in r['versions'] for r in es) for es in gt.values()),
                        single_or_composed_each_event=sum(all(v in set(r['versions'])|set(r['composition_versions']) for r in es) for es in gt.values()))
    target={r['event_id']:r for es in gt.values() for r in es if 'Reaction_templates' not in r['versions']}
    fixes={r['event_id']:r for r in audit.rows(audit.BASE/'gt_recorded_import_corrections.jsonl.gz')}
    reasons=Counter();examples={}
    with gzip.open(out/'unmatched_single_gt_inputs_and_diagnostics.jsonl.gz','wt') as f:
        for r in audit.rows(audit.INPUTS):
            if r['event_id'] not in target:continue
            result=target[r['event_id']]
            if r['event_id'] in fixes:r=dict(r,state_before=fixes[r['event_id']]['state_before_action'])
            if 'Reaction_templates' in result['composition_versions']:reason='composition_only_not_single_rule'
            elif r['event_id'] in fixes:reason='recorded_free_H_import_case_unresolved'
            elif audit.matcher.Event(r['state_before'],r['state_after']).radical_changed:reason='radical_transition_unresolved'
            else:reason='closed_shell_transition_unresolved'
            reasons[reason]+=1
            examples.setdefault(reason,r['event_id'])
            f.write(json.dumps(dict(**r,result=result,diagnostic=reason),ensure_ascii=False,separators=(',',':'))+'\n')
    assert sum(reasons.values())==len(target)
    summary=dict(raw_summary,gt_reaction_paths=gt_paths,current_gt_single_unmatched=len(target),
                 current_gt_single_unmatched_diagnostics=dict(reasons),diagnostic_example_ids=examples,
                 diagnostic_note='Observable categories, not proof that GT or chemistry is wrong.',
                 rdkit_version=rdBase.rdkitVersion)
    audit.save(out/'summary.json',summary)
    snapshot={}
    dependencies=[*audit.HERE.glob('*.py'),audit.V2/'representation_match.py',
                  audit.V2.parent/'flower_author_template_audit_20261019_v1/author_template_match.py',
                  audit.matcher.legacy.H_SOURCE]
    for p in dependencies:snapshot[str(p)]=dict(sha256=audit.sha(p),source=p.read_text())
    audit.save(out/'runtime_source_snapshot.json',snapshot)
    tests=[]
    for folder in (audit.V2,audit.HERE):
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(folder),'-p','test_*.py','-v'],capture_output=True,text=True)
        (out/(folder.name+'_unit_tests.log')).write_text(p.stdout+p.stderr)
        tests.append(dict(folder=str(folder),returncode=p.returncode))
    assert all(t['returncode']==0 for t in tests)
    for p in out.glob('*.gz'):subprocess.run(['gzip','-t',str(p)],check=True)
    audit.save(out/'progress.json',dict(status='complete',completed=189233,events=189233,percentage=100))
    audit.save(out/'manifest.json',dict(manifest,rdkit_version=rdBase.rdkitVersion,
               numerical_source=str(source),numerical_source_sha256={str(p):audit.sha(p) for p in source.iterdir() if p.is_file()},
               code_source_snapshot=str(out/'runtime_source_snapshot.json'),
               missing_preexisting_dependency_hashes_added={str(p):audit.sha(p) for p in dependencies}))
    lines=['Full frozen population:189233 events; no model generation.','Current author library: Reaction_templates',
           'Population | denominator | v2 single | v3 single | v3 single OR composed']
    for name,c in summary['coverage'].items():
        n=c['events'];fields=[c['v2:Reaction_templates'],c['single:Reaction_templates'],c['single_or_composed:Reaction_templates']]
        lines.append(f'{name} | {n} | '+' | '.join(f'{v} ({100*v/n:.3f}%)' for v in fields))
    lines+=['Remaining single-rule GT diagnostics:'+json.dumps(dict(reasons)),
            'Local author-transition compatibility only; NOT endpoint accuracy, chemical truth, reaction conditions or stage-sequence certification.',
            'Single-rule matches and multi-rule compositions are separate metrics; no old positive removed.',
            'Whole-repository pytest:2 existing collection failures:',
            'tests/test_mech_uspto31k_causal_import.py: missing mechet.a7_rescue',
            'tests/test_nl_history_beam.py: missing mechet.in_place_grounded_flow']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    audit.save(out/'verification.json',dict(status='passed',events=189233,tests=tests,
                  all_new_witnesses_verified=raw_summary['new_fixed_binding_witnesses_verified'],
                  processing_errors=raw_summary['processing_errors'],guarded_events=raw_summary['inconclusive_guard_events'],
                  reporting_changes_numerical_counts=False,
                  source_snapshot_sha256=audit.sha(out/'runtime_source_snapshot.json'),
                  main_repository_suite={'status':'collection_errors','tests':['tests/test_mech_uspto31k_causal_import.py: mechet.a7_rescue','tests/test_nl_history_beam.py: mechet.in_place_grounded_flow']},
                  artifact_sha256={str(p):audit.sha(p) for p in out.iterdir() if p.is_file()}))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
