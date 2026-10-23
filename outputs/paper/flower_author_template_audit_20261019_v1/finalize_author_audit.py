"""Reconcile complete outputs, stratify heavy-bond changes, retain actual cases."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

from rdkit import Chem
from author_template_match import Event
from run_author_audit import ROOT, OLD, SOURCE, VERSIONS, dumps, sha, save_json

HERE = Path(__file__).parent
OUT = HERE/'results_v2'


def read_rows(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt') as f:
        yield from (json.loads(l) for l in f)


def write_csv(p, rows):
    rows = list(rows)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    summary = json.loads((OUT/'summary.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('full audit is not complete')
    matches = {r['event_id']:r for r in read_rows(OUT/'event_matches.jsonl.gz')}
    if len(matches) != 189233:
        raise ValueError('missing or duplicate match rows')
    bins, cases = defaultdict(Counter), {}
    actual_gt = defaultdict(list)
    processing_errors = Counter()
    for r in read_rows(OUT/'event_inputs.jsonl.gz'):
        result = matches[r['event_id']]
        try:
            event = Event(r['state_before'], r['state_after'])
            kind = 'heavy_bond_rearrangement' if event.db else 'no_heavy_bond_change'
        except Exception as exc:
            kind = 'hydrogen_projection_or_state_processing_unsupported'
            processing_errors[str(exc)] += 1
        pop = r['population']
        c = bins[(pop,kind)]
        c['events'] += 1
        c['either_version_compatible'] += bool(result['versions'])
        c.update(result['versions'])
        key = (pop, r.get('run_id','GT'), kind, result['status'])
        if key not in cases:
            cases[key] = dict(input=r, match=result)
        if 'reaction_id' in r:
            actual_gt[r['reaction_id']].append(result)
    gt_rows = []
    for r in read_rows(ROOT/'data/flower_inverse_tool_sft_action_delta_v1/test.jsonl'):
        n = len(r['metadata']['trace_plan']['steps'])
        events = actual_gt.get(r['id'], [])
        if n != len(events):
            raise ValueError('GT reaction lost an event: '+r['id'])
        gt_rows.append(dict(reaction_id=r['id'], events=n, **{
            'compatible_events_'+v: sum(bool(x['versions']) if v=='either' else v in x['versions'] for x in events)
            for v in ['either', *VERSIONS]}))
    if len(gt_rows) != 28967:
        raise ValueError('wrong strict GT reaction population')
    write_csv(OUT/'gt_reaction_coverage.csv', gt_rows)
    write_csv(OUT/'coverage_by_event_type.csv', [dict(population=p, event_type=k, **{
        name:c[name] for name in ['events','either_version_compatible',*VERSIONS]}) for (p,k),c in bins.items()])
    # Bind raw state compatibility to the separate prior declaration-delta check.
    declaration_checks = Counter()
    seen = set()
    for r in read_rows(OLD/'results/event_annotations_v2.jsonl.gz'):
        if r['event_id'] in seen:
            continue
        seen.add(r['event_id'])
        x = matches[r['event_id']]
        if x['versions']:
            declaration_checks[r.get('transition_check','unavailable')] += 1
    if len(seen) != 71868:
        raise ValueError('saved-event reconciliation failed')
    with gzip.open(OUT/'representative_cases.jsonl.gz', 'wt') as f:
        for key,r in sorted(cases.items()):
            f.write(dumps(dict(stratum=key, **r))+'\n')
    primary = {}
    for pop in summary['populations']:
        n = summary['populations'][pop]
        primary[pop] = dict(events=n,
            current_version_matches=summary['version_event_coverage'][pop].get(VERSIONS[0],0),
            dated_version_matches=summary['version_event_coverage'][pop].get(VERSIONS[1],0),
            either_version_matches=summary['status_counts'][pop].get('author_transition_compatible',0))
    artifacts = {str(p): sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'delivery_summary.json'}
    raw_files = list((SOURCE/'Mechanistic_dataset/templates').glob('*.py'))
    result = dict(primary_coverage=primary,
        by_event_type=[dict(population=p,event_type=k,**dict(c)) for (p,k),c in bins.items()],
        gt_reaction_count=28967, gt_zero_event_reactions=sum(r['events']==0 for r in gt_rows),
        author_compatible_model_events_prior_declaration_delta_check=dict(declaration_checks),
        unsupported_projection_errors=dict(processing_errors), representative_case_count=len(cases),
        verification=dict(standalone_tests=15, main_repository_suite='collection failed on existing missing a7_rescue and in_place_grounded_flow'),
        raw_author_source_sha256={str(p):sha(p) for p in raw_files},
        artifact_sha256=artifacts,
        local_code_sha256={str(p):sha(p) for p in HERE.glob('*.py')},
        limitations=['local paired-state compatibility, not independent chemical truth',
                     'full reagent screens, stage sequence, unique arrow pairing and kinetics unverified',
                     'GT is the complete STRICT COMPILED test view; official test has 4 extra corrupt endpoint rows',
                     'unsupported chemistry stays in denominator; no nonmatch is labelled chemically incorrect',
                     'no new sampling or training; earlier endpoint-hit counts are not altered'])
    save_json(OUT/'delivery_summary.json',result)
    lines = ['FlowER official author-template compatibility audit (complete)',
             'Primary: current author version. Dated version and union reported separately.', '']
    for p,r in primary.items():
        lines.append(p+': '+str(r['events'])+' events')
        for name in ['current_version_matches','dated_version_matches','either_version_matches']:
            lines.append(f'  {name}: {r[name]}/{r["events"]} = {100*r[name]/r["events"]:.3f}%')
    lines += ['', 'Separate heavy-bond-changing events from charge/H-only events:']
    for (p,k),c in bins.items():
        lines.append(f'{p} | {k}: current {c[VERSIONS[0]]}/{c["events"]}; either {c["either_version_compatible"]}/{c["events"]}')
    lines += ['', 'All-event path coverage and per-run counts: summary.json.',
              'All original states, moves and sources: event_inputs.jsonl.gz.',
              'Every match and binding: event_matches.jsonl.gz.',
              'Nonmatch does NOT mean chemically incorrect. This library is the author generator, not independent experimental validation.',
              '15 standalone unit tests passed. Whole repository pytest is blocked at collection by the two missing historical modules recorded in README.txt.']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
