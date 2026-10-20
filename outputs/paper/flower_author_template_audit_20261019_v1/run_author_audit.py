"""Full offline saved-event + full strict-GT-test audit. No model sampling."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

from author_template_match import Event, MatchWorkLimit, load_catalog

ROOT = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
SOURCE = Path('/aaa/fionafyang/buddy1/whaleywang/datasets/flower_original_mechanistic_templates_20261019')
OLD = ROOT / 'outputs/paper/electron_event_templates_20261019_v1'
GROUPS = None
VERSIONS = ['Reaction_templates', 'Reaction_templates_20250216']


def reconcile_predictions(rows):
    seen = {}
    for row in rows:
        eid = row['event_id']
        if eid in seen:
            old = seen[eid]
            for key in ('state_before', 'state_after', 'inverse_moves', 'arguments', 'run_id'):
                if old.get(key) != row.get(key):
                    raise ValueError(f'conflicting shared event {eid}: {key}')
            old['program_occurrences'] = old.get('program_occurrences', 0)+row.get('program_occurrences', 0)
            for key in ('program_record_ids', 'reaction_keys'):
                old[key] = sorted(set(old.get(key, ())) | set(row.get(key, ())))
        else:
            seen[eid] = dict(row, native_sources=[])
        seen[eid]['native_sources'].append({k: row.get(k) for k in
            ('native_source_path', 'native_source_sha256', 'native_action_pointer')})
    return list(seen.values())


def dumps(r):
    return json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(p, r):
    Path(p).write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n')


def check_one(r):
    out = {k: r[k] for k in ('event_id', 'population', 'run_id', 'reaction_id', 'action_index') if k in r}
    out.update(matching_template_ids=[], bindings={}, versions=[], guarded_template_ids=[])
    try:
        event = Event(r['state_before'], r['state_after'])
        if event.radical_changed:
            out['status'] = 'radical_change_not_implemented'
            return out
        if not event.centre:
            out['status'] = 'no_normalized_change'
            return out
        versions = set()
        for rule in event.candidate_rules(GROUPS):
            try:
                binding = event.match(rule)
            except MatchWorkLimit:
                out['guarded_template_ids'].append(hashlib.sha256(rule.smarts.encode()).hexdigest())
                continue
            if binding:
                rid = hashlib.sha256(rule.smarts.encode()).hexdigest()
                out['matching_template_ids'].append(rid)
                out['bindings'][rid] = binding
                versions.update(ref['catalog_version'] for ref in rule.refs)
        out['versions'] = sorted(versions)
        out['status'] = 'author_transition_compatible' if versions else (
            'matching_inconclusive_guard' if out['guarded_template_ids'] else 'no_implemented_author_template_match')
        out['matching_template_ids'].sort()
    except Exception as exc:
        out['status'] = 'state_processing_error'
        out['error'] = f'{type(exc).__name__}: {exc}'
    return out


def check_chunk(chunk):
    return [check_one(r) for r in chunk]


def prepare(out):
    events, predicted_ids = [], set()
    gt_ids = set()
    with gzip.open(out/'event_inputs.jsonl.gz', 'wt', compresslevel=3) as f:
        with gzip.open(OLD/'results/event_annotations_v2.jsonl.gz', 'rt') as src:
            for r in reconcile_predictions(json.loads(line) for line in src):
                predicted_ids.add(r['event_id'])
                clean = {k: r[k] for k in ('event_id', 'run_id', 'target_id', 'state_before', 'state_after',
                            'inverse_moves', 'arguments', 'action_index', 'program_record_ids',
                            'program_occurrences', 'native_source_path', 'native_source_sha256',
                            'native_action_pointer', 'native_sources', 'reaction_keys') if k in r}
                clean['population'] = 'saved_model_event'
                f.write(dumps(clean)+'\n')
                events.append({k: clean[k] for k in ('event_id', 'population', 'run_id', 'state_before', 'state_after', 'action_index')})
        source = ROOT/'data/flower_inverse_tool_sft_action_delta_v1/test.jsonl'
        for line in source.open():
            r = json.loads(line)
            if r['id'] in gt_ids:
                raise ValueError('duplicate GT reaction ID')
            gt_ids.add(r['id'])
            for step in r['metadata']['trace_plan']['steps']:
                event_id = 'gt:'+r['id']+':'+str(step['step_index'])
                clean = dict(event_id=event_id, population='flower_strict_test_gt_event',
                             reaction_id=r['id'], action_index=step['step_index'],
                             state_before=step['state_before'], state_after=step['state_after'],
                             inverse_moves=step['moves'])
                f.write(dumps(clean)+'\n')
                events.append({k: clean[k] for k in ('event_id', 'population', 'reaction_id', 'state_before', 'state_after', 'action_index')})
    if len(gt_ids) != 28967 or len(predicted_ids) != 71868:
        raise ValueError(f'wrong population: GT {len(gt_ids)}, saved {len(predicted_ids)}')
    counts = Counter(r['population'] for r in events)
    if counts['flower_strict_test_gt_event'] != 117365:
        raise ValueError(f'unexpected GT event count {counts}')
    return events, dict(counts)


def summarize(out, totals, wall, source_hashes, catalog_counts, unsupported):
    counts = defaultdict(Counter)
    by_run = defaultdict(Counter)
    version_counts = defaultdict(Counter)
    gt_reactions = defaultdict(lambda: {'events': 0, 'matches': Counter()})
    result_by_id = {}
    with gzip.open(out/'event_matches.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            pop = r['population']
            counts[pop][r['status']] += 1
            for version in r['versions']:
                version_counts[pop][version] += 1
            result_by_id[r['event_id']] = r
            if 'run_id' in r:
                by_run[r['run_id']]['events'] += 1
                by_run[r['run_id']][r['status']] += 1
                by_run[r['run_id']].update(r['versions'])
            if 'reaction_id' in r:
                item = gt_reactions[r['reaction_id']]
                item['events'] += 1
                item['matches'].update(r['versions'])
                item['matches']['either'] += bool(r['versions'])
    for pop, n in totals.items():
        if sum(counts[pop].values()) != n:
            raise ValueError('result denominator mismatch')
    program_counts = defaultdict(Counter)
    with gzip.open(OLD/'results/program_event_links.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            events = [result_by_id[i] for i in r['event_ids']]
            c = program_counts[r['run_id']]
            c['programs'] += 1
            c['event_occurrences'] += len(events)
            c['compatible_event_occurrences_either'] += sum(bool(x['versions']) for x in events)
            if not events:
                c['no_event_programs'] += 1
            for v in ['either', *VERSIONS]:
                c['all_events_compatible_'+v] += bool(events) and all(
                    bool(x['versions']) if v == 'either' else v in x['versions'] for x in events)
    summary = dict(status='complete', source_hashes=source_hashes, source_commit='8bd5b38e9602c4d45485bf50076d03c292dcc486',
                   populations=totals, status_counts={k: dict(v) for k, v in counts.items()},
                   version_event_coverage={k: dict(v) for k, v in version_counts.items()},
                   by_run={k: dict(v) for k, v in by_run.items()},
                   program_coverage={k: dict(v) for k, v in program_counts.items()},
                   gt_reactions=28967, gt_official_test_denominator=28971, gt_upstream_corrupt_no_trace=4,
                   gt_all_event_reactions={v: sum(bool(r['events']) and r['matches'][v] == r['events']
                                                for r in gt_reactions.values()) for v in ['either', *VERSIONS]},
                   catalog_expansion_counts=catalog_counts, unsupported_template_occurrences=len(unsupported),
                   unsupported_reasons=dict(Counter(r['unsupported_reason'] for r in unsupported)), wall_seconds=wall,
                   interpretation='Full local author-SMARTS transition compatibility; not full mechanism sequence, reagent screen, kinetic feasibility or unique arrow-pairing proof.',
                   normalization='Remove ordinary mapped attached H into H counts; preserve charge/isotope/stereo. Joint private atom binding across BOTH states; all normalized bond/charge/H changes must be covered.',
                   nonmatches_are_not_chemical_negatives=True,
                   experimental_model_sampling=False)
    save_json(out/'summary.json', summary)
    return summary


def main():
    global GROUPS
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t = time.monotonic()
    source_paths = [SOURCE/'Reaction_templates_index.jsonl', SOURCE/'Reaction_templates_20250216_index.jsonl',
                    SOURCE/'Mechanistic_dataset/templates/AcidBase_lookup.py',
                    ROOT/'data/flower_inverse_tool_sft_action_delta_v1/test.jsonl',
                    OLD/'results/event_annotations_v2.jsonl.gz', OLD/'results/program_event_links.jsonl.gz']
    source_hashes = {str(p): sha(p) for p in source_paths}
    if source_hashes[str(source_paths[3])] != '70a06f01a08490d4b5da602056a8f8e2f978128d3da3224d1601186529aeee3f':
        raise ValueError('GT manifest SHA mismatch')
    GROUPS, unsupported, catalog_counts = load_catalog(source_paths[:2], source_paths[2])
    with gzip.open(args.out/'compiled_author_templates.jsonl.gz', 'wt') as f:
        for rules in GROUPS.values():
            for r in rules:
                f.write(dumps(dict(template_id=hashlib.sha256(r.smarts.encode()).hexdigest(),
                                   reaction_smarts=r.smarts, source_refs=r.refs))+'\n')
    with gzip.open(args.out/'unsupported_author_templates.jsonl.gz', 'wt') as f:
        for r in unsupported:
            f.write(dumps(r)+'\n')
    print(dumps(dict(phase='prepare_complete_catalog', rules=sum(map(len,GROUPS.values())), seconds=time.monotonic()-t)), flush=True)
    events, totals = prepare(args.out)
    print(dumps(dict(phase='matching', totals=totals, workers=args.workers)), flush=True)
    done, counter = 0, Counter()
    status = dict(status='running', completed=done, total=len(events), populations=totals, source_hashes=source_hashes)
    save_json(args.out/'progress.json', status)
    chunks = [events[i:i+64] for i in range(0,len(events),64)]
    with gzip.open(args.out/'event_matches.jsonl.gz', 'wt', compresslevel=3) as f:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
            for rows in pool.map(check_chunk, chunks, chunksize=1):
                for r in rows:
                    f.write(dumps(r)+'\n')
                    counter[(r['population'],r['status'])] += 1
                done += len(rows)
                if done % 512 == 0 or done == len(events):
                    f.flush()
                    status.update(completed=done, percent=100*done/len(events), elapsed_seconds=time.monotonic()-t)
                    save_json(args.out/'progress.json', status)
                    print(dumps({k:v for k,v in status.items() if k!='source_hashes'}), flush=True)
    summary = summarize(args.out, totals, time.monotonic()-t, source_hashes, catalog_counts, unsupported)
    save_json(args.out/'progress.json', dict(status='complete', completed=done, total=len(events), percent=100, elapsed_seconds=summary['wall_seconds']))
    print(dumps(summary), flush=True)


if __name__ == '__main__':
    main()
