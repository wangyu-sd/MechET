"""Describe literature missingness without changing retrieval or evidence grades.

All frozen queries are retained. A bibliographic data-set citation, a cached
document, an indexed structural candidate and a checked performed experiment
are different stages; none is silently substituted for another.
"""
from collections import Counter, defaultdict
import argparse
import gzip
import json
from pathlib import Path

from .literature_sources import file_sha256


def _unique(rows, label):
    result = {}
    for row in rows:
        key = row['reaction_key']
        if key in result:
            raise ValueError('duplicate ' + label + ' reaction identity')
        result[key] = row
    return result


def summarize(queries, contexts, evidence, hits):
    queries = _unique(queries, 'query')
    contexts = _unique(contexts, 'context')
    evidence = _unique(evidence, 'evidence')
    if not queries or queries.keys() != contexts.keys() or queries.keys() != evidence.keys():
        raise ValueError('query/context/evidence populations differ')
    grouped = defaultdict(list)
    nonpatent = Counter()
    for hit in hits:
        key = hit['query_reaction_key']
        if key not in queries or hit.get('match_type') not in ('exact', 'local_edit_analogue_candidate'):
            raise ValueError('unknown candidate query or match type')
        if not hit.get('source_id') or not hit.get('source_url'):
            raise ValueError('missing candidate source identity')
        grouped[key].append(hit)
        if not hit.get('patent_id'):
            nonpatent[hit['source_url']] += 1
    rows, queue = [], []
    counts = Counter({k: 0 for k in ('indexed_queries', 'no_index_hit_queries',
        'exact_candidate_queries', 'analogue_candidate_queries', 'source_supported_queries',
        'supported_with_index_hit', 'supported_without_index_hit', 'unresolved_with_candidates')})
    for key, query in sorted(queries.items()):
        ev, ctx, candidates = evidence[key], contexts[key], grouped[key]
        tier = ev['tier']
        supported = tier in ('A', 'B')
        if tier not in ('A', 'B', 'C', 'D') or ev.get('supported') is not supported:
            raise ValueError('inconsistent evidence grade and support flag')
        exact = sum(h['match_type'] == 'exact' for h in candidates)
        analogue = len(candidates) - exact
        counts['indexed_queries' if candidates else 'no_index_hit_queries'] += 1
        counts['exact_candidate_queries'] += bool(exact)
        counts['analogue_candidate_queries'] += bool(analogue)
        counts['source_supported_queries'] += supported
        if supported:
            counts['supported_with_index_hit' if candidates else 'supported_without_index_hit'] += 1
            action = 'already_source_supported'
        elif tier == 'D':
            action = 'inspect_documented_conflict'
        elif candidates:
            counts['unresolved_with_candidates'] += 1
            action = 'check_exact_primary_example' if exact else 'review_analogue_compatibility'
        else:
            action = ('audit_frozen_retrieval_coverage' if ctx['status'] == 'available'
                      else 'resolve_reaction_center_then_retrieve')
        record = {'reaction_key': key, 'tier': tier, 'source_supported': supported,
            'index_hit_count': len(candidates), 'exact_source_records': exact,
            'analogue_source_records': analogue, 'context_status': ctx['status'],
            'next_action': action, 'chemical_validity': 'not_determined_by_retrieval',
            'reaction_smiles': query.get('reaction_smiles'),
            'candidate_sources': [{'source_id': h['source_id'], 'source_url': h['source_url'],
                'match_type': h['match_type'], 'source_reaction_key': h['reaction_key'],
                'patent_id': h.get('patent_id'), 'doi': h.get('doi')} for h in candidates]}
        rows.append(record)
        if candidates and tier == 'C':
            queue.append(record)
    report = {'schema': 'lgrv_coverage_missingness_audit_v1',
        'unique_reaction_queries': len(queries), 'stage_counts': dict(counts),
        'context_statuses': dict(Counter(x['status'] for x in contexts.values())),
        'no_hit_context_statuses': dict(Counter(r['context_status'] for r in rows if not r['index_hit_count'])),
        'unique_nonpatent_urls': len(nonpatent), 'nonpatent_reference_counts': dict(sorted(nonpatent.items())),
        'new_evidence_grades': False, 'new_model_inference': False,
        'population_chemical_accuracy_established': False,
        'interpretation': 'Retrieval and primary-review coverage, not chemical success or failure rate.'}
    return report, rows, queue


def recover_patent_references(index, hits):
    """Trace a retrieved candidate's exact chemistry to alternate patent records.

    This never substitutes the query's structures for an analogue's structures.
    These are alternate bibliographic pointers, not additional independently
    sampled chemical candidates and not proof of performance or patent family.
    """
    from .literature_primary_acquisition import patent_reference
    from .literature_reaction_keys import make_reaction
    results = []
    for hit in hits:
        if hit.get('patent_id'):
            continue
        chemical = make_reaction(hit['product'], hit['precursors'])
        if chemical['reaction_key'] != hit['reaction_key']:
            raise ValueError('candidate structure does not match its frozen identity')
        patents = {}
        cursor = index.connection.execute('SELECT record FROM reactions WHERE reaction_key=? ORDER BY source_id',
                                          (hit['reaction_key'],))
        for (encoded,) in cursor:
            record = json.loads(encoded)
            if not record.get('patent_id'):
                continue
            other = make_reaction(record['product'], record['precursors'])
            if other['reaction_key'] != hit['reaction_key']:
                raise ValueError('indexed candidate structure differs from exact lookup')
            reference = patent_reference(record['patent_id'], record['source_url'])
            url = reference['lookup_url']
            if url not in patents:
                patents[url] = {**reference, 'source_ids': [], 'source_locators': []}
            patents[url]['source_ids'].append(record['source_id'])
            patents[url]['source_locators'].append(record.get('source_locator'))
        results.append({'query_reaction_key': hit['query_reaction_key'],
            'original_candidate_source_id': hit['source_id'],
            'original_candidate_url': hit['source_url'], 'match_type': hit['match_type'],
            'candidate_reaction_key': hit['reaction_key'],
            'recovered_patents': [patents[k] for k in sorted(patents)],
            'performed_example_checked': False, 'new_evidence_grade': False,
            'chemical_candidate_budget_changed': False,
            'relation': 'same_complete_candidate_reaction_not_inferred_patent_family'})
    return results


def _read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def export_audit(analysis_root, output_root, *, traceback_patents=False):
    """Verify archived inputs then write a separate, full-population audit."""
    root, output = Path(analysis_root).resolve(), Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = root/'analysis_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get('complete') is not True:
        raise ValueError('analysis is incomplete')
    names = ['unique_reaction_queries_enriched.jsonl.gz', 'retrieval_query_contexts.jsonl.gz',
             'reaction_evidence.jsonl.gz', 'literature_hits.jsonl.gz']
    if traceback_patents:
        names.append('source_corpus_manifest.json')
    sources = {}
    for name in names:
        digest = file_sha256(root/name)
        if manifest['output_files'][name]['sha256'] != digest:
            raise ValueError('analysis input checksum differs: ' + name)
        sources[name] = {'path': str(root/name), 'sha256': digest}
    queries, contexts, evidence, hits = [_read_rows(root/n) for n in names[:4]]
    report, rows, queue = summarize(queries, contexts, evidence, hits)
    trace = None
    if traceback_patents:
        from .literature_frozen_index import FrozenReactionIndex
        corpus = json.loads((root/'source_corpus_manifest.json').read_bytes())
        if corpus.get('complete') is not True:
            raise ValueError('source corpus is incomplete')
        print(json.dumps({'phase': 'validate_frozen_index', 'index_path': corpus['index_path']}), flush=True)
        index = FrozenReactionIndex(corpus['index_path'], index_sha256=corpus['index_sha256'])
        try:
            trace = recover_patent_references(index, hits)
        finally:
            index.close()
        report['patent_traceback'] = {
            'nonpatent_candidate_records': len(trace),
            'candidate_records_with_patent_pointers': sum(bool(r['recovered_patents']) for r in trace),
            'unique_recovered_patent_urls': len({p['lookup_url'] for r in trace for p in r['recovered_patents']}),
            'additional_experiment_support_claimed': False,
            'source_index_sha256': corpus['index_sha256']}
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in [('query_coverage_audit.jsonl.gz', rows), ('primary_review_queue.jsonl.gz', queue),
                          ('candidate_patent_traceback.jsonl.gz', trace)]:
        if payload is None:
            continue
        with gzip.open(output/name, 'wt', encoding='utf-8') as f:
            for row in payload:
                f.write(json.dumps(row, sort_keys=True)+'\n')
    report.update(analysis_manifest_path=str(manifest_path), analysis_manifest_sha256=file_sha256(manifest_path),
                  input_files=sources, producer_sha256=file_sha256(__file__), complete=True)
    report['output_files'] = {p.name: {'sha256': file_sha256(p)} for p in sorted(output.iterdir())}
    (output/'coverage_audit_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'phase': 'coverage_audit_complete', 'queries': len(rows),
                      'review_queue': len(queue), 'output_root': str(output)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--traceback-patents', action='store_true')
    args = parser.parse_args()
    export_audit(args.analysis_root, args.output_root, traceback_patents=args.traceback_patents)


if __name__ == '__main__':
    main()
