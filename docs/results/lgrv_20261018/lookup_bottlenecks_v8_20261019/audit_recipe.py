"""Read-only full-population SQL diagnosis; never changes retrieval/evidence."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from mechet.analysis.literature_frozen_index import FrozenReactionIndex
from mechet.analysis.literature_sources import file_sha256


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root
    manifest_path = root / 'analysis_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get('complete') is not True:
        raise ValueError('completed source analysis required')
    names = ('source_corpus_manifest.json', 'unique_reaction_queries_enriched.jsonl.gz',
             'retrieval_query_contexts.jsonl.gz', 'literature_hits.jsonl.gz')
    for name in names:
        if file_sha256(root / name) != manifest['output_files'][name]['sha256']:
            raise ValueError('source input changed: ' + name)
    corpus = json.loads((root / names[0]).read_bytes())
    queries = rows(root / names[1])
    contexts = {r['reaction_key']: r for r in rows(root / names[2])}
    hits = Counter(r['query_reaction_key'] for r in rows(root / names[3]))
    if len(queries) != len(contexts) or {q['reaction_key'] for q in queries} != contexts.keys():
        raise ValueError('query/context populations differ')
    if not set(hits) <= contexts.keys():
        raise ValueError('foreign indexed hit')
    out = args.output_root
    if out.exists():
        raise FileExistsError(out)
    print(json.dumps({'phase': 'authenticate_original_index',
                      'queries': len(queries), 'index': corpus['index_path']}), flush=True)
    index = FrozenReactionIndex(corpus['index_path'], index_sha256=corpus['index_sha256'])
    collected = []
    try:
        for number, query in enumerate(queries, 1):
            key = query['reaction_key']
            context = contexts[key]
            exact = index.connection.execute(
                'SELECT count(*) FROM reactions WHERE reaction_key=?', (key,)).fetchone()[0]
            legacy = None
            aliases = context['legacy_family_keys']
            if context['status'] == 'available':
                if not aliases or context['legacy_alias_space_complete'] is not True:
                    raise ValueError('available context has no complete lookup aliases')
                sql = ('SELECT count(*) FROM reactions WHERE family_key IN (' +
                       ','.join('?' for _ in aliases) + ') AND reaction_key!=?')
                legacy = index.connection.execute(sql, (*aliases, key)).fetchone()[0]
            if exact and not hits[key]:
                raise ValueError('exact index rows exist but frozen retrieval has no hit')
            if hits[key]:
                cause = 'retained_indexed_candidate'
            elif context['status'] != 'available':
                cause = 'no_hit_reaction_center_unavailable'
            elif legacy:
                cause = 'no_hit_legacy_candidates_full_context_rejected'
            else:
                cause = 'no_hit_no_exact_or_legacy_family_candidate'
            collected.append({'reaction_key': key, 'context_status': context['status'],
                              'exact_index_rows': exact, 'legacy_family_index_rows': legacy,
                              'saved_distinct_source_hits': hits[key], 'lookup_stage': cause,
                              'reaction_smiles': query.get('reaction_smiles'),
                              'evidence_grade_changed': False})
            if number % 1000 == 0:
                print(json.dumps({'phase': 'lookup_bottleneck_audit', 'done': number,
                                  'total': len(queries)}), flush=True)
    finally:
        index.close()
    for name in names:
        if file_sha256(root / name) != manifest['output_files'][name]['sha256']:
            raise ValueError('source input changed during audit: ' + name)
    if file_sha256(manifest_path) != 'a1b11d5c6c44e009e79ac92c28003bb44bb711f835ffb99715dc09b5e515b497':
        raise ValueError('source analysis identity differs from requested v8')
    counts = Counter(r['lookup_stage'] for r in collected)
    if sum(counts.values()) != len(queries) or counts['retained_indexed_candidate'] != len(hits):
        raise ValueError('audit partition failed')
    summary = {'schema': 'lgrv_full_lookup_bottleneck_audit_v1', 'complete': True,
               'unique_reaction_queries': len(queries), 'lookup_stage_counts': dict(counts),
               'index_rows_are_representation_records_not_unique_reactions': True,
               'source_index_path': corpus['index_path'], 'source_index_sha256': corpus['index_sha256'],
               'source_analysis_manifest_sha256': file_sha256(manifest_path),
               'producer_sha256': file_sha256(__file__),
               'source_files': {str(root / n): file_sha256(root / n) for n in names},
               'new_model_inference': False, 'new_retrieval_policy': False,
               'evidence_grades_changed': False, 'chemical_validity_established': False,
               'interpretation': 'Lookup-stage missingness only. No-index-key match is not absent literature; rejected legacy candidates are not automatically compatible chemistry.'}
    out.mkdir(parents=True, exist_ok=False)
    row_path = out / 'query_lookup_bottlenecks.jsonl.gz'
    with gzip.open(row_path, 'wt') as stream:
        for row in collected:
            stream.write(json.dumps(row) + '\n')
    summary['output_files'] = {row_path.name: {'sha256': file_sha256(row_path)}}
    with (out / 'lookup_bottleneck_manifest.json').open('x') as stream:
        stream.write(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
