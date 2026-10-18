"""Bind explicitly authored source/query reviews; never infer family-wide support.

The scientific judgements are inputs, not predictions of this recipe. It only
authenticates original documents, source structures, capped candidates and the
existing full-stereo evidence checker. Unresolved reviews are retained.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from mechet.analysis.literature_checked_registry import bind_preparation
from mechet.analysis.literature_evidence import assess
from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_sources import file_sha256


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return list(map(json.loads, stream))


def compile_reviews(spec_path, analysis_root, traceback_path, cache_root):
    spec_path, analysis_root, traceback_path = map(Path, (spec_path, analysis_root, traceback_path))
    spec = json.loads(spec_path.read_bytes())
    manifest_path = analysis_root / 'analysis_manifest.json'
    if file_sha256(manifest_path) != spec['frozen_analysis_manifest_sha256']:
        raise ValueError('frozen analysis manifest differs')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get('complete') is not True:
        raise ValueError('completed source analysis required')
    for name, record in manifest['output_files'].items():
        if Path(name).name != name or file_sha256(analysis_root / name) != record['sha256']:
            raise ValueError('frozen analysis bytes differ: ' + name)
    if file_sha256(traceback_path) != spec['traceback_sha256']:
        raise ValueError('original candidate traceback differs')
    queries = rows(analysis_root / 'unique_reaction_queries_enriched.jsonl.gz')
    contexts = rows(analysis_root / 'retrieval_query_contexts.jsonl.gz')
    by_key = {r['reaction_key']: r for r in queries}
    by_context = {r['reaction_key']: r for r in contexts}
    if (len(queries) != spec['expected_query_count'] or len(by_key) != len(queries)
            or len(by_context) != len(contexts) or set(by_context) != set(by_key)):
        raise ValueError('query/context population differs')
    hits = defaultdict(list)
    for hit in rows(analysis_root / 'literature_hits.jsonl.gz'):
        if hit['query_reaction_key'] not in by_key:
            raise ValueError('candidate outside frozen query population')
        hits[hit['query_reaction_key']].append(hit)
    trace = {(r['query_reaction_key'], r['original_candidate_source_id']): r for r in rows(traceback_path)}
    bound_sources, reviewed, positives, seen = [], [], [], set()
    fields = ('analogue_compatibility_checked', 'substrate_differences',
              'functional_group_compatibility', 'stereo_compatibility_checked', 'review_limitations')
    for group in spec['sources']:
        source_spec = group['preparation']
        bound = bind_preparation(source_spec, cache_root)
        if any(s['source_id'] == bound['source_id'] for s in bound_sources):
            raise ValueError('duplicate primary preparation identity')
        bound_sources.append(bound)
        source_key = bound['expected_exact_reaction_key']
        for review in group['query_reviews']:
            key = review['query_reaction_key']
            pair = bound['source_id'], key
            if pair in seen:
                raise ValueError('duplicate source/query review')
            seen.add(pair)
            if review['expected_tier'] not in ('B', 'C') or key not in by_key:
                raise ValueError('unknown query or unsupported review tier')
            selected = []
            for hit in hits[key]:
                if hit['reaction_key'] != source_key:
                    continue
                pointer = trace.get((key, hit['source_id']))
                traced = bool(pointer and pointer['candidate_reaction_key'] == source_key and
                              any(p['lookup_url'] == bound['source_url'] for p in pointer['recovered_patents']))
                if hit.get('source_url') == bound['source_url'] or traced:
                    selected.append(hit)
            if not selected:
                raise ValueError('reviewed primary is not an original capped candidate')
            hit = sorted(selected, key=lambda r: r['source_id'])[0]
            if make_reaction(hit['product'], hit['precursors'])['reaction_key'] != source_key:
                raise ValueError('candidate representation differs from independently named source')
            record = {**bound, **{field: review[field] for field in fields},
                      'source_id': bound['source_id'] + ':query:' + key,
                      'analogue_query_reaction_key': key,
                      'source_product': hit['product'], 'source_precursors': hit['precursors'],
                      'independently_transcribed_source_product': source_spec['source_product'],
                      'independently_transcribed_source_precursors': source_spec['source_precursors'],
                      'original_candidate_source_id': hit['source_id'],
                      'mapping_basis': 'original indexed correspondence, not experimentally observed electron mapping'}
            query = {**by_key[key], 'stereo_family_context': by_context[key]}
            result = assess(query, hits[key], [record])
            if result['tier'] != review['expected_tier']:
                raise ValueError('reviewed grade does not pass existing evidence guards')
            reviewed.append({**review, 'source_id': bound['source_id'],
                             'source_reaction_key': source_key, 'original_candidate_source_id': hit['source_id'],
                             'assessed_tier': result['tier'], 'physical_mechanism_verified': False})
            if result['tier'] == 'B':
                positives.append(record)
    provenance = dict(review_spec_sha256=file_sha256(spec_path), producer_sha256=file_sha256(__file__),
                      frozen_analysis_manifest_sha256=file_sha256(manifest_path),
                      traceback_sha256=file_sha256(traceback_path), new_model_inference=False)
    audit = dict(schema='lgrv_query_scoped_grouped_primary_review_v1', complete=True,
                 sources=bound_sources, reviews=reviewed, unique_reaction_queries=len(by_key),
                 scientific_judgement_supplied_by='explicit assistant original-source/query review, not this compiler',
                 independent_policy_calibration=False, **provenance)
    registry = dict(schema='lgrv_primary_source_registry_v1',
                    scope='Only explicitly reviewed query-scoped analogues passing original capped-candidate/full-stereo guards.',
                    records=positives, **provenance)
    return audit, registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['review-spec', 'analysis-root', 'traceback', 'cache-root', 'audit-output', 'registry-output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.audit_output.exists() or args.registry_output.exists():
        raise FileExistsError('never overwrite an existing source review')
    audit, registry = compile_reviews(args.review_spec, args.analysis_root, args.traceback, args.cache_root)
    for path, value in [(args.audit_output, audit), (args.registry_output, registry)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(preparations=len(audit['sources']), reviewed_pairs=len(audit['reviews']),
                         query_scoped_supports=len(registry['records']),
                         audit_sha256=file_sha256(args.audit_output), registry_sha256=file_sha256(args.registry_output))), flush=True)


if __name__ == '__main__':
    main()
