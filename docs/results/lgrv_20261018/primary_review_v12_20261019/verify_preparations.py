"""Rebind all three reviewed preparations and their original frozen candidates.

CPU-only provenance/chemistry checks. Hypothesis tests are software checks, not
new model predictions or independent experimental confirmation of an analogue.
"""
import argparse
import gzip
import json
from pathlib import Path

from mechet.analysis.literature_checked_registry import bind_preparation
from mechet.analysis.literature_evidence import assess
from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_sources import file_sha256
from mechet.analysis.literature_stereo_family import family_context


def rows(path):
    with gzip.open(path, 'rt') as stream:
        yield from map(json.loads, stream)


def verify(analysis_root, cache_root, spec_path, registry_path):
    analysis_root = Path(analysis_root)
    manifest_path = analysis_root / 'analysis_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get('complete') is not True:
        raise ValueError('completed frozen analysis required')
    names = ['unique_reaction_queries_enriched.jsonl.gz',
             'retrieval_query_contexts.jsonl.gz', 'literature_hits.jsonl.gz']
    for name in names:
        if file_sha256(analysis_root / name) != manifest['output_files'][name]['sha256']:
            raise ValueError('frozen source changed: ' + name)
    queries = {r['reaction_key']: r for r in rows(analysis_root / names[0])}
    contexts = {r['reaction_key']: r for r in rows(analysis_root / names[1])}
    hits = {}
    for hit in rows(analysis_root / names[2]):
        hits.setdefault(hit['query_reaction_key'], []).append(hit)
    specs = json.loads(Path(spec_path).read_bytes())['records']
    registry = json.loads(Path(registry_path).read_bytes())
    if registry['review_spec_sha256'] != file_sha256(spec_path):
        raise ValueError('registry no longer binds review specification')
    records = {r['source_id']: r for r in registry['records']}
    if len(records) != len(specs):
        raise ValueError('reviewed source population changed')
    result = []
    for spec in specs:
        bound = bind_preparation(spec, cache_root)
        if records.get(spec['source_id']) != bound:
            raise ValueError('primary bytes or named structure changed')
        key = bound.get('analogue_query_reaction_key', bound['expected_exact_reaction_key'])
        query = {**queries[key], 'stereo_family_context': contexts[key]}
        expected = 'A'
        if 'analogue_query_reaction_key' in bound:
            named = make_reaction(bound['independently_transcribed_source_product'],
                                  bound['independently_transcribed_source_precursors'])
            if named['reaction_key'] != bound['expected_exact_reaction_key']:
                raise ValueError('named source differs from indexed mapped source')
            if not any(h['reaction_key'] == named['reaction_key'] for h in hits.get(key, [])):
                raise ValueError('analogue source was not an original capped candidate')
            local = family_context('.'.join(bound['source_precursors']), bound['source_product'])
            if (local['status'] != 'available' or contexts[key]['status'] != 'available'
                    or local['family_key_v2'] != contexts[key]['family_key_v2']):
                raise ValueError('analogue does not retain frozen full-stereo local environment')
            unbound = {k: v for k, v in bound.items() if k != 'analogue_query_reaction_key'}
            if assess(query, hits[key], [unbound])['tier'] != 'C':
                raise ValueError('unscoped analogue review incorrectly promotes this query')
            expected = 'B'
        evidence = assess(query, hits.get(key, []), [bound])
        if evidence['tier'] != expected or evidence['physical_mechanism_verified'] is not False:
            raise ValueError('source grade or mechanism boundary differs')
        result.append({'source_id': bound['source_id'], 'query_reaction_key': key,
                       'source_reaction_key': bound['expected_exact_reaction_key'],
                       'tier': expected, 'raw_document_sha256': bound['raw_document_sha256'],
                       'document_sha256': bound['document_sha256']})
    return {'schema': 'lgrv_query_bound_preparation_review_v1', 'complete': True,
            'frozen_analysis_manifest_sha256': file_sha256(manifest_path),
            'review_spec_sha256': file_sha256(spec_path),
            'registry_sha256': file_sha256(registry_path),
            'producer_sha256': file_sha256(__file__), 'records': result,
            'new_model_inference': False, 'physical_mechanism_verified': False,
            'population_reaggregation_completed_by_this_recipe': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['analysis-root', 'cache-root', 'review-spec', 'registry', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.analysis_root, args.cache_root, args.review_spec, args.registry)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'bound_preparations': len(result['records']),
                      'output_sha256': file_sha256(args.output)}), flush=True)


if __name__ == '__main__':
    main()
