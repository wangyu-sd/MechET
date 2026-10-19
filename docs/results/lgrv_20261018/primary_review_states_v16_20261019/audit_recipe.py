"""Track every unresolved indexed query without conflating review with support."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import runpy

from mechet.analysis.literature_evidence import _passage_checked
from mechet.analysis.literature_primary_acquisition import patent_reference
from mechet.analysis.literature_sources import file_sha256, normalize_primary_text


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return list(map(json.loads, stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root
    base = Path('docs/results/lgrv_20261018')
    manifest = json.loads((root / 'analysis_manifest.json').read_bytes())
    if manifest.get('complete') is not True:
        raise ValueError('completed source analysis required')
    names = ('reaction_evidence.jsonl.gz', 'literature_hits.jsonl.gz')
    for name in names:
        if file_sha256(root / name) != manifest['output_files'][name]['sha256']:
            raise ValueError('source analysis input checksum differs')
    evidence = {r['reaction_key']: r for r in rows(root / names[0])}
    grouped = defaultdict(list)
    for hit in rows(root / names[1]):
        grouped[hit['query_reaction_key']].append(hit)
    trace_path = base / 'coverage_traceback_20261018/candidate_patent_traceback.jsonl.gz'
    traces = {(r['query_reaction_key'], r['original_candidate_source_id']): r for r in rows(trace_path)}
    scope_path = base / 'analogue_scope_review_20261019/source_scope_review.json'
    scopes = {r['query_reaction_key']: r for r in json.loads(scope_path.read_bytes())['records']}
    mismatch_path = base / 'primary_structure_review_20261018/source_mismatch_review.json'
    mismatch = json.loads(mismatch_path.read_bytes())
    stereo_root = base / 'primary_stereo_review_20261019'
    stereo_path = stereo_root / 'source_stereo_review.json'
    stereo = json.loads(stereo_path.read_bytes())
    stereo_spec = stereo_root / 'review_spec.json'
    stereo_binder = stereo_root / 'bind_review.py'
    if (file_sha256(stereo_spec) != stereo['review_spec_sha256'] or
            file_sha256(stereo_binder) != stereo['producer_sha256']):
        raise ValueError('stereo-limit review spec/producer changed')
    bind_stereo = runpy.run_path(str(stereo_binder))['bind']
    if bind_stereo(stereo_spec, Path(stereo['source_query_path']),
                   Path('/aaa/fionafyang/buddy1/whaleywang/MechET/data/raw/literature/primary_documents_bulk_20261018_v2')) != stereo:
        raise ValueError('stereo-limit source binding differs')
    for source in [*scopes.values(), mismatch['checked_named_source']]:
        if not _passage_checked(source):
            raise ValueError('previously reviewed primary binding no longer authenticates')
    extra_scope_path = base / 'primary_review_v13_20261019/source_scope_review.json'
    extra_scope_spec = Path('configs/analysis/lgrv_reviewed_analogue_limits_20261019_v2.json')
    extra_scope = json.loads(extra_scope_path.read_bytes())
    from mechet.analysis.literature_checked_registry import bind_preparation
    cache = Path('/aaa/fionafyang/buddy1/whaleywang/MechET/data/raw/literature/primary_documents_bulk_20261018_v2')
    scoped_specs = json.loads(extra_scope_spec.read_bytes())['records']
    if [bind_preparation(s, cache) for s in scoped_specs] != extra_scope['records']:
        raise ValueError('additional analogue scope review no longer binds original source')
    for record in extra_scope['records']:
        scopes[record['analogue_query_reaction_key']] = record
    grouped_recipe = base / 'grouped_primary_review_20261019/compile_reviews.py'
    compile_grouped = runpy.run_path(str(grouped_recipe))['compile_reviews']
    analysis_base = Path('/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/paper/nmi_literature_evidence')
    bundles = [
        ('grouped_primary_review_20261019', 'v1', 'v13'),
        ('grouped_primary_review_batch2_20261019', 'v2', 'v14'),
        ('grouped_primary_review_batch3_20261019', 'v3', 'v14'),
        ('grouped_primary_review_batch4_20261019', 'v4', 'v15'),
        ('grouped_primary_review_batch5_20261019', 'v5', 'v15'),
        ('grouped_primary_review_batch6_20261019', 'v6', 'v15'),
    ]
    grouped_source_paths = []
    reviewed_pairs = set()
    for folder, spec_version, frozen_version in bundles:
        grouped_path = base / folder / 'source_family_review.json'
        grouped_spec = Path('configs/analysis/lgrv_grouped_primary_reviews_20261019_' + spec_version + '.json')
        grouped_review = json.loads(grouped_path.read_bytes())
        if (file_sha256(grouped_spec) != grouped_review['review_spec_sha256'] or
                file_sha256(grouped_recipe) != grouped_review['producer_sha256']):
            raise ValueError('grouped source-review specification/producer changed')
        old_root = analysis_base / ('source_checked_20261019_' + frozen_version)
        rebound, _ = compile_grouped(grouped_spec, old_root, trace_path, cache)
        if rebound != grouped_review:
            raise ValueError('grouped primary/query binding changed')
        source_map = {r['source_id']: r for r in grouped_review['sources']}
        for review in grouped_review['reviews']:
            pair = review['source_id'], review['query_reaction_key']
            if pair in reviewed_pairs:
                raise ValueError('duplicate cross-bundle source/query review')
            reviewed_pairs.add(pair)
            if review['assessed_tier'] == 'C':
                scopes[review['query_reaction_key']] = {**source_map[review['source_id']], **review}
        grouped_source_paths.extend([grouped_path, grouped_spec])
    limits_root = base / 'source_preparation_limits_20261019'
    limits_recipe = limits_root / 'bind_limits.py'
    bind_limits = runpy.run_path(str(limits_recipe))['bind_limits']
    limit_inputs = [limits_recipe]
    preparation_limits = set()
    limit_pairs = set()
    for filename, version in [('source_limits_review.json', 'v1'),
                              ('phosphate_source_limits_review.json', 'v2')]:
        limits_path = limits_root / filename
        limits_spec = Path('configs/analysis/lgrv_source_preparation_limits_20261019_' + version + '.json')
        saved_limits = json.loads(limits_path.read_bytes())
        if (file_sha256(limits_spec) != saved_limits['review_spec_sha256'] or
                file_sha256(limits_recipe) != saved_limits['producer_sha256']):
            raise ValueError('source preparation-limit spec/producer differs')
        if bind_limits(limits_spec, analysis_base / 'source_checked_20261019_v15',
                       trace_path, cache) != saved_limits:
            raise ValueError('source preparation-limit binding differs')
        for review in saved_limits['reviews']:
            pair = review['source_id'], review['query_reaction_key']
            if pair in limit_pairs:
                raise ValueError('duplicate preparation-limit source/query pair')
            limit_pairs.add(pair)
            preparation_limits.add(review['query_reaction_key'])
        limit_inputs.extend([limits_path, limits_spec])
    macro_root = base / 'primary_macrocycle_stereo_review_20261019'
    macro_path = macro_root / 'source_stereo_review.json'
    macro_spec = macro_root / 'review_spec.json'
    macro_binder = macro_root / 'bind_review.py'
    macro = json.loads(macro_path.read_bytes())
    if (file_sha256(macro_spec) != macro['review_spec_sha256'] or
            file_sha256(macro_binder) != macro['producer_sha256']):
        raise ValueError('macrocycle source-image spec/producer changed')
    bind_macro = runpy.run_path(str(macro_binder))['bind']
    if bind_macro(macro_spec, Path(macro['source_query_path']), cache) != macro:
        raise ValueError('macrocycle primary page/image binding changed')
    stereo_keys = {stereo['query_reaction_key'], macro['query_reaction_key']}
    documents = {}

    def document(url):
        if url in documents:
            return documents[url]
        receipt_path = cache / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        item = {'lookup_url': url, 'cached_document_available': False,
                'performed_example_inferred_from_acquisition': False}
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_bytes())
            raw, text = Path(receipt['raw_document_path']), Path(receipt['document_path'])
            if receipt['source_url'] != url or file_sha256(raw) != receipt['raw_document_sha256'] or file_sha256(text) != receipt['document_sha256']:
                raise ValueError('cached primary identity changed')
            if (receipt.get('text_normalization') != 'html_data_join_unicode_whitespace_v1' or
                    normalize_primary_text(raw.read_bytes(), receipt.get('source_encoding', 'utf-8')) != text.read_text()):
                raise ValueError('cached primary normalization differs')
            item.update(cached_document_available=True,
                        receipt_path=str(receipt_path), receipt_sha256=file_sha256(receipt_path),
                        raw_document_sha256=receipt['raw_document_sha256'],
                        document_sha256=receipt['document_sha256'])
        documents[url] = item
        return item

    result = []
    for key, hits in sorted(grouped.items()):
        if evidence[key]['tier'] != 'C':
            continue
        candidates = []
        for hit in hits:
            if hit.get('patent_id'):
                pointers = [patent_reference(hit['patent_id'], hit['source_url'])]
            else:
                trace = traces.get((key, hit['source_id']))
                if trace and (trace['query_reaction_key'] != key or trace['candidate_reaction_key'] != hit['reaction_key']):
                    raise ValueError('traceback belongs to another query/candidate')
                pointers = trace['recovered_patents'] if trace else []
            candidates.append({'source_id': hit['source_id'], 'match_type': hit['match_type'],
                               'source_reaction_key': hit['reaction_key'], 'original_source_url': hit['source_url'],
                               'primary_pointers': [document(p['lookup_url']) for p in pointers]})
        available = sum(d['cached_document_available'] for c in candidates for d in c['primary_pointers'])
        pointers = sum(len(c['primary_pointers']) for c in candidates)
        if key in scopes:
            state = 'checked_performed_candidate_applicability_unresolved'
        elif key in preparation_limits:
            state = 'checked_candidate_preparation_evidence_unresolved'
        elif key == mismatch['query_reaction_key']:
            state = 'checked_one_source_structure_mismatch_other_candidates_pending'
        elif key in stereo_keys:
            state = 'checked_performed_candidate_stereo_identity_unresolved'
        elif available:
            state = 'cached_primary_documents_preparation_review_pending'
        elif pointers:
            state = 'primary_pointers_available_document_unavailable'
        else:
            state = 'original_preparation_pointer_unresolved'
        result.append({'reaction_key': key, 'population_tier': 'C', 'review_state': state,
                       'indexed_candidate_count': len(candidates), 'primary_pointer_occurrences': pointers,
                       'cached_primary_pointer_occurrences': available, 'candidates': candidates,
                       'query_experimentally_disproven': False, 'review_state_is_not_evidence_grade': True})
    if len(result) != sum(bool(grouped.get(k)) and v['tier'] == 'C' for k, v in evidence.items()):
        raise ValueError('not every unresolved indexed query was retained')
    output = args.output_root
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    p = output / 'primary_review_states.jsonl.gz'
    with gzip.open(p, 'wt') as stream:
        for row in result:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    sources = [root / 'analysis_manifest.json', *[root / n for n in names], trace_path, scope_path, mismatch_path,
               stereo_path, stereo_spec, stereo_binder, extra_scope_path, extra_scope_spec,
               macro_path, macro_spec, macro_binder, grouped_recipe, *grouped_source_paths, *limit_inputs]
    receipt = {'schema': 'lgrv_unresolved_primary_review_states_v1', 'complete': True,
               'unresolved_indexed_queries': len(result),
               'review_state_counts': dict(Counter(r['review_state'] for r in result)),
               'unique_referenced_primary_documents': len(documents),
               'unique_available_primary_documents': sum(d['cached_document_available'] for d in documents.values()),
               'source_files': {str(s): file_sha256(s) for s in sources},
               'producer_sha256': file_sha256(__file__),
               'output_files': {p.name: {'sha256': file_sha256(p)}},
               'new_model_inference': False, 'evidence_grades_changed': False,
               'independent_policy_calibration': False,
               'interpretation': 'Review progress, document availability and chemical support are distinct. C does not mean invalid; a cached primary is not a checked experiment.'}
    with (output / 'review_states_manifest.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
