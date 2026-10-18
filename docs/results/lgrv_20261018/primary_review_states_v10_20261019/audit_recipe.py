"""Track every unresolved indexed query without conflating review with support."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

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
    for source in [*scopes.values(), mismatch['checked_named_source']]:
        if not _passage_checked(source):
            raise ValueError('previously reviewed primary binding no longer authenticates')
    cache = Path('/aaa/fionafyang/buddy1/whaleywang/MechET/data/raw/literature/primary_documents_bulk_20261018_v2')
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
        elif key == mismatch['query_reaction_key']:
            state = 'checked_one_source_structure_mismatch_other_candidates_pending'
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
    sources = [root / 'analysis_manifest.json', *[root / n for n in names], trace_path, scope_path, mismatch_path]
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
