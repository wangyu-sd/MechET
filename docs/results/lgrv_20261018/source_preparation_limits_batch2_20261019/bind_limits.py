"""Authenticate source-side limits without inventing support or chemical failure.

This version additionally binds validated zero-padded patent lookup aliases.
The original v1 recipe and all its sealed artifacts remain unchanged.
The authored findings are inputs. This recipe binds them to actual primary
bytes and the original capped candidate, not to a fabricated experiment.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from mechet.analysis.literature_primary_acquisition import patent_reference
from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_sources import file_sha256, normalize_primary_text


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return list(map(json.loads, stream))


def bind_limits(spec_path, analysis_root, traceback_path, cache_root):
    spec_path, analysis_root, traceback_path = map(Path, (spec_path, analysis_root, traceback_path))
    spec = json.loads(spec_path.read_bytes())
    manifest_path = analysis_root / 'analysis_manifest.json'
    if file_sha256(manifest_path) != spec['frozen_analysis_manifest_sha256']:
        raise ValueError('frozen analysis manifest differs')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get('complete') is not True:
        raise ValueError('completed frozen analysis required')
    for name, record in manifest['output_files'].items():
        if Path(name).name != name or file_sha256(analysis_root / name) != record['sha256']:
            raise ValueError('sealed analysis output changed')
    if file_sha256(traceback_path) != spec['traceback_sha256']:
        raise ValueError('original source traceback changed')
    queries = rows(analysis_root / 'unique_reaction_queries_enriched.jsonl.gz')
    evidence = rows(analysis_root / 'reaction_evidence.jsonl.gz')
    by_key = {r['reaction_key']: r for r in queries}
    by_evidence = {r['reaction_key']: r for r in evidence}
    if (len(queries) != spec['expected_query_count'] or len(by_key) != len(queries)
            or len(by_evidence) != len(evidence) or set(by_key) != set(by_evidence)):
        raise ValueError('query/evidence population differs')
    hits = defaultdict(list)
    for hit in rows(analysis_root / 'literature_hits.jsonl.gz'):
        if hit['query_reaction_key'] not in by_key:
            raise ValueError('candidate outside frozen query population')
        hits[hit['query_reaction_key']].append(hit)
    traces = {(r['query_reaction_key'], r['original_candidate_source_id']): r for r in rows(traceback_path)}
    sources, reviews, seen, source_ids = [], [], set(), set()
    quotes_by_url = defaultdict(set)
    for source in spec['sources']:
        if (not source.get('source_id') or source['source_id'] in source_ids or
                source.get('source_preparation_status') != 'not_established_for_indexed_precursors' or
                not source.get('source_review') or not source.get('query_reviews')):
            raise ValueError('nonpositive source finding and scoped queries required')
        source_ids.add(source['source_id'])
        url = source['source_url']
        receipt_path = Path(cache_root) / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        if not receipt_path.is_file():
            raise ValueError('reviewed original primary is not cached')
        receipt = json.loads(receipt_path.read_bytes())
        raw, text = Path(receipt['raw_document_path']), Path(receipt['document_path'])
        if (receipt['source_url'] != url or receipt['document_sha256'] != source['document_sha256'] or
                file_sha256(raw) != receipt['raw_document_sha256'] or
                file_sha256(text) != receipt['document_sha256']):
            raise ValueError('cached primary identity differs')
        document = text.read_text(encoding='utf-8')
        if (receipt.get('text_normalization') != 'html_data_join_unicode_whitespace_v1' or
                normalize_primary_text(raw.read_bytes(), receipt.get('source_encoding', 'utf-8')) != document):
            raise ValueError('primary normalization differs from original bytes')
        witnesses = []
        for witness in source.get('witnesses', []):
            quote = witness.get('text')
            if not quote or not witness.get('locator') or document.count(quote) != 1:
                raise ValueError('source witness absent, ambiguous or unlocated')
            quotes_by_url[url].add(quote)
            witnesses.append({**witness, 'offset': document.index(quote),
                              'sha256': hashlib.sha256(quote.encode()).hexdigest()})
        if not witnesses or sum(len(q.split()) for q in quotes_by_url[url]) > 25:
            raise ValueError('short source witnesses required')
        candidate = make_reaction(source['indexed_candidate_product'], source['indexed_candidate_precursors'])
        key = source['expected_candidate_reaction_key']
        if candidate['reaction_key'] != key:
            raise ValueError('independently transcribed candidate identity differs')
        bound = {k: v for k, v in source.items() if k != 'query_reviews'}
        bound.update(witnesses=witnesses, source_receipt_path=str(receipt_path.resolve()),
                     source_receipt_sha256=file_sha256(receipt_path),
                     **{k: receipt[k] for k in ('raw_document_path', 'raw_document_sha256',
                                                'document_path', 'document_sha256')},
                     candidate_experimentally_established=False, source_human_reviewed=False)
        sources.append(bound)
        for review in source['query_reviews']:
            query = review['query_reaction_key']
            pair = source['source_id'], query
            if pair in seen or query not in by_key or not review.get('review_limitations'):
                raise ValueError('duplicate, unknown or unreviewed query')
            seen.add(pair)
            if by_evidence[query]['tier'] != 'C' or by_evidence[query].get('supported') is not False:
                raise ValueError('a nonpositive finding cannot downgrade an established support')
            matched, aliases = [], []
            for hit in hits[query]:
                if hit['reaction_key'] != key:
                    continue
                trace = traces.get((query, hit['source_id']))
                traced = bool(trace and trace['candidate_reaction_key'] == key and
                              any(p['lookup_url'] == url for p in trace['recovered_patents']))
                lookup = (patent_reference(hit['patent_id'], hit['source_url'])
                          if hit.get('patent_id') else None)
                direct = hit.get('source_url') == url or bool(lookup and lookup['lookup_url'] == url)
                if direct or traced:
                    if make_reaction(hit['product'], hit['precursors'])['reaction_key'] != key:
                        raise ValueError('original candidate structure differs')
                    matched.append(hit['source_id'])
                    if direct and lookup:
                        aliases.append({'original_candidate_source_id': hit['source_id'], **lookup})
            if not matched:
                raise ValueError('finding does not belong to an original capped candidate')
            reviews.append({**review, 'source_id': source['source_id'], 'source_candidate_reaction_key': key,
                            'original_candidate_source_ids': sorted(matched),
                            'direct_patent_lookup_bindings': aliases,
                            'review_state': 'checked_candidate_preparation_evidence_unresolved',
                            'evidence_tier_retained': 'C', 'generated_reaction_proven_wrong': False})
    if not sources or not reviews:
        raise ValueError('nonempty source-side review required')
    return dict(schema='lgrv_source_preparation_limits_v2', complete=True, sources=sources, reviews=reviews,
                frozen_analysis_manifest_sha256=file_sha256(manifest_path),
                review_spec_sha256=file_sha256(spec_path), traceback_sha256=file_sha256(traceback_path),
                producer_sha256=file_sha256(__file__), evidence_grades_changed=False,
                scientific_judgement_supplied_by='explicit assistant source review, not this binder',
                new_model_inference=False, independent_policy_calibration=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['review-spec', 'analysis-root', 'traceback', 'cache-root', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('never overwrite a previous source review')
    result = bind_limits(args.review_spec, args.analysis_root, args.traceback, args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(sources=len(result['sources']), reviewed_queries=len(result['reviews']),
                         output_sha256=file_sha256(args.output))), flush=True)


if __name__ == '__main__':
    main()
