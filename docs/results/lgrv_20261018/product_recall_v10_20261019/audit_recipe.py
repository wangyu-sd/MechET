"""Read-only product-identity audit; a matching product is NOT reaction evidence."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sqlite3

from mechet.analysis.literature_reaction_keys import canonical_components
from mechet.analysis.literature_sources import file_sha256

PRODUCTS = None


def initialize(products):
    global PRODUCTS
    PRODUCTS = products


def check_batch(batch):
    matches = []
    failed = []
    for source_id, reaction_key, product in batch:
        try:
            identity = canonical_components(product)
        except (ValueError, RuntimeError):
            failed.append(source_id)
            continue
        if identity in PRODUCTS:
            matches.append((identity, source_id, reaction_key))
    return len(batch), matches, failed


def batches(cursor, size=1000):
    while True:
        batch = cursor.fetchmany(size)
        if not batch:
            return
        yield batch


def rows(path):
    with gzip.open(path, 'rt') as handle:
        yield from (json.loads(line) for line in handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root, out = args.analysis_root, args.output_root
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((root/'analysis_manifest.json').read_bytes())
    assert manifest['complete'] is True
    inputs = [root/'analysis_manifest.json', root/'source_corpus_manifest.json',
              root/'unique_reaction_queries_enriched.jsonl.gz', root/'literature_hits.jsonl.gz']
    input_hashes = {str(path): file_sha256(path) for path in inputs}
    for path in inputs[1:]:
        assert input_hashes[str(path)] == manifest['output_files'][path.name]['sha256']
    corpus = json.loads((root/'source_corpus_manifest.json').read_bytes())
    index = Path(corpus['index_path'])
    if file_sha256(index) != corpus['index_sha256']:
        raise ValueError('frozen source-index hash differs')
    query = {r['reaction_key']: r for r in rows(inputs[2])}
    assert len(query) == 18210
    hits = {r['query_reaction_key'] for r in rows(inputs[3])}
    assert len(hits) == 165
    products = {tuple(r['product_components']) for r in query.values()}
    # Actual row products are canonicalized with the same map-free, isotope/
    # charge/stereo-preserving component multiset as the frozen reaction key.
    counts = Counter()
    reaction_keys = defaultdict(set)
    witnesses = defaultdict(list)
    failed = []
    done = 0
    # Pool.imap advances its input iterator in one feeder thread. Only that
    # thread reads this cursor; the creating thread queries the count before
    # the pool starts and closes the connection after every worker has joined.
    connection = sqlite3.connect(index.as_uri()+'?mode=ro&immutable=1', uri=True,
                                 check_same_thread=False)
    expected_rows = connection.execute('SELECT COUNT(*) FROM reactions').fetchone()[0]
    cursor = connection.execute("SELECT source_id,reaction_key,json_extract(record,'$.product') FROM reactions")
    with mp.get_context('spawn').Pool(args.workers, initializer=initialize, initargs=(products,)) as pool:
        for number, (count, matched, invalid) in enumerate(pool.imap_unordered(check_batch, batches(cursor), chunksize=1), 1):
            done += count
            failed.extend(invalid)
            for product, source_id, reaction_key in matched:
                counts[product] += 1
                reaction_keys[product].add(reaction_key)
                witnesses[product].append((source_id, reaction_key))
            if number % 100 == 0:
                print(json.dumps(dict(phase='full_product_identity_scan', rows=done, total_rows=expected_rows,
                                      percent=round(100*done/expected_rows, 2))), flush=True)
    connection.close()
    assert done == expected_rows
    assert not failed, 'invalid source products must be investigated, not silently omitted'
    for path in inputs:
        assert file_sha256(path) == input_hashes[str(path)]
    assert file_sha256(index) == corpus['index_sha256']
    result = []
    for key, record in sorted(query.items()):
        product = tuple(record['product_components'])
        status = ('product_present_reaction_hit' if key in hits else 'product_present_no_reaction_hit') if counts[product] else 'product_absent_under_conservative_identity'
        result.append(dict(reaction_key=key, product_components=list(product), product_lookup_state=status,
            retained_reaction_candidate=key in hits, source_representation_count=counts[product],
            distinct_source_reaction_keys=len(reaction_keys[product]),
            product_match_is_reaction_evidence=False, chemical_validity='not_determined'))
    out.mkdir(parents=True, exist_ok=False)
    payload = ''.join(json.dumps(row, sort_keys=True)+'\n' for row in result).encode()
    (out/'product_recall_queries.jsonl.gz').write_bytes(gzip.compress(payload, mtime=0))
    # Every matching representation is retained, not just a handful of examples.
    with gzip.open(out/'product_matching_source_rows.jsonl.gz', 'wt') as handle:
        for product, values in sorted(witnesses.items()):
            for source_id, reaction_key in sorted(values):
                handle.write(json.dumps(dict(product_components=list(product), source_id=source_id,
                                            source_reaction_key=reaction_key), sort_keys=True)+'\n')
    output = dict(schema='lgrv_frozen_product_identity_recall_audit_v1', complete=True,
        unique_reaction_queries=len(query), unique_query_products=len(products),
        scanned_source_representation_rows=done, invalid_source_products=failed,
        query_state_counts=dict(Counter(r['product_lookup_state'] for r in result)),
        source_files={**input_hashes, str(index): corpus['index_sha256']},
        producer_sha256=file_sha256(__file__),
        output_files={path.name: dict(sha256=file_sha256(path)) for path in sorted(out.iterdir())},
        interpretation='Product identity diagnoses corpus/lookup coverage only. Same product does not verify reactants, conditions or chemistry; no product hit does not exclude analogues or unindexed literature.',
        new_model_inference=False, evidence_grades_changed=False, retrieval_policy_changed=False)
    (out/'product_recall_manifest.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ['complete', 'unique_reaction_queries', 'unique_query_products',
                                            'scanned_source_representation_rows', 'query_state_counts']}), flush=True)


if __name__ == '__main__':
    main()
