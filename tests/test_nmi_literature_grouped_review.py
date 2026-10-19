"""Grouped source review must retain query-specific applicability and provenance."""
import gzip
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from mechet.analysis.literature_reaction_keys import make_reaction


def recipe():
    path = Path(__file__).resolve().parents[1] / 'docs/results/lgrv_20261018/grouped_primary_review_20261019/compile_reviews.py'
    if not path.is_file():
        pytest.fail('grouped primary review compiler is missing')
    return runpy.run_path(str(path))['compile_reviews']


def fixture(tmp_path):
    from test_nmi_literature_protocol import primary
    from test_nmi_literature_stereo_family import source, remote_query
    from mechet.analysis.literature_stereo_family import family_context
    s = source()
    q = remote_query(s)
    peer = make_reaction(q['product_raw'].replace('[F:10000]', '[Cl:10000]'),
                         [q['precursors_raw'][0].replace('[F:10000]', '[Cl:10000]')])
    source_key = make_reaction(s['mapped_product'], [s['mapped_precursors']])['reaction_key']
    base = primary(tmp_path, source_id='software-fixture', source_product=s['mapped_product'],
                   source_precursors=[s['mapped_precursors']], expected_exact_reaction_key=source_key,
                   structural_reconciliation='SOFTWARE fixture, not experimental chemical evidence')
    root = tmp_path / 'analysis'
    root.mkdir()
    def write_rows(name, rows):
        with gzip.open(root / name, 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
    write_rows('unique_reaction_queries_enriched.jsonl.gz', [q, peer])
    write_rows('retrieval_query_contexts.jsonl.gz',
               [dict(family_context(r['precursors_raw'][0], r['product_raw']), reaction_key=r['reaction_key'])
                for r in [q, peer]])
    hits = [dict(query_reaction_key=r['reaction_key'], source_id='indexed-fixture',
                 source_url=base['source_url'], reaction_key=source_key, product=s['mapped_product'],
                 precursors=[s['mapped_precursors']], match_type='local_edit_analogue_candidate') for r in [q, peer]]
    write_rows('literature_hits.jsonl.gz', hits)
    trace = tmp_path / 'trace.jsonl.gz'
    with gzip.open(trace, 'wt'):
        pass
    reviews = [dict(query_reaction_key=r['reaction_key'], expected_tier=tier,
                    analogue_compatibility_checked=tier == 'B',
                    substrate_differences='hypothetical remote substituent; SOFTWARE fixture only',
                    functional_group_compatibility='fixture-scoped compatibility, not empirical evidence',
                    stereo_compatibility_checked=True, review_limitations='not a chemical result')
               for r, tier in [(q, 'B'), (peer, 'C')]]
    spec = dict(expected_query_count=2, sources=[dict(preparation=base, query_reviews=reviews)])
    spec_path = tmp_path / 'spec.json'
    def seal():
        outputs = {p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.glob('*.gz')}
        manifest = root / 'analysis_manifest.json'
        manifest.write_text(json.dumps(dict(complete=True, output_files=outputs)))
        spec['frozen_analysis_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        spec['traceback_sha256'] = hashlib.sha256(trace.read_bytes()).hexdigest()
        spec_path.write_text(json.dumps(spec))
    seal()
    return spec_path, root, trace, tmp_path / 'primary_cache', spec, hits, seal, q, peer


def test_grouped_review_exports_only_individually_supported_queries(tmp_path):
    from mechet.analysis.literature_evidence import assess
    args = fixture(tmp_path)
    audit, registry = recipe()(*args[:4])
    assert [r['assessed_tier'] for r in audit['reviews']] == ['B', 'C']
    assert len(audit['sources']) == 1
    assert len(registry['records']) == 1
    assert registry['records'][0]['analogue_query_reaction_key'] == args[7]['reaction_key']
    assert assess(args[8], [], registry['records'])['tier'] == 'C'


@pytest.mark.parametrize('corruption', ['analysis_bytes', 'raw_source', 'candidate_identity',
                                      'unrelated_primary', 'duplicate_pair', 'unsupported_B', 'wrong_count'])
def test_grouped_review_cannot_publish_corrupt_or_unreviewed_support(tmp_path, corruption):
    args = fixture(tmp_path)
    spec_path, root, trace, cache, spec, hits, seal, _, _ = args
    if corruption == 'analysis_bytes':
        with (root / 'literature_hits.jsonl.gz').open('ab') as stream:
            stream.write(b'corrupt')
    elif corruption == 'raw_source':
        Path(spec['sources'][0]['preparation']['raw_document_path']).write_bytes(b'changed source')
    elif corruption in {'candidate_identity', 'unrelated_primary'}:
        for hit in hits:
            hit['reaction_key' if corruption == 'candidate_identity' else 'source_url'] = 'unrelated'
        with gzip.open(root / 'literature_hits.jsonl.gz', 'wt') as stream:
            for hit in hits:
                stream.write(json.dumps(hit) + '\n')
        seal()
    else:
        group = spec['sources'][0]
        if corruption == 'duplicate_pair':
            group['query_reviews'].append(group['query_reviews'][0])
        elif corruption == 'unsupported_B':
            group['query_reviews'][0]['analogue_compatibility_checked'] = False
        else:
            spec['expected_query_count'] = 3
        seal()
    with pytest.raises(ValueError):
        recipe()(spec_path, root, trace, cache)
