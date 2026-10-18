"""A source-side preparation limit is C, never a fabricated failure or support."""
import gzip
import hashlib
import json
from pathlib import Path
import runpy

import pytest


def binder():
    path = Path(__file__).resolve().parents[1] / 'docs/results/lgrv_20261018/source_preparation_limits_20261019/bind_limits.py'
    if not path.is_file():
        pytest.fail('source-preparation limit binder is missing')
    return runpy.run_path(str(path))['bind_limits']


def fixture(tmp_path):
    from test_nmi_literature_grouped_review import fixture as grouped_fixture
    _, root, trace, cache, grouped, hits, _, q, peer = grouped_fixture(tmp_path)
    base = grouped['sources'][0]['preparation']
    evidence_path = root / 'reaction_evidence.jsonl.gz'
    evidence = [dict(reaction_key=r['reaction_key'], tier='C', supported=False) for r in [q, peer]]
    spec = dict(expected_query_count=2, sources=[dict(
        source_id='software-preparation-limit', source_url=base['source_url'],
        document_sha256=base['document_sha256'],
        witnesses=[dict(text=base['checked_passage'], locator='software fixture source passage')],
        indexed_candidate_product=base['source_product'],
        indexed_candidate_precursors=base['source_precursors'],
        expected_candidate_reaction_key=base['expected_exact_reaction_key'],
        source_preparation_status='not_established_for_indexed_precursors',
        source_review='SOFTWARE fixture: this candidate preparation is not established',
        query_reviews=[dict(query_reaction_key=r['reaction_key'],
                            review_limitations='SOFTWARE fixture, no chemical-validity conclusion') for r in [q, peer]]
    )])
    spec_path = tmp_path / 'source_limits.json'

    def seal():
        with gzip.open(evidence_path, 'wt') as stream:
            for r in evidence:
                stream.write(json.dumps(r) + '\n')
        outputs = {p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.glob('*.gz')}
        manifest = root / 'analysis_manifest.json'
        manifest.write_text(json.dumps(dict(complete=True, output_files=outputs)))
        spec['frozen_analysis_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        spec['traceback_sha256'] = hashlib.sha256(trace.read_bytes()).hexdigest()
        spec_path.write_text(json.dumps(spec))

    seal()
    return spec_path, root, trace, cache, spec, evidence, base, seal


def test_source_limits_keep_every_scoped_query_unresolved_without_chemical_failure(tmp_path):
    args = fixture(tmp_path)
    result = binder()(*args[:4])
    assert len(result['sources']) == 1 and len(result['reviews']) == 2
    assert {r['query_reaction_key'] for r in result['reviews']} == {r['reaction_key'] for r in args[5]}
    assert all(r['evidence_tier_retained'] == 'C' and r['generated_reaction_proven_wrong'] is False
               for r in result['reviews'])
    assert result['evidence_grades_changed'] is False


@pytest.mark.parametrize('corruption', ['analysis_bytes', 'raw_source', 'missing_witness',
                                      'wrong_candidate', 'unrelated_primary', 'duplicate_pair',
                                      'previously_supported', 'wrong_population', 'positive_outcome'])
def test_source_limits_reject_unbound_or_misclassified_evidence(tmp_path, corruption):
    args = fixture(tmp_path)
    spec_path, root, trace, cache, spec, evidence, base, seal = args
    source = spec['sources'][0]
    if corruption == 'analysis_bytes':
        with (root / 'literature_hits.jsonl.gz').open('ab') as stream:
            stream.write(b'changed')
    elif corruption == 'raw_source':
        Path(base['raw_document_path']).write_bytes(b'changed source')
    else:
        if corruption == 'missing_witness':
            source['witnesses'][0]['text'] = 'not present in the original primary source'
        elif corruption == 'wrong_candidate':
            source['expected_candidate_reaction_key'] = 'another-candidate'
        elif corruption == 'unrelated_primary':
            source['source_url'] = 'https://example.org/another-primary'
        elif corruption == 'duplicate_pair':
            source['query_reviews'].append(source['query_reviews'][0])
        elif corruption == 'previously_supported':
            evidence[0].update(tier='A', supported=True)
        elif corruption == 'wrong_population':
            spec['expected_query_count'] = 3
        else:
            source['source_preparation_status'] = 'experimentally_validated'
        seal()
    with pytest.raises(ValueError):
        binder()(spec_path, root, trace, cache)
