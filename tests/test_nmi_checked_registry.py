"""Checked preparation binding must not fabricate a witness or change chemistry."""
import hashlib
import json

import pytest


def source_fixture(tmp_path):
    from mechet.analysis.literature_sources import snapshot_primary_document
    from mechet.analysis.literature_reaction_keys import make_reaction
    source = tmp_path/'source.html'
    source.write_text('<p>Example 1. Methanol was prepared from methyl bromide and hydroxide.</p>')
    cache = tmp_path/'cache'
    receipt = snapshot_primary_document(source.as_uri(), cache)
    spec = {'source_id': 'software-source:example1', 'source_url': source.as_uri(),
        'document_sha256': receipt['document_sha256'],
        'checked_passage': 'Methanol was prepared from methyl bromide and hydroxide.',
        'locator': 'Example 1', 'source_product': 'CO', 'source_precursors': ['CBr', '[OH-]'],
        'structural_reconciliation': 'Named methanol and substrates transcribed independently in this software fixture.',
        'performed_example': True, 'passage_checked': True, 'checker': 'software_fixture_not_model_result',
        'expected_exact_reaction_key': make_reaction('CO', ['CBr', '[OH-]'])['reaction_key']}
    return cache, receipt, spec


def test_bind_preparation_keeps_real_cache_bytes_short_quote_and_named_structures(tmp_path):
    from mechet.analysis.literature_checked_registry import bind_preparation
    from mechet.analysis.literature_evidence import assess
    from mechet.analysis.literature_reaction_keys import make_reaction
    cache, receipt, spec = source_fixture(tmp_path)
    record = bind_preparation(spec, cache)
    assert record['raw_document_sha256'] == receipt['raw_document_sha256']
    assert record['document_sha256'] == receipt['document_sha256']
    assert record['checked_passage_sha256'] == hashlib.sha256(spec['checked_passage'].encode()).hexdigest()
    assert record['normalized_passage_offset'] == 11
    assert record['source_human_reviewed'] is False and record['physical_mechanism_verified'] is False
    assert assess(make_reaction('CO', ['CBr', '[OH-]']), [], [record])['tier'] == 'A'


@pytest.mark.parametrize('change', ['wrong_text_hash', 'absent_quote', 'unreviewed', 'wrong_structure', 'raw_drift'])
def test_bind_preparation_rejects_unreviewed_or_changed_source_without_creating_new_evidence(tmp_path, change):
    from mechet.analysis.literature_checked_registry import bind_preparation
    cache, receipt, spec = source_fixture(tmp_path)
    if change == 'wrong_text_hash': spec['document_sha256'] = '0'*64
    if change == 'absent_quote': spec['checked_passage'] = 'This statement is not in the source.'
    if change == 'unreviewed': spec['performed_example'] = False
    if change == 'wrong_structure': spec['source_product'] = 'CCO'
    if change == 'raw_drift':
        from pathlib import Path
        Path(receipt['raw_document_path']).write_text('changed source')
    with pytest.raises(ValueError): bind_preparation(spec, cache)


def test_registry_cli_refuses_overwrite_and_preserves_review_spec_provenance(tmp_path):
    from mechet.analysis.literature_checked_registry import build_registry
    cache, _, spec = source_fixture(tmp_path)
    path = tmp_path/'spec.json'; path.write_text(json.dumps({'records': [spec]}))
    output = tmp_path/'registry.json'
    result = build_registry(path, cache, output)
    assert result['review_spec_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['records'][0]['source_id'] == spec['source_id']
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError): build_registry(path, cache, output)
