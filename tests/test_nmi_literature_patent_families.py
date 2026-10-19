"""Source-authenticated patent families must consume one retrieval slot.

Hand-checked family41216960 comes from the actual Google Patents Family
sections of US8609705B2, US8445522B2 and US8324387B2. These fixtures test
bibliographic identity only, never experimental chemical support.
"""
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from mechet.analysis.literature_index import ReactionIndex
from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_sources import snapshot_primary_document


def implementation():
    try:
        return importlib.import_module('mechet.analysis.literature_patent_families')
    except ModuleNotFoundError:
        pytest.fail('authenticated patent-family overlay missing')


def registry(tmp_path, entries):
    receipts = []
    for number, (patent, family) in enumerate(entries):
        raw = tmp_path / f'page{number}.html'
        raw.write_text('<html><section itemprop="family" itemscope>'
                       f'<h1>Family</h1><h2>ID={family}</h2></section>'
                       '<section itemprop="familyToFamilyCitations"><h2>ID=999</h2></section></html>')
        cache = tmp_path / f'cache{number}'
        receipt = snapshot_primary_document(raw.as_uri(), cache)
        # Local source fixture retains real receipt checks while using the
        # public URL identity the production parser is required to bind.
        receipt['source_url'] = f'https://patents.google.com/patent/{patent}/en'
        path = cache / (hashlib.sha256(receipt['source_url'].encode()).hexdigest() + '.json')
        path.write_text(json.dumps(receipt))
        receipts.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    spec = tmp_path / 'family_sources.json'
    spec.write_text(json.dumps({'schema': 'lgrv_patent_family_sources_v1', 'receipts': receipts}))
    return spec


@pytest.mark.parametrize('pathway', ['legacy', 'full_stereo'])
def test_actual_repeated_family_does_not_consume_independent_cap(tmp_path, pathway):
    m = implementation()
    spec = registry(tmp_path, [('US8609705B2', '41216960'), ('US8445522B2', '41216960'),
                               ('US8324387B2', '41216960'), ('US7834039B2', '99991')])
    resolver = m.load_family_registry(spec)
    index = ReactionIndex(tmp_path / 'sources.sqlite', family_resolver=resolver)
    for name, patent in [('a', 'US08609705B2'), ('b', 'US08445522B2'),
                         ('c', 'US08324387B2'), ('d', 'US07834039B2')]:
        index.add({'source_id': name, 'patent_id': patent, 'patent_family': None,
                   'product': 'CO', 'precursors': ['CBr', '[OH-]'], 'license': 'CC-BY-SA-4.0'})
    index.commit()
    reaction = make_reaction('CO', ['CBr', '[OH-]'])
    if pathway == 'legacy':
        hits = index.query(reaction, cap=2)
    else:
        from mechet.analysis.literature_stereo_family import family_context, query_sources
        hits, _ = query_sources(index, reaction, family_context('CBr.[OH-]', 'CO'), cap=2)
    assert [h['source_id'] for h in hits] == ['a', 'd']
    assert hits[0]['source_family_key'] == 'GOOGLE_PATENTS_FAMILY:41216960'
    assert hits[0]['source_representation_ids'] == ['a', 'b', 'c']
    assert hits[0]['patent_id'] == 'US08609705B2'
    index.close()


def test_family_parser_uses_own_family_not_cited_families(tmp_path):
    m = implementation()
    resolver = m.load_family_registry(registry(tmp_path, [('US8324387B2', '41216960')]))
    assert resolver.resolve({'patent_id': 'US08324387B2'}) == 'GOOGLE_PATENTS_FAMILY:41216960'
    assert resolver.resolve({'patent_id': 'US8324387B1'}) == 'US8324387B1'
    assert resolver.resolve({'doi': '10.1234/example'}) == '10.1234/example'
    assert resolver.manifest['family_count'] == 1
    assert resolver.manifest['unknown_family_policy'] == 'publication_fallback_explicitly_unresolved'


@pytest.mark.parametrize('mutation', ['receipt', 'raw', 'text', 'source_url'])
def test_family_overlay_rejects_tampered_receipt_or_document(tmp_path, mutation):
    m = implementation()
    spec = registry(tmp_path, [('US8324387B2', '41216960')])
    entry = json.loads(spec.read_bytes())['receipts'][0]
    receipt_path = Path(entry['path'])
    receipt = json.loads(receipt_path.read_bytes())
    if mutation == 'receipt':
        receipt_path.write_text('{}')
    elif mutation in {'raw', 'text'}:
        Path(receipt['raw_document_path' if mutation == 'raw' else 'document_path']).write_text('tampered')
    else:
        receipt['source_url'] = 'https://patents.google.com/patent/US9999999B2/en'
        receipt_path.write_text(json.dumps(receipt))
        value = json.loads(spec.read_bytes())
        value['receipts'][0]['sha256'] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        spec.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        m.load_family_registry(spec)


def test_conflicting_original_family_metadata_is_not_silently_chosen(tmp_path):
    m = implementation()
    spec = registry(tmp_path, [('US8324387B2', '41216960'), ('US8324387B2', '99991')])
    with pytest.raises(ValueError, match='conflict'):
        m.load_family_registry(spec)


def test_missing_family_section_remains_unknown_without_guessing(tmp_path):
    m = implementation()
    spec = registry(tmp_path, [('US8324387B2', '41216960')])
    entry = json.loads(spec.read_bytes())['receipts'][0]
    p = Path(entry['path']); receipt = json.loads(p.read_bytes())
    raw = b'<html><section itemprop="familyToFamilyCitations"><h2>ID=999</h2></section></html>'
    from mechet.analysis.literature_sources import normalize_primary_text
    Path(receipt['raw_document_path']).write_bytes(raw)
    Path(receipt['document_path']).write_text(normalize_primary_text(raw))
    receipt['raw_document_sha256'] = hashlib.sha256(raw).hexdigest()
    receipt['document_sha256'] = hashlib.sha256(normalize_primary_text(raw).encode()).hexdigest()
    p.write_text(json.dumps(receipt))
    value = json.loads(spec.read_bytes()); value['receipts'][0]['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    spec.write_text(json.dumps(value))
    resolver = m.load_family_registry(spec)
    assert resolver.resolve({'patent_id': 'US8324387B2'}) == 'US8324387B2'
    assert resolver.manifest['unresolved_documents'] == 1
