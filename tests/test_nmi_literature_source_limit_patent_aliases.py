"""A zero-padded US lookup alias is not a different chemical candidate."""
import gzip
import hashlib
import json
from pathlib import Path
import runpy

import pytest


def binder():
    base = Path(__file__).resolve().parents[1] / 'docs/results/lgrv_20261018'
    corrected = base / 'source_preparation_limits_batch2_20261019/bind_limits.py'
    legacy = base / 'source_preparation_limits_20261019/bind_limits.py'
    # Before the versioned fix exists this deliberately exercises the old
    # real reader, reproducing its rejection of a valid padded lookup.
    return runpy.run_path(str(corrected if corrected.is_file() else legacy))['bind_limits']


def fixture(tmp_path):
    from test_nmi_literature_source_limits import fixture as original_fixture
    args = original_fixture(tmp_path)
    spec_path, root, trace, cache, spec, evidence, base, seal = args
    url = 'https://patents.google.com/patent/US6858642B1/en'
    original_receipt = cache / (hashlib.sha256(base['source_url'].encode()).hexdigest() + '.json')
    receipt = json.loads(original_receipt.read_bytes())
    receipt['source_url'] = url
    (cache / (hashlib.sha256(url.encode()).hexdigest() + '.json')).write_text(json.dumps(receipt))
    spec['sources'][0]['source_url'] = url
    hits_path = root / 'literature_hits.jsonl.gz'
    with gzip.open(hits_path, 'rt') as stream:
        hits = list(map(json.loads, stream))
    for hit in hits:
        hit.update(patent_id='US06858642B1',
                   source_url='https://patents.google.com/patent/US06858642B1/en')

    def reseal():
        with gzip.open(hits_path, 'wt') as stream:
            for hit in hits:
                stream.write(json.dumps(hit) + '\n')
        seal()

    reseal()
    return args, hits, reseal


def test_padded_patent_lookup_retains_original_capped_identity_and_unresolved_grade(tmp_path):
    args, hits, _ = fixture(tmp_path)
    result = binder()(*args[:4])
    assert len(result['reviews']) == 2
    for review in result['reviews']:
        assert review['evidence_tier_retained'] == 'C'
        assert review['generated_reaction_proven_wrong'] is False
        assert review['original_candidate_source_ids'] == ['indexed-fixture']
        aliases = review['direct_patent_lookup_bindings']
        assert len(aliases) == 1
        assert aliases[0]['original_patent_id'] == 'US06858642B1'
        assert aliases[0]['original_source_url'] == 'https://patents.google.com/patent/US06858642B1/en'
        assert aliases[0]['lookup_url'] == 'https://patents.google.com/patent/US6858642B1/en'
        assert aliases[0]['publication_kind_inferred'] is False
        assert aliases[0]['patent_family_inferred'] is False
    assert hits[0]['source_url'] == 'https://patents.google.com/patent/US06858642B1/en'


@pytest.mark.parametrize('corruption', ['different_kind', 'different_number', 'mismatched_id_url',
                                      'untrusted_host', 'wrong_candidate', 'source_bytes', 'prior_support'])
def test_patent_alias_does_not_waive_source_chemistry_or_identity_guards(tmp_path, corruption):
    args, hits, reseal = fixture(tmp_path)
    spec_path, root, trace, cache, spec, evidence, base, _ = args
    if corruption == 'different_kind':
        for hit in hits:
            hit.update(patent_id='US06858642B2', source_url='https://patents.google.com/patent/US06858642B2/en')
    elif corruption == 'different_number':
        for hit in hits:
            hit.update(patent_id='US07094801B1', source_url='https://patents.google.com/patent/US07094801B1/en')
    elif corruption == 'mismatched_id_url':
        hits[0]['source_url'] = 'https://patents.google.com/patent/US07094801B1/en'
    elif corruption == 'untrusted_host':
        hits[0]['source_url'] = 'https://example.org/patent/US06858642B1/en'
    elif corruption == 'wrong_candidate':
        hits[0]['product'] = 'CO'
    elif corruption == 'source_bytes':
        Path(base['raw_document_path']).write_bytes(b'changed primary source')
    else:
        evidence[0].update(tier='A', supported=True)
    reseal()
    with pytest.raises(ValueError):
        binder()(spec_path, root, trace, cache)
