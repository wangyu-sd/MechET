"""Repair bibliographic lookup addresses without changing source identities."""
import gzip
import importlib
import json

import pytest


def implementation():
    try:return importlib.import_module('mechet.analysis.literature_primary_acquisition')
    except ModuleNotFoundError:pytest.fail('provenance-preserving patent reference acquisition missing')


@pytest.mark.parametrize('raw,expected',[
    ('US08501746B2','US8501746B2'),('US03957833','US3957833'),
    ('US20160022662A1','US20160022662A1'),('WO2018104954A1','WO2018104954A1')])
def test_lookup_strips_only_us_numeric_padding_without_inventing_publication_kind(raw,expected):
    u='https://patents.google.com/patent/'+raw+'/en'
    value=implementation().patent_reference(raw,u)
    assert value['original_patent_id']==raw and value['original_source_url']==u
    assert value['lookup_patent_id']==expected
    assert value['lookup_url']=='https://patents.google.com/patent/'+expected+'/en'
    assert value['publication_kind_inferred'] is False
    assert value['patent_family_inferred'] is False


def test_acquisition_merges_equivalent_lookup_urls_but_preserves_all_original_records(tmp_path,monkeypatch):
    import urllib.request
    from mechet.analysis.literature_sources import file_sha256
    raw=tmp_path/'actual_source.html';raw.write_bytes(b'<p>Public source fixture, not experimental evidence.</p>')
    original=urllib.request.urlopen
    monkeypatch.setattr(urllib.request,'urlopen',lambda url,timeout=90:original(raw.as_uri(),timeout=timeout))
    hits=tmp_path/'hits.jsonl.gz'
    rows=[{'source_id':'s'+str(i),'query_reaction_key':'q'+str(i),'patent_id':p,
        'source_url':'https://patents.google.com/patent/'+p+'/en'}
        for i,p in enumerate(('US08501746B2','US8501746B2','US03957833'))]
    with gzip.open(hits,'wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    result=implementation().acquire(hits,tmp_path/'cache',tmp_path/'output',workers=2)
    assert result['complete'] is True and result['unique_lookup_urls']==2
    assert result['cached']==2 and result['unavailable']==0
    assert {r['source_id'] for d in result['records'] for r in d['input_references']}=={'s0','s1','s2'}
    assert all(d['receipt']['raw_document_sha256']==file_sha256(raw) for d in result['records'])
    assert all(d['receipt']['source_passage_checked'] is False for d in result['records'])
    assert result['new_evidence_grades'] is False
    with pytest.raises(FileExistsError):implementation().acquire(hits,tmp_path/'cache',tmp_path/'output')


def test_unavailable_public_lookup_is_recorded_without_manufactured_evidence(tmp_path,monkeypatch):
    import urllib.request,urllib.error
    def fail(url,timeout=90):raise urllib.error.HTTPError(url,404,'Not Found',{},None)
    monkeypatch.setattr(urllib.request,'urlopen',fail)
    path=tmp_path/'hits.gz'
    with gzip.open(path,'wt') as f:f.write(json.dumps({'source_id':'s','query_reaction_key':'q',
        'patent_id':'US03957833','source_url':'https://patents.google.com/patent/US03957833/en'})+'\n')
    result=implementation().acquire(path,tmp_path/'cache',tmp_path/'output')
    assert result['cached']==0 and result['unavailable']==1 and result['complete'] is True
    assert result['records'][0]['status']=='unavailable'
    assert result['source_passages_checked'] is False and result['new_evidence_grades'] is False
