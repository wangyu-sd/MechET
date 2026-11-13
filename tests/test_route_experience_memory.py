import pytest
import json
import hashlib


def scope():
    return {'state':'[CH3:1][OH:2]','target':'CO','route_root':'CO','next_map':3,
        'imported':{},'visited':['CO'],'accepted_action_types':[], 'ancestors':[],
        'stock_version':'s1','executor_version':'e1','occurrence':'0','event_history_sha256':'b'*64}


def evidence(tmp_path, uid, success=True, *, entry='a', content=None, current=None):
    p=tmp_path/(uid+'.json')
    d=dict(receipt_id=uid,split='training',episode_complete=True,resource_censored=False,
           success=success,signal='executor_recovery',verification='executor_replayed',memory_id=entry,
           scope_key=hashlib.sha256(json.dumps(current or scope(),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
           content_sha256=hashlib.sha256(json.dumps(content or {},sort_keys=True,separators=(',',':')).encode()).hexdigest())
    p.write_text(json.dumps(d))
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def test_incomplete_scope_cannot_be_used_as_a_global_negative_cache():
    from mechet.route_experience_memory import ExperienceMemory
    with pytest.raises(ValueError,match='SCOPE'):
        ExperienceMemory().add('bad', {'state': 'CO'}, {}, evidence_ids=['r'], split='training')


def test_bare_receipt_ids_are_not_completed_training_evidence():
    from mechet.route_experience_memory import ExperienceMemory
    m=ExperienceMemory()
    # A string and a caller-declared split cannot certify an episode/verifier.
    with pytest.raises(ValueError,match='EVIDENCE'):
        m.add('bad', scope(), {},
            evidence_ids=['r'], split='training')


def test_memory_requires_exact_applicability_and_feedback_changes_selection(tmp_path):
    from mechet.route_experience_memory import ExperienceMemory
    m = ExperienceMemory()
    current = scope()
    m.add('a', current, {'strategy': 'first'}, evidence=[evidence(tmp_path,'receipt-a',content={'strategy':'first'})])
    m.add('b', current, {'strategy': 'second'}, evidence=[evidence(tmp_path,'receipt-b',entry='b',content={'strategy':'second'})])
    m.feedback('a', success=False, evidence=evidence(tmp_path,'failed-a',False,content={'strategy':'first'}))
    m.feedback('b', success=True, evidence=evidence(tmp_path,'passed-b',entry='b',content={'strategy':'second'}))
    assert [x['id'] for x in m.retrieve(current, limit=1)] == ['b']
    assert m.retrieve(dict(current, ancestors=['different']), limit=5) == []
    assert m.retrieve(dict(current, visited=['CO', 'C.O']), limit=5) == []
    assert m.retrieve(dict(current, executor_version='v2'), limit=5) == []


def test_memory_rejects_heldout_or_evidenceless_learning_and_ignores_resources(tmp_path):
    from mechet.route_experience_memory import ExperienceMemory
    m = ExperienceMemory()
    with pytest.raises(ValueError, match='EVIDENCE'):
        m.add('a', scope(), {}, evidence_ids=[], split='training')
    with pytest.raises(ValueError, match='TRAIN'):
        m.add('a', {'state': 'x'}, {}, evidence_ids=['r'], split='validation')
    m.add('a', scope(), {}, evidence=[evidence(tmp_path,'r')])
    before = m.snapshot()
    assert m.feedback('a', success=False, evidence_id='oom', split='training', resource_censored=True) is False
    assert m.snapshot() == before
    with pytest.raises(ValueError, match='TRAIN'):
        m.feedback('a', success=True, evidence_id='eval', split='test')


def test_memory_roundtrip_keeps_counts_and_deduplicates_feedback_receipts(tmp_path):
    from mechet.route_experience_memory import ExperienceMemory
    m = ExperienceMemory(); current=scope()
    m.add('a', current, {'lesson': 'recover'}, evidence=[evidence(tmp_path,'source',content={'lesson':'recover'})])
    proof=evidence(tmp_path,'r2',content={'lesson':'recover'})
    m.feedback('a', success=True, evidence=proof)
    m.feedback('a', success=True, evidence=proof)
    restored = ExperienceMemory(m.snapshot())
    assert restored.snapshot() == m.snapshot()
    item = restored.retrieve(current, limit=1)[0]
    assert item['successes'] == 1 and item['utility'] == 1.0
    item['content']['lesson'] = 'external mutation'
    assert restored.retrieve(current, limit=1)[0]['content']['lesson'] == 'recover'


def test_evidence_tampering_and_incomplete_episodes_are_rejected(tmp_path):
    from mechet.route_experience_memory import ExperienceMemory
    m=ExperienceMemory();proof=evidence(tmp_path,'bad')
    from pathlib import Path
    p=Path(proof['path']);d=json.loads(p.read_text());d['episode_complete']=False;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='HASH'):
        m.add('a',scope(),{},evidence=[proof])
    proof['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='INELIGIBLE'):
        m.add('a',scope(),{},evidence=[proof])


def test_receipt_cannot_support_an_unrelated_memory_candidate(tmp_path):
    from mechet.route_experience_memory import ExperienceMemory
    m=ExperienceMemory();proof=evidence(tmp_path,'belongs-to-a')
    with pytest.raises(ValueError,match='BINDING'):
        m.add('unrelated',scope(),{'unverified':'different advice'},evidence=[proof])


def test_scope_includes_rejected_event_history(tmp_path):
    from mechet.route_experience_memory import ExperienceMemory
    m=ExperienceMemory();current=scope()
    m.add('a',current,{},evidence=[evidence(tmp_path,'source')])
    assert m.retrieve(dict(current,event_history_sha256='f'*64))==[]


def test_scope_without_history_digest_is_incomplete(tmp_path):
    from mechet.route_experience_memory import ExperienceMemory
    current=scope();current.pop('event_history_sha256',None)
    with pytest.raises(ValueError,match='SCOPE'):
        ExperienceMemory().add('a',current,{},evidence=[evidence(tmp_path,'source')])
