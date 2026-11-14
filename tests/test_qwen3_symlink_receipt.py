import hashlib
import json

import pytest


def fixture_bindings(tmp_path, monkeypatch):
    from mechet import qwen3_continuation as m
    def write(name, value):
        p=tmp_path/name; p.write_text(value); return p, hashlib.sha256(p.read_bytes()).hexdigest()
    train, ts=write('train-real.jsonl','train payload')
    link=tmp_path/'train-alias.jsonl'; link.symlink_to(train)
    valid, vs=write('valid.jsonl','valid payload')
    parent=tmp_path/'parent';parent.mkdir()
    weight, ws=write('parent/adapter_model.safetensors','weight payload')
    pc, ps=write('parent/adapter_config.json',json.dumps({'base_model_name_or_path':'Qwen/Qwen3-8B','r':16}))
    ref, rs=write('reference.json','{}'); hist, hs=write('history.json','{}')
    app, aps=write('approval.json',json.dumps(dict(reference_training_authorized=True,
        reference_manifest_sha256=rs,history_manifest_sha256=hs)))
    # Business population validation has its own strict tests. These fixtures
    # exercise the real shared-filesystem hash/receipt logic without full data.
    monkeypatch.setattr(m,'validate_config',lambda cfg:None)
    for k,v in dict(PARENT=parent,TRAIN_SHA=ts,VALID_SHA=vs,PARENT_SHA=ws,PARENT_CONFIG_SHA=ps).items():
        monkeypatch.setattr(m,k,v)
    b={}
    for name,p,digest in [('reference_approval',app,aps),('reference_manifest',ref,rs),('history_manifest',hist,hs)]:
        b[name]=str(p);b[name+'_sha256']=digest
    cfg=dict(train_file=str(link),validation_file=str(valid),output_dir=str(tmp_path/'out'),
             qwen3_continuation=b,condition_name='unit')
    original=m.verify_files;calls=[]
    def counted(bindings):
        calls.append(1);return original(bindings)
    monkeypatch.setattr(m,'verify_files',counted)
    return m,cfg,train,link,calls


def test_verified_symlink_receipt_reuses_hashes(tmp_path,monkeypatch):
    m,cfg,train,link,calls=fixture_bindings(tmp_path,monkeypatch)
    first=m.verify_bindings(cfg)
    assert str(train) in first['files'] and str(link) not in first['files']
    assert m.verify_bindings(cfg)==first
    assert len(calls)==1,'unchanged symlink inputs must not trigger repeated full scans'


def test_symlink_retarget_cannot_reuse_previous_verification(tmp_path,monkeypatch):
    m,cfg,train,link,calls=fixture_bindings(tmp_path,monkeypatch)
    m.verify_bindings(cfg)
    other=tmp_path/'changed.jsonl';other.write_text('changed payload')
    link.unlink();link.symlink_to(other)
    with pytest.raises(ValueError,match='BINDING'):
        m.verify_bindings(cfg)


def test_receipt_digest_mismatch_is_reverified(tmp_path,monkeypatch):
    m,cfg,train,link,calls=fixture_bindings(tmp_path,monkeypatch)
    value=m.verify_bindings(cfg);value['files'][str(train)]['sha256']='0'*64
    p=tmp_path/'out/runtime'/('input-verification-'+value['key']+'.json');p.write_text(json.dumps(value))
    fixed=m.verify_bindings(cfg)
    assert fixed['files'][str(train)]['sha256']==hashlib.sha256(train.read_bytes()).hexdigest()
    assert len(calls)==2
