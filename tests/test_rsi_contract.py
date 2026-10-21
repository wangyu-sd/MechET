import json
import hashlib
import pytest


def test_adapter_authentication_rejects_incoherent_saved_step(tmp_path):
    from mechet.rsi_contract import authenticate_adapter
    p=tmp_path/'checkpoint-12';p.mkdir()
    (p/'adapter_model.safetensors').write_bytes(b'weights')
    (p/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'Qwen/Qwen3-0.6B','r':16,'target_modules':['q_proj','k_proj','v_proj','o_proj']}))
    (p/'trainer_state.json').write_text(json.dumps({'global_step':11,'max_steps':20}))
    with pytest.raises(ValueError,match='step'):authenticate_adapter(p)
    (p/'trainer_state.json').write_text(json.dumps({'global_step':12,'max_steps':20}))
    assert authenticate_adapter(p)['status']=='provisional_checkpoint'
    with pytest.raises(ValueError,match='SHA'):authenticate_adapter(p,expected_sha='0'*64)


def test_stratification_deterministic_and_never_silently_shrinks():
    from mechet.rsi_contract import select_stratified,stable_seed
    rows=[{'id':f'train_{i}','event_count':i%5,'import_count':i%3} for i in range(20)]
    a=select_stratified(rows,12,17)
    assert [r['id'] for r in a]==[r['id'] for r in select_stratified(list(reversed(rows)),12,17)]
    assert len({r['id'] for r in a})==12
    assert stable_seed('x',1)==stable_seed('x',1)!=stable_seed('x',2)
    with pytest.raises(ValueError):select_stratified(rows,21,17)
