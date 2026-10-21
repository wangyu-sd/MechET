import copy
import json
from pathlib import Path
import pytest


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    root=Path('/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17')
    if not root.exists(): pytest.skip('saved real tokenizer unavailable')
    return AutoTokenizer.from_pretrained(root,local_files_only=True)


def rows():
    return json.loads((Path(__file__).parent/'fixtures/pr92_real_history_train1.json').read_text())


def test_real_train_replay_exact_prefix_full_call_eos_and_balanced_selection(tokenizer):
    from mechet.rsi_replay import verify_reaction,select_actions
    from mechet.assistant_masking import render_chat,render_qwen_sft_tool_prefix
    rs=rows();source=rs[0]['source_id']
    verified=verify_reaction(rs,tokenizer,source,{source})
    assert len(verified)==len(rs)
    for row,got in zip(rs,verified):
        prefix=render_qwen_sft_tool_prefix(tokenizer,row['messages'][:2],tools=row['tools'])
        full=render_chat(tokenizer,row['messages'][:3],tools=row['tools'])
        assert got['prompt_ids']==tokenizer.encode(prefix,add_special_tokens=False)
        assert full.startswith(prefix)
        action=tokenizer.decode(got['completion_ids'],skip_special_tokens=False)
        assert action.endswith('<|im_end|>') and full[len(prefix):].startswith(action)
        assert '<|im_start|>tool' not in action
        assert got['source_id']==source and got['executor_verified']
    selected=select_actions(verified,4)
    assert len(selected)<=4 and {x['tool'] for x in selected}=={'import_fragments','apply_electron_flow','finish_trace'}


@pytest.mark.parametrize('corruption',['source','history','result','decision','future'])
def test_replay_rejects_wrong_source_history_executor_or_future(tokenizer,corruption):
    from mechet.rsi_replay import verify_reaction
    rs=copy.deepcopy(rows());source=rs[0]['source_id']
    if corruption=='source':rs[0]['source_id']='flower_mech_proof_valid_1'
    if corruption=='history':rs[0]['messages'][1]['content']+=' altered'
    if corruption=='result':rs[0]['messages'][3]['content']='{"ok":true,"current_state":"C"}'
    if corruption=='decision':rs[0]['metadata']['decision_index']=3
    if corruption=='future':rs[0]['metadata']['history_model_visible_gold_horizon']=True
    with pytest.raises(ValueError):verify_reaction(rs,tokenizer,source,{source})


def test_validation_source_cannot_enter_replay(tokenizer):
    from mechet.rsi_replay import verify_reaction
    rs=rows();source=rs[0]['source_id']
    with pytest.raises(ValueError,match='TRAIN'):verify_reaction(rs,tokenizer,source,set())


def test_authentic_mapped_source_target_preserves_hidden_kekule_graph_without_prompt_change(tokenizer):
    from mechet.rsi_replay import verify_reaction
    rs=json.loads((Path(__file__).parent/'fixtures/pr92_real_history_train110212.json').read_text())
    source=rs[0]['source_id']
    target='[CH3:1][n:2]1[c:3]2[c:4]([cH:5][c:6]([NH:7][c:30]3[n:29][c:28]([Cl:27])[n:34][c:33]4[c:32]3[cH:47][cH:46][n:35]4[S:36]([c:37]3[cH:38][cH:39][c:40]([CH3:41])[cH:42][cH:43]3)(=[O:44])=[O:45])[cH:8][cH:9]2)[cH:10][n:11]1'
    with pytest.raises(ValueError,match='stored tool result differs'):
        verify_reaction(rs,tokenizer,source,{source})
    got=verify_reaction(rs,tokenizer,source,{source},mapped_target=target)
    assert len(got)==len(rs) and all(r['executor_verified'] for r in got)


def test_mapped_parent_reader_uses_exact_source_id_and_rejects_changed_sha(tmp_path):
    from mechet.rsi_replay import mapped_replay_targets
    from mechet.rsi_contract import sha256
    p=tmp_path/'train.jsonl';source='flower_mech_proof_train_1'
    p.write_text(json.dumps({'source_id':source,'target_smiles':'[CH3:1][OH:2]'})+'\n')
    got=mapped_replay_targets(p,[source],sha256(p),tmp_path/'cache.json')
    assert got=={source:'[CH3:1][OH:2]'}
    with pytest.raises(ValueError,match='cache changed'):
        mapped_replay_targets(p,[source],'wrong',tmp_path/'cache.json')
    with pytest.raises(ValueError,match='SHA/coverage'):
        mapped_replay_targets(p,['flower_mech_proof_valid_1'],sha256(p),tmp_path/'other.json')


def test_cached_replay_rejects_changed_private_parent_mapping(tmp_path):
    from mechet.rsi_replay import build_replay_pool
    from mechet.rsi_contract import fingerprint,write_json
    source='flower_mech_proof_train_1';path=tmp_path/'source.jsonl'
    rows=[{'id':'real','source_id':source}]
    write_json(tmp_path/'actions.json',rows)
    write_json(tmp_path/'manifest.json',{'contract':fingerprint([str(path),[source],'sha','real-history-execute-v1']),
        'actions_sha':fingerprint(rows),'mapped_source_target_sha':'different-original-parent'})
    with pytest.raises(ValueError,match='mapped'):
        build_replay_pool(path,[source],None,'sha',tmp_path,{source:{'replay_mapped_target':'[CH4:1]'}})
