"""Catches cross-track parents, eight-rank assumptions and repeated full-file loads."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_four_gpu_flower_contract_keeps_own_parent_and_full_accepted_train():
    m = load('pr96_earho_contract')
    cfg = dict(algorithm='pr96_earho_v1', track='flower', world_size=4, candidates=2,
               parent_sha256='2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453',
               train_reactions=243753, train_file='/ref/train.jsonl', history_file='/hist/train.jsonl')
    m.validate(cfg)
    with pytest.raises(ValueError, match='PARENT'):
        m.validate({**cfg, 'parent_sha256': m.PARENT_SHA})
    with pytest.raises(ValueError, match='DENOMINATOR'):
        m.validate({**cfg, 'train_reactions': 10137})
    m.validate_hardware(['NVIDIA H20']*4, 'H20', world_size=4)
    with pytest.raises(ValueError, match='HARDWARE'):
        m.validate_hardware(['NVIDIA H20']*8, 'H20', world_size=4)


def test_four_rank_flower_last_checkpoint_uses_its_actual_schedule(tmp_path):
    m = load('pr96_earho_contract')
    cfg = dict(world_size=4, train_reactions=243753, products_per_round=128,
               initial_horizon=2, maximum_horizon=12)
    path = tmp_path/'round-01905'
    path.mkdir()
    for name in ['optimizer.pt', 'adapter_model.safetensors', 'adapter_config.json']+[f'rng.rank{r:02d}.pt' for r in range(4)]:
        (path/name).write_bytes(b'unit')
    state = dict(next_round=1905, horizon=3, optimizer_step=40, visited_train_roots=243753,
                 train_denominator=243753, completed_train_ids_sha256='a'*64)
    (path/'trainer_state.json').write_text(json.dumps(state))
    manifest = dict(contract_sha256='frozen', **state,
                    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()})
    (path/'complete.json').write_text(json.dumps(manifest))
    assert m.verify_checkpoint(path, 'frozen', cfg)['visited_train_roots'] == 243753
    del manifest['files']['rng.rank03.pt']
    (path/'complete.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='COMPONENT'):
        m.verify_checkpoint(path, 'frozen', cfg)


def test_byte_index_returns_only_requested_rows_in_original_history_order(tmp_path):
    m = load('pr96_earho_storage')
    source = tmp_path/'train.jsonl'
    source.write_text('\n'.join(json.dumps(dict(source_id=rid, n=n, text='分子'))
                                 for rid,n in [('b',0),('b',1),('a',0),('c',0)])+'\n')
    index = tmp_path/'offsets.sqlite'
    m.build_index(source, index, grouped=True, expected_reactions=3)
    with m.IndexedJsonl(source, index) as store:
        assert set(store.ids()) == {'a','b','c'}
        assert [r['n'] for r in store.get('b')] == [0,1]
        assert store.get('a')[0]['text'] == '分子'
        with pytest.raises(KeyError):
            store.get('missing')


def test_byte_index_rejects_noncontiguous_history_and_duplicate_sources(tmp_path):
    m = load('pr96_earho_storage')
    source = tmp_path/'train.jsonl'
    source.write_text('\n'.join(json.dumps(dict(source_id=rid)) for rid in ['a','b','a'])+'\n')
    with pytest.raises(ValueError, match='NONCONTIGUOUS|DUPLICATE'):
        m.build_index(source, tmp_path/'bad.sqlite', grouped=True, expected_reactions=2)
    source.write_text('{"source_id":"a"}\n{"source_id":"a"}\n')
    with pytest.raises(ValueError, match='DUPLICATE'):
        m.build_index(source, tmp_path/'duplicate.sqlite', grouped=False, expected_reactions=1)


def test_bucketed_gradient_reduction_preserves_none_grads_and_group_mean():
    import torch
    m = load('pr96_earho_contract')
    model = torch.nn.Linear(2,1,bias=True)
    model.weight.grad = torch.tensor([[6.,9.]])
    m.synchronize_gradients(model, 3, bucket_bytes=4)
    assert model.weight.grad.tolist() == [[2.,3.]]
    assert model.bias.grad.tolist() == [0.]


def test_full_valid_keeps_ambiguous_main_product_in_denominator_without_gold_selection():
    runner = load('train_pr96_earho')
    ready, failed = runner.validation_sources([
        dict(source_id='ok', target_smiles='CCO.[Cl-]', full_precursor_state='CCO'),
        dict(source_id='tie', target_smiles='CC.CO', full_precursor_state='CC')])
    assert [r['source_id'] for r in ready] == ['ok']
    assert ready[0]['target_smiles'] == 'CCO'
    assert ready[0]['full_precursor_state'] == 'CCO'
    assert len(failed) == 1 and failed[0]['reaction_id'] == 'tie'
    assert failed[0]['structural_exact'] is False
    assert failed[0]['error'] == 'MAIN_PRODUCT_AMBIGUOUS'


def test_submission_payload_requests_four_real_cards_with_default_h20_route(tmp_path):
    m = load('prepare_pr96_earho_submission')
    cfg = dict(track='flower', world_size=4, gpu_name='H20', train_reactions=243753,
               validation_reactions=2890, entry_archive_sha256='a'*64)
    payload = m.task_payload(cfg, deployment=tmp_path, archive=tmp_path/'entry.tgz',
                             contract=tmp_path/'contract.json', suffix='test')
    assert payload['host_num'] == 1 and payload['host_gpu_num'] == 4
    assert payload['business_flag'] == 'AI4LSLab_KEMO_ZJK_H20'
    assert payload['location'] == '' and payload['quota_type'] == ''
    assert payload['is_elasticity'] is False
    assert payload['task_flag'].startswith('meteor_mechet_pr96_flower_')
    assert payload['image_full_name'] == 'mirrors.tencent.com/whaleywang/metabo:taiji6'


def _four_rank_gradient_worker(rank, init_file, output):
    import torch
    import torch.distributed as dist
    m = load('pr96_earho_contract')
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=4)
    model = torch.nn.Linear(1,1,bias=True)
    if rank in (1,3):
        model.weight.grad = torch.tensor([[3. if rank == 1 else 6.]])
    m.synchronize_gradients(model,3,bucket_bytes=8)
    torch.save(dict(weight=model.weight.grad,bias=model.bias.grad),Path(output)/f'{rank}.pt')
    dist.destroy_process_group()


def test_four_rank_buckets_preserve_unequal_work_and_final_partial_global_batch(tmp_path):
    import torch
    import torch.multiprocessing as mp
    mp.spawn(_four_rank_gradient_worker,args=(str(tmp_path/'gloo'),str(tmp_path)),nprocs=4,join=True)
    for rank in range(4):
        result = torch.load(tmp_path/f'{rank}.pt',weights_only=True)
        assert result['weight'].item() == 3.
        assert result['bias'].item() == 0.


def test_byte_index_rejects_changed_source_instead_of_reusing_wrong_offsets(tmp_path):
    m = load('pr96_earho_storage')
    source,index = tmp_path/'train.jsonl',tmp_path/'index.sqlite'
    source.write_text('{"source_id":"a"}\n')
    m.build_index(source,index,grouped=False,expected_reactions=1)
    source.write_text('{"source_id":"other"}\n')
    with pytest.raises(ValueError,match='SOURCE_CHANGED'):
        m.IndexedJsonl(source,index)


def test_batched_response_logprobs_and_gradients_match_individual_causal_scoring():
    import torch
    from types import SimpleNamespace
    m = load('pr96_earho_contract')
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=torch.nn.Embedding(17,5)
            self.head=torch.nn.Linear(5,17)
        def forward(self,input_ids,attention_mask,logits_to_keep,use_cache,position_ids=None):
            # Causal prefix sum makes incorrect pad masks affect real logits.
            hidden=(self.embedding(input_ids)*attention_mask[...,None]).cumsum(1)
            return SimpleNamespace(logits=self.head(hidden[:,logits_to_keep]))
    torch.manual_seed(17)
    model=Tiny()
    sequences=[([1,2],[3,4,5]),([6],[7]),([8,9,10],[11,12])]
    separate=[m.token_logps(model,p,r) for p,r in sequences]
    sum(v.mean() for v in separate).backward()
    expected_grads={n:p.grad.clone() for n,p in model.named_parameters()}
    expected=[v.detach() for v in separate]
    model.zero_grad()
    batched=m.token_logps_batch(model,sequences,pad_token_id=0)
    for a,b in zip(batched,expected):
        assert torch.allclose(a,b,atol=1e-6)
    sum(v.mean() for v in batched).backward()
    for n,p in model.named_parameters():
        assert torch.allclose(p.grad,expected_grads[n],atol=1e-6)


def test_qwen35_linear_and_full_attention_batch_padding_matches_unpadded_scoring(monkeypatch):
    import torch
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
    # Select all real upstream CPU implementations before constructing modules;
    # merely having FLA installed otherwise creates a CUDA-only fused norm.
    for name in ('FusedRMSNormGated','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule',
                 'causal_conv1d_fn','causal_conv1d_update'):
        monkeypatch.setattr(upstream,name,None)
    m=load('pr96_earho_contract')
    torch.manual_seed(19)
    config=Qwen3_5TextConfig(vocab_size=19,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        layer_types=['linear_attention','full_attention'],max_position_embeddings=128,
        rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=1.,mrope_section=[1,1,2]))
    model=Qwen3_5ForCausalLM(config).eval()
    sequences=[([1,2],[3,4]),([5,6,7,8],[9]),([10],[11,12,13])]
    separate=[m.token_logps(model,p,r) for p,r in sequences]
    sum(v.mean() for v in separate).backward()
    expected_grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    expected=[v.detach() for v in separate]
    model.zero_grad()
    batched=m.token_logps_batch(model,sequences)
    for a,b in zip(batched,expected):
        assert torch.allclose(a,b,atol=2e-6)
    sum(v.mean() for v in batched).backward()
    for n,p in model.named_parameters():
        if n in expected_grads:
            assert torch.allclose(p.grad,expected_grads[n],atol=2e-5),n
