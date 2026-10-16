import json
from types import SimpleNamespace

import pytest
import torch


def test_joint_checkpoint_snapshots_both_models_not_live_tensor_aliases(tmp_path):
    from scripts.pr83_resume import save_joint_training_state
    qwen = torch.tensor([1.])
    graph = torch.tensor([2.])
    path = tmp_path/'latest.pt'
    save_joint_training_state(path, {'qwen_policy':{'a':qwen}, 'graph_policy':{'b':graph},
                                    'optimizer':{}, 'manifest':{'algorithm':'joint'}})
    qwen.add_(9)
    graph.add_(9)
    loaded = torch.load(path, weights_only=False)
    assert loaded['qwen_policy']['a'].item() == 1
    assert loaded['graph_policy']['b'].item() == 2


def test_paired_export_reloads_real_peft_adapter_and_rejects_corruption(tmp_path):
    pytest.importorskip('peft')
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from scripts.pr83_resume import export_joint_policy, load_joint_policy_contract
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    model = get_peft_model(Qwen3ForCausalLM(config),
        LoraConfig(r=2, target_modules=['q_proj'], task_type='CAUSAL_LM'), adapter_name='policy')
    parent = tmp_path/'parent'
    parent.mkdir()
    (parent/'adapter_manifest.json').write_text(json.dumps({
        'environment_revision':'natural_language_electron_event_v2',
        'executor_revision':'MECH_PROOF_v1_full_coverage_v4'}))
    runtime = SimpleNamespace(model=model, tokenizer=None, graph=SimpleNamespace(
        sha256='base', model=SimpleNamespace(policy=torch.nn.Linear(2,2))),
        args=SimpleNamespace(policy_adapter=parent))
    manifest = {'algorithm':'joint_import_flow_endpoint_v1', 'settings':{
        'dtype':'bfloat16', 'no_4bit':True}, 'qwen_adapter_sha256':'parent',
        'crn_pretrained_sha256':'base', 'train_source_sha256':'train'}
    result = export_joint_policy(runtime, tmp_path/'output', manifest, {'optimizer_updates':1})
    root = tmp_path/'output'/result['relative_directory']
    checked = load_joint_policy_contract(root)
    adapter = root/checked['qwen_adapter_relative_path']
    restored = PeftModel.from_pretrained(Qwen3ForCausalLM(config), adapter, adapter_name='policy')
    original = {k:v for k,v in model.state_dict().items() if 'lora_' in k}
    restored_weights = {k:v for k,v in restored.state_dict().items() if 'lora_' in k}
    for key, value in original.items():
        torch.testing.assert_close(value, restored_weights[key])
    with (adapter/'adapter_model.safetensors').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        load_joint_policy_contract(root)


def test_paired_inference_rejects_precision_mismatch_before_loading_weights(tmp_path):
    from scripts.run_crn_qwen_hybrid import bind_joint_policy
    from scripts.pr83_resume import joint_controller_settings
    args = SimpleNamespace(dtype='float16', no_4bit=True)
    settings = joint_controller_settings(SimpleNamespace(dtype='bfloat16', no_4bit=True))
    with pytest.raises(ValueError, match='dtype'):
        bind_joint_policy(args, {'controller':settings}, tmp_path)


def test_joint_resume_refuses_partial_or_wrong_shape_model_snapshots():
    from scripts.pr83_resume import validate_joint_weights
    expected_qwen = {'lora_A':torch.zeros(2, 4)}
    expected_graph = {'head':torch.zeros(4, 4)}
    with pytest.raises(ValueError, match='Qwen'):
        validate_joint_weights(expected_qwen, expected_graph,
            {'qwen_policy':{}, 'graph_policy':expected_graph})
    with pytest.raises(ValueError, match='graph'):
        validate_joint_weights(expected_qwen, expected_graph,
            {'qwen_policy':expected_qwen, 'graph_policy':{'head':torch.zeros(3,4)}})
    validate_joint_weights(expected_qwen, expected_graph,
        {'qwen_policy':expected_qwen, 'graph_policy':expected_graph})


def test_source_lock_detects_objective_and_external_crn_changes(tmp_path):
    from scripts.pr83_resume import joint_source_fingerprints, verify_joint_sources
    root, crn = tmp_path/'code', tmp_path/'crn'
    (root/'src'/'mechet').mkdir(parents=True)
    (root/'scripts').mkdir()
    (crn/'src'/'crn_jepa').mkdir(parents=True)
    loss = root/'src'/'mechet'/'graph_endpoint_rl.py'
    external = crn/'src'/'crn_jepa'/'policy.py'
    loss.write_text('objective=1\n')
    external.write_text('decoder=1\n')
    before = joint_source_fingerprints(root, crn)
    verify_joint_sources(before, root, crn)
    loss.write_text('objective=2\n')
    assert joint_source_fingerprints(root, crn) != before
    with pytest.raises(ValueError, match='source contract'):
        verify_joint_sources(before, root, crn)
    loss.write_text('objective=1\n')
    external.write_text('decoder=2\n')
    assert joint_source_fingerprints(root, crn) != before
    with pytest.raises(ValueError, match='source contract'):
        verify_joint_sources(before, root, crn)
