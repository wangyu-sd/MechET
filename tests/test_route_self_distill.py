"""TRAIN-only offline distillation data/loss software checks; no model execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ACTOR = 'a' * 64
EOS = 9


def api():
    assert importlib.util.find_spec('mechet.route_self_distill') is not None, 'self-distill primitive missing'
    from mechet import route_self_distill
    return route_self_distill


def fixture():
    import hashlib
    def sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    row = {'split': 'training', 'node_id': 'node-1', 'source_actor_sha': ACTOR,
           'behavior_policy_version': 'frozen-behavior-1', 'refit_config_sha': 'b' * 64,
           'budget_sha': 'c' * 64, 'tokenizer_sha': 'd' * 64, 'tool_schema_sha': 'e' * 64,
           'executor_sha': 'f' * 64, 'stock_contract_sha': '1' * 64,
           'external_teacher_calls': 0, 'prompt_token_ids': [1, 2],
           'conditioning': {'target_id': 'train-1', 'reaction_occurrence_id': 'reaction-1',
                            'observation_sha': '2' * 64, 'history_sha': '3' * 64,
                            'route_context_sha': '4' * 64, 'inventory_sha': '5' * 64}}
    binding = dict(row); binding.pop('split'); binding.pop('external_teacher_calls')
    conditioning_sha = sha(binding)
    receipts = {}
    row['candidates'] = []
    for i, (semantic, q) in enumerate([('flow-A', 0.0), ('flow-B', 2.0), ('flow-A', 2.0)]):
        c = {'semantic_key': semantic, 'generated_token_ids': [3+i, EOS], 'Q_search': q,
             'receipt_id': f'r{i}', 'conditioning_sha': conditioning_sha,
             'source_actor_sha': ACTOR, 'behavior_policy_version': row['behavior_policy_version'],
             'refit_config_sha': row['refit_config_sha'], 'budget_sha': row['budget_sha'],
             'proposal_kind': 'puct_search', 'generated_role': 'assistant'}
        receipt = dict(c, split='training', verified=True, completed=True, censored=False,
                       executor_accepted=True, trust_tier='reference_supported_closed', external_teacher_calls=0)
        receipts[c['receipt_id']] = receipt
        c['receipt_sha256'] = sha(receipt)
        row['candidates'].append(c)
    return row, receipts


def prepare(row, receipts, **kwargs):
    return api().prepare_state(row, receipts, expected_actor_sha=ACTOR,
                               training_target_ids={'train-1'}, eos_token_id=EOS, **kwargs)


def test_semantic_dedup_uses_mean_observed_Q_and_keeps_native_tokens_and_receipts():
    row, receipts = fixture()
    result = prepare(row, receipts, temperature=1.0)
    assert result['status'] == 'prepared'
    assert result['objective'] == 'offline_search_selected_native_ce_not_grpo'
    targets = result['targets']
    assert [v['semantic_key'] for v in targets] == ['flow-A', 'flow-B']
    assert [v['Q_search'] for v in targets] == [1.0, 2.0]
    assert [v['probability'] for v in targets] == pytest.approx([0.2689414213699951, 0.7310585786300049])
    assert targets[0]['generated_token_ids'] == [3, EOS]
    assert targets[0]['receipt_ids'] == ['r0', 'r2']
    assert result['prompt_token_ids'] == [1, 2]


@pytest.mark.parametrize('change,code', [('split', 'TRAINING_ONLY'), ('actor', 'ACTOR'),
                                       ('prompt', 'CONDITIONING'), ('node', 'CONDITIONING'),
                                       ('unknown_proposal', 'PROPOSAL'), ('budget', 'BINDING'),
                                       ('receipt_bytes', 'RECEIPT_SHA'), ('external_teacher', 'TEACHER'),
                                       ('outside_train', 'TRAIN_TARGET')])
def test_contamination_and_conditioning_mismatch_are_refused(change, code):
    row, receipts = fixture()
    if change == 'split': row['split'] = 'validation'
    elif change == 'actor': row['candidates'][0]['source_actor_sha'] = '0' * 64
    elif change == 'prompt': row['prompt_token_ids'] = [1, 7]
    elif change == 'node': row['node_id'] = 'different-node'
    elif change == 'unknown_proposal': row['candidates'][0]['proposal_kind'] = 'unknown'
    elif change == 'budget': row['candidates'][0]['budget_sha'] = '0' * 64
    elif change == 'receipt_bytes': receipts['r0']['Q_search'] = 20
    elif change == 'external_teacher': row['external_teacher_calls'] = 1
    else: row['conditioning']['target_id'] = 'test-1'
    with pytest.raises(ValueError, match=code): prepare(row, receipts)


def reseal(row, receipts):
    for c in row['candidates']:
        c['receipt_sha256'] = api().content_sha256(receipts[c['receipt_id']])


@pytest.mark.parametrize('bad_field', ['censored', 'rejected', 'unsupported', 'unverified', 'incomplete'])
def test_ineligible_candidates_never_become_positive_targets(bad_field):
    row, receipts = fixture()
    for receipt in receipts.values():
        if bad_field == 'censored': receipt['censored'] = True
        elif bad_field == 'rejected': receipt['executor_accepted'] = False
        elif bad_field == 'unsupported': receipt['trust_tier'] = 'stock_closed_unverified'
        elif bad_field == 'unverified': receipt['verified'] = False
        else: receipt['completed'] = False
    reseal(row, receipts)
    result = prepare(row, receipts)
    assert result['status'] == 'skipped' and result['reason'] == 'all_ineligible'
    assert result['targets'] == []


def test_equal_values_and_single_semantic_choice_are_skipped():
    row, receipts = fixture()
    for c in row['candidates']:
        c['Q_search'] = 1.0; receipts[c['receipt_id']]['Q_search'] = 1.0
    reseal(row, receipts)
    assert prepare(row, receipts)['reason'] == 'noninformative_values'
    row['candidates'] = [row['candidates'][0]]
    assert prepare(row, receipts)['reason'] == 'insufficient_distinct_actions'


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf')])
def test_invalid_softmax_temperature_is_rejected(temperature):
    row, receipts = fixture()
    with pytest.raises(ValueError, match='TEMPERATURE'): prepare(row, receipts, temperature=temperature)


def test_eligible_native_action_requires_eos_and_assistant_provenance():
    row, receipts = fixture()
    row['candidates'][0]['generated_token_ids'] = [3]
    receipts['r0']['generated_token_ids'] = [3]
    reseal(row, receipts)
    with pytest.raises(ValueError, match='EOS'): prepare(row, receipts)


def test_duplicate_receipt_cannot_double_weight_a_semantic_action():
    row, receipts = fixture(); row['candidates'].append(deepcopy(row['candidates'][0]))
    with pytest.raises(ValueError, match='DUPLICATE_RECEIPT'): prepare(row, receipts)


def test_weighted_native_ce_sums_response_tokens_including_eos_and_masks_prompt_padding():
    import torch
    logits = torch.zeros((2, 5, 12), dtype=torch.float64, requires_grad=True)
    loss = api().native_assistant_ce(logits, [[1, 2, 3], [1]], [[4, EOS], [5, 6, EOS]], [0.25, 0.75], EOS)
    assert loss.item() == pytest.approx(2.75 * __import__('math').log(12))
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, :2]) == 0
    assert torch.count_nonzero(logits.grad[0, 2:4]) > 0
    assert torch.count_nonzero(logits.grad[0, 4:]) == 0
    assert torch.count_nonzero(logits.grad[1, :3]) > 0
    assert torch.count_nonzero(logits.grad[1, 3:]) == 0


def test_ce_does_not_touch_nan_prompt_padding_or_zero_weight_actions():
    import torch
    logits = torch.zeros((2, 4, 12), requires_grad=True)
    with torch.no_grad(): logits[0, 0] = float('nan'); logits[0, 3] = float('nan'); logits[1] = float('nan')
    loss = api().native_assistant_ce(logits, [[1, 2], [1]], [[4, EOS], [5, EOS]], [1.0, 0.0], EOS)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[1]) == 0


def anchor_fixture(kind='original_sft'):
    row, receipts = fixture()
    anchor = {'split': 'training', 'anchor_kind': kind, 'source_artifact_sha': '6' * 64,
              'source_actor_sha': ACTOR if kind == 'actor_state' else None,
              'source_example_id': 'training-example-1', 'generated_role': 'assistant', 'prompt_token_ids': [1, 2],
              'generated_token_ids': [3, EOS], 'receipt_id': 'anchor1', 'external_teacher_calls': 0}
    receipt = dict(anchor, verified=True, completed=True, censored=False, executor_accepted=True)
    anchor['receipt_sha256'] = api().content_sha256(receipt)
    return anchor, {'anchor1': receipt}


def test_checked_anchors_have_explicit_lineage_and_mixture_mass():
    a = api(); row, receipts = fixture(); search = prepare(row, receipts)
    anchors = []
    for kind in ['original_sft', 'actor_state']:
        raw, proofs = anchor_fixture(kind)
        anchors.append(a.prepare_anchor(raw, proofs, expected_actor_sha=ACTOR,
                                       expected_source_sha='6' * 64, eos_token_id=EOS))
    mixed = a.mix_examples([search], anchors, sft_weight=0.1, actor_weight=0.2)
    assert [v['mixture_weight'] for v in mixed] == pytest.approx([0.7, 0.1, 0.2])
    assert [v['kind'] for v in mixed] == ['self_search', 'original_sft', 'actor_state']
    assert mixed[1]['source_artifact_sha'] == '6' * 64
    with pytest.raises(ValueError, match='MISSING_ANCHOR'):
        a.mix_examples([search], [], sft_weight=0.1, actor_weight=0)


def test_anchor_test_split_or_wrong_source_lineage_cannot_enter_mix():
    raw, proofs = anchor_fixture(); raw['split'] = 'test'
    with pytest.raises(ValueError, match='TRAINING_ONLY'):
        api().prepare_anchor(raw, proofs, expected_actor_sha=ACTOR, expected_source_sha='6' * 64, eos_token_id=EOS)
    raw, proofs = anchor_fixture()
    with pytest.raises(ValueError, match='ANCHOR_SOURCE'):
        api().prepare_anchor(raw, proofs, expected_actor_sha=ACTOR, expected_source_sha='7' * 64, eos_token_id=EOS)


def test_cli_prepares_dataset_and_refuses_existing_output_or_holdout_target(tmp_path):
    row, receipts = fixture()
    inp = tmp_path/'states.jsonl'; inp.write_text(json.dumps(row)+'\n')
    rp = tmp_path/'receipts.json'; rp.write_text(json.dumps({'verifier_sha': '7'*64, 'receipts': receipts}))
    tp = tmp_path/'training.jsonl'; tp.write_text('{"task_id":"train-1"}\n')
    script = Path(__file__).resolve().parents[1]/'scripts/prepare_route_self_distill.py'
    command = [sys.executable, str(script), '--input', str(inp), '--receipt-manifest', str(rp),
               '--training-targets', str(tp), '--output-dir', str(tmp_path/'prepared'),
               '--source-actor-sha', ACTOR, '--eos-token-id', str(EOS)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path/'prepared/manifest.json').read_text())
    assert manifest['prepared_states'] == 1 and manifest['trained_checkpoint'] is None
    assert manifest['initialized_from_actor_sha'] == ACTOR
    assert manifest['external_teacher_calls'] == 0
    assert subprocess.run(command, capture_output=True).returncode != 0
    tp.write_text('{"task_id":"other-training-target"}\n')
    command[command.index('--output-dir')+1] = str(tmp_path/'holdout-blocked')
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not (tmp_path/'holdout-blocked').exists()


def test_one_eligible_success_against_trusted_rejection_yields_unit_positive_target():
    row, receipts = fixture(); row['candidates'] = row['candidates'][:2]
    for c, q in zip(row['candidates'], [1.0, 0.0]):
        c['Q_search'] = q; receipts[c['receipt_id']]['Q_search'] = q
    receipts['r1']['executor_accepted'] = False
    receipts['r1']['trust_tier'] = 'executor_rejected'
    reseal(row, receipts)
    result = prepare(row, receipts)
    assert result['status'] == 'prepared'
    assert len(result['targets']) == 1 and result['targets'][0]['probability'] == 1.0
    assert result['targets'][0]['semantic_key'] == 'flow-A'
    assert result['informativeness']['trusted_semantic_actions'] == 2


def test_equal_success_Q_still_trains_when_another_trusted_semantic_action_failed():
    row, receipts = fixture()
    for c in row['candidates']:
        c['Q_search'] = 1.; receipts[c['receipt_id']]['Q_search'] = 1.
    # Introduce a third observed semantic choice with a genuine failed outcome.
    c = deepcopy(row['candidates'][0]); c.update(receipt_id='failed', semantic_key='flow-C', Q_search=0.)
    receipt = dict(receipts['r0'], receipt_id='failed', semantic_key='flow-C', Q_search=0.,
                   executor_accepted=False, trust_tier='executor_rejected')
    receipts['failed'] = receipt; row['candidates'].append(c); reseal(row, receipts)
    result = prepare(row, receipts)
    assert result['status'] == 'prepared'
    assert [t['probability'] for t in result['targets']] == [0.5, 0.5]
    assert result['informativeness']['trusted_semantic_actions'] == 3


def test_censored_rejection_cannot_supply_the_only_discriminating_outcome():
    row, receipts = fixture(); row['candidates'] = row['candidates'][:2]
    for c, q in zip(row['candidates'], [1., 0.]):
        c['Q_search'] = q; receipts[c['receipt_id']]['Q_search'] = q
    receipts['r1'].update(executor_accepted=False, censored=True)
    reseal(row, receipts)
    assert prepare(row, receipts)['status'] == 'skipped'


def test_nonassistant_generated_tokens_cannot_be_positive_targets():
    row, receipts = fixture()
    row['candidates'][0]['generated_role'] = 'tool'; receipts['r0']['generated_role'] = 'tool'
    reseal(row, receipts)
    with pytest.raises(ValueError, match='ASSISTANT'): prepare(row, receipts)


@pytest.mark.parametrize('bad_weights', [[-1., 2.], [0., 0.], [float('nan'), 1.]])
def test_ce_rejects_bad_source_weights(bad_weights):
    import torch
    with pytest.raises(ValueError, match='WEIGHTS'):
        api().native_assistant_ce(torch.zeros(2, 4, 12), [[1], [1]], [[2, EOS], [3, EOS]], bad_weights, EOS)


def test_cli_does_not_import_neural_packages(tmp_path):
    row, receipts = fixture()
    inp=tmp_path/'states.jsonl'; inp.write_text(json.dumps(row)+'\n')
    rp=tmp_path/'receipts.json'; rp.write_text(json.dumps({'verifier_sha':'7'*64,'receipts':receipts}))
    tp=tmp_path/'training.jsonl'; tp.write_text('{"id":"train-1"}\n')
    script=Path(__file__).resolve().parents[1]/'scripts/prepare_route_self_distill.py'
    args=[str(script),'--input',str(inp),'--receipt-manifest',str(rp),'--training-targets',str(tp),
          '--output-dir',str(tmp_path/'no-neural'), '--source-actor-sha',ACTOR,'--eos-token-id',str(EOS)]
    wrapper='''import builtins, runpy, sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','retrochimera','syntheseus'}:
        raise ImportError('NEURAL_IMPORT_FORBIDDEN:'+name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-c',wrapper,*args],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='prepared_not_trained'


def test_duplicate_anchor_or_search_state_cannot_change_mixture_weights():
    row, receipts = fixture(); search = prepare(row, receipts)
    with pytest.raises(ValueError, match='DUPLICATE'):
        api().mix_examples([search, deepcopy(search)], [], sft_weight=0, actor_weight=0)
    raw, proofs = anchor_fixture()
    anchor = api().prepare_anchor(raw, proofs, expected_actor_sha=ACTOR, expected_source_sha='6'*64, eos_token_id=EOS)
    with pytest.raises(ValueError, match='DUPLICATE'):
        api().mix_examples([search], [anchor, deepcopy(anchor)], sft_weight=.1, actor_weight=0)


def test_cli_all_ineligible_writes_explicit_zero_coverage_without_arbitrary_labels(tmp_path):
    row, receipts = fixture()
    for receipt in receipts.values(): receipt['censored'] = True
    reseal(row, receipts)
    inp=tmp_path/'states.jsonl'; inp.write_text(json.dumps(row)+'\n')
    rp=tmp_path/'receipts.json'; rp.write_text(json.dumps({'verifier_sha':'7'*64,'receipts':receipts}))
    tp=tmp_path/'training.jsonl'; tp.write_text('{"task_id":"train-1"}\n')
    result=api().prepare_dataset(inp,rp,tp,tmp_path/'empty',expected_actor_sha=ACTOR,eos_token_id=EOS)
    assert result['prepared_states']==0 and result['skipped_states']==1
    assert result['trainable'] is False
    assert (tmp_path/'empty/targets.jsonl').read_text()==''
    assert json.loads((tmp_path/'empty/skipped.jsonl').read_text())['reason']=='all_ineligible'
