"""Catch inverse labels, target leakage and unsafe forward adapter writes."""
import json
from pathlib import Path

import pytest

from mechet.natural_language_electron_flow import compile_event_arguments


def test_forward_substitution_executes_from_reactants_without_target_in_prompt():
    from mechet.forward_qwen_events import build_transition_row, execute_forward_call

    row = build_transition_row(
        '[OH-:1].[CH3:2][Cl:3]>>[OH:1][CH3:2].[Cl-:3]|7',
        reaction_id='flower-train-7', decision_index=0,
    )
    call = row['messages'][2]['tool_calls'][0]['function']
    assert call['name'] == 'apply_forward_electron_flow'
    assert call['arguments']['direction'] == 'forward'
    assert len(call['arguments']['electron_flow']) == 2
    result = execute_forward_call('[OH-:1].[CH3:2][Cl:3]', call['name'], call['arguments'])
    assert result['ok'] is True
    from mechet.proof_program import sides_equal
    assert sides_equal(result['state_smiles'], '[OH:1][CH3:2].[Cl-:3]', ignore_maps=False)
    prompt = row['messages'][1]['content']
    assert 'TARGET' not in prompt and 'ground_truth' not in prompt
    assert '[OH:1][CH3:2]' not in prompt
    # Old inverse entry point must still reject forward calls.
    with pytest.raises(ValueError, match='retrosynthetic'):
        compile_event_arguments('[OH-:1].[CH3:2][Cl:3]', call['arguments'])


def test_terminal_self_loop_emits_explicit_forward_finish():
    from mechet.forward_qwen_events import build_transition_row, execute_forward_call

    row = build_transition_row('[CH4:1]>>[CH4:1]|7', reaction_id='r7', decision_index=1)
    call = row['messages'][2]['tool_calls'][0]['function']
    assert call == {'name': 'finish_forward', 'arguments': {}}
    assert execute_forward_call('[CH4:1]', **call)['state_smiles'] == '[CH4:1]'


def test_forward_aliases_keep_private_identity_after_equivalent_mixture_reordering():
    from mechet.forward_qwen_events import build_transition_row, execute_forward_call
    from mechet.proof_program import sides_equal

    source = '[OH-:1].[OH-:4].[CH3:2][Cl:3]'
    reordered = '[OH-:4].[OH-:1].[CH3:2][Cl:3]'
    destination = '[OH:1][CH3:2].[Cl-:3].[OH-:4]'
    row = build_transition_row(source + '>>' + destination + '|7',
                               reaction_id='r7', decision_index=0)
    reordered_row = build_transition_row(reordered + '>>' + destination + '|7',
                                         reaction_id='r7', decision_index=0)
    call = row['messages'][2]['tool_calls'][0]['function']
    assert reordered_row['messages'][2]['tool_calls'][0]['function'] == call
    result = execute_forward_call(reordered, **call)
    assert result['ok']
    assert sides_equal(result['state_smiles'], destination, ignore_maps=False)


def test_proton_transfer_retains_explicit_hydrogen_and_replays():
    from mechet.forward_qwen_events import build_transition_row, execute_forward_call
    from mechet.proof_program import sides_equal

    source = '[O-:1][H:3].[O:2]([H:4])[H:5]'
    destination = '[O:1]([H:3])[H:4].[O-:2][H:5]'
    row = build_transition_row(source + '>>' + destination + '|9', reaction_id='r9', decision_index=0)
    call = row['messages'][2]['tool_calls'][0]['function']
    assert '<A' in row['messages'][1]['content'] and '[H]' in row['messages'][1]['content']
    result = execute_forward_call(source, **call)
    assert result['ok']
    assert sides_equal(result['state_smiles'], destination, ignore_maps=False)


def test_wrong_direction_and_unknown_tool_are_rejected():
    from mechet.forward_qwen_events import execute_forward_call

    with pytest.raises(ValueError, match='forward'):
        execute_forward_call('[CH4:1]', 'apply_forward_electron_flow', {'direction': 'retrosynthetic'})
    with pytest.raises(ValueError, match='unknown'):
        execute_forward_call('[CH4:1]', 'import_fragments', {})


def test_inverse_adapter_overwrite_guard_and_actual_hash(tmp_path):
    from mechet.forward_qwen_events import check_adapter_isolation
    import hashlib

    reverse = tmp_path / 'reverse'
    reverse.mkdir()
    (reverse / 'adapter_model.safetensors').write_bytes(b'reverse-weights')
    expected = hashlib.sha256(b'reverse-weights').hexdigest()
    check_adapter_isolation(reverse, tmp_path / 'forward', expected)
    with pytest.raises(ValueError, match='overlap'):
        check_adapter_isolation(reverse, reverse / 'checkpoint-1', expected)
    with pytest.raises(ValueError, match='SHA'):
        check_adapter_isolation(reverse, tmp_path / 'forward', '0' * 64)


def test_forward_tool_decision_passes_existing_assistant_only_validation():
    from mechet.forward_qwen_events import build_transition_row
    from scripts.train_tool_sft import validate_conversation

    row = build_transition_row('[CH4:1]>>[CH4:1]|7', reaction_id='r7', decision_index=1)
    counts = validate_conversation(row, require_trace_owned=False, require_tool_decision=True)
    assert counts['tool_calls'] == counts['tool_results'] == 1


def test_group_builder_retains_steps_and_attaches_only_prior_actions():
    from scripts.build_forward_qwen_event_sft import convert_group
    lines = [
        '[OH-:1].[CH3:2][Cl:3]>>[OH:1][CH3:2].[Cl-:3]|7',
        '[OH:1][CH3:2].[Cl-:3]>>[OH:1][CH3:2].[Cl-:3]|7',
    ]
    result = convert_group(('r7', lines))
    assert result['errors'] == []
    assert len(result['rows']) == 2
    assert 'ACCEPTED FORWARD HISTORY' not in result['rows'][0]['messages'][1]['content']
    assert 'ACCEPTED FORWARD HISTORY' in result['rows'][1]['messages'][1]['content']
    assert result['terminal_rows'] == 1


def test_source_groups_merges_interleaved_ids_without_dropping_repeated_rows(tmp_path):
    from scripts.build_forward_qwen_event_sft import source_groups

    first = '[OH-:1].[CH3:2][Cl:3]>>[OH:1][CH3:2].[Cl-:3]|7\n'
    other = '[CH4:1]>>[CH4:1]|8\n'
    last = '[OH:1][CH3:2].[Cl-:3]>>[OH:1][CH3:2].[Cl-:3]|7\n'
    raw = tmp_path / 'interleaved.txt'
    raw.write_text(first + other + last + last)

    assert list(source_groups(raw)) == [('7', [first, last, last]), ('8', [other])]


def test_history_follows_graph_predecessors_even_when_terminal_is_exported_first():
    from scripts.build_forward_qwen_event_sft import convert_group

    result = convert_group(('r7', [
        '[OH:1][CH3:2].[Cl-:3]>>[OH:1][CH3:2].[Cl-:3]|7',
        '[OH-:1].[CH3:2][Cl:3]>>[OH:1][CH3:2].[Cl-:3]|7',
    ]))
    assert result['errors'] == []
    terminal, initial = result['rows']
    assert terminal['metadata']['decision_index'] == 0
    history = json.loads(terminal['messages'][1]['content'].split(
        'ACCEPTED FORWARD HISTORY (temporary names belong to their own past states):\n')[1])
    assert history == [initial['messages'][2]['tool_calls'][0]['function']]
    assert 'ACCEPTED FORWARD HISTORY' not in initial['messages'][1]['content']
    assert terminal['metadata']['history_decision_indices'] == [1]
    assert terminal['metadata']['history_contract'] == 'recorded_forward_graph_path_actions_v2'


def test_history_at_a_join_uses_one_recorded_branch_not_both():
    from scripts.build_forward_qwen_event_sft import convert_group

    source = '[OH-:1].[CH3:2][Cl:3].[OH-:4].[CH3:5][Cl:6]'
    left = '[OH:1][CH3:2].[Cl-:3].[OH-:4].[CH3:5][Cl:6]'
    right = '[OH-:1].[CH3:2][Cl:3].[OH:4][CH3:5].[Cl-:6]'
    joined = '[OH:1][CH3:2].[Cl-:3].[OH:4][CH3:5].[Cl-:6]'
    result = convert_group(('diamond', [f'{a}>>{b}|9' for a, b in [
        (joined, joined), (left, joined), (source, left), (right, joined), (source, right),
    ]]))
    assert result['errors'] == []
    assert [row['metadata']['history_decision_indices'] for row in result['rows']] == [
        [2, 1], [2], [], [4], [],
    ]


def test_history_does_not_silently_treat_an_unrooted_cycle_as_an_initial_mixture():
    from scripts.build_forward_qwen_event_sft import convert_group

    result = convert_group(('r7', [
        '[OH-:1].[CH3:2][Cl:3]>>[OH:1][CH3:2].[Cl-:3]|7',
        '[OH:1][CH3:2].[Cl-:3]>>[OH-:1].[CH3:2][Cl:3]|7',
        '[OH:1][CH3:2].[Cl-:3]>>[OH:1][CH3:2].[Cl-:3]|7',
    ]))
    assert any('UNROOTED_RECORDED_FORWARD_STATE' in error['error'] for error in result['errors'])


def test_split_counts_interleaved_reaction_once_and_preserves_all_source_rows(tmp_path):
    from scripts.build_forward_qwen_event_sft import build_split

    raw = tmp_path / 'raw.txt'
    raw.write_text(
        '[OH-:1].[CH3:2][Cl:3]>>[OH:1][CH3:2].[Cl-:3]|7\n'
        '[CH4:1]>>[CH4:1]|8\n'
        '[OH:1][CH3:2].[Cl-:3]>>[OH:1][CH3:2].[Cl-:3]|7\n'
    )
    proof = tmp_path / 'proof.jsonl'
    proof.write_text(''.join(json.dumps({'id': f'r{i}', 'metadata': {'trajectory_id': i}})
                             + '\n' for i in [7, 8]))
    output = tmp_path / 'forward.jsonl'
    result = build_split(raw, proof, output, expected=2, workers=1)
    assert result['errors'] == 0
    assert result['reactions'] == 2
    assert result['rows'] == 3
    assert result['source_rows'] == 3
    assert result['source_rows_preserved'] is True
    assert result['terminal_rows'] == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['id'] for row in rows] == ['r7:forward:0', 'r7:forward:1', 'r8:forward:0']
    assert 'ACCEPTED FORWARD HISTORY' in rows[1]['messages'][1]['content']


def test_training_config_rejects_resuming_the_inverse_adapter(tmp_path):
    from scripts.build_forward_qwen_event_sft import render_training_config
    template = {'initial_adapter_path': '/old/reverse', 'contract': {}}
    with pytest.raises(ValueError, match='initial_adapter'):
        render_training_config(template, {'train': {'rows': 2}, 'valid': {'rows': 1}})


def test_token_audit_rejects_one_window_with_cropped_conditioning():
    from mechet.forward_qwen_events import validate_forward_token_cache
    audit = {'n_rows': 2, 'n_training_windows': 2, 'n_windowed_source_rows': 0,
             'max_raw_input_tokens': 16385, 'configured_max_length': 16384}
    with pytest.raises(ValueError, match='conditioning'):
        validate_forward_token_cache(audit, 2)


def test_forward_initializer_adds_explicit_hydrogens_without_target_or_external_maps():
    from mechet.forward_qwen_events import initialize_forward_state
    from rdkit import Chem
    from mechet.proof_program import sides_equal
    state = initialize_forward_state('[OH-].CCl')
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(state, params)
    assert mol.GetNumAtoms() == 7
    assert sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms()) == 4
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    assert sorted(maps) == list(range(1, 8))
    assert sides_equal(state, '[OH-].CCl')
