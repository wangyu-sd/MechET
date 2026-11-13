from copy import deepcopy

import pytest

from test_route_memory_integration import episode
from test_route_action_feedback import send


def unresolved_episode():
    ep = episode()
    send(ep, 'finish_trace', {})
    send(ep, 'import_fragments', {'fragments': [dict(smiles='[Cl-]', count=1, purpose='electron_participant')]})
    for _ in range(8):
        send(ep, 'finish_trace', {})
    assert ep.status == 'recovery_exhausted'
    return ep


def recovery_episode():
    from mechet.route_memory_integration import MemoryTargetOnlyRouteEpisode
    from mechet.natural_language_electron_flow import build_inventory
    from mechet.route_training_contract import chemical_identity
    from rdkit import Chem
    from test_route_memory_integration import empty_config
    def args(ep):
        ep.next_decision(); inv=build_inventory(ep.node.state); mol=Chem.MolFromSmiles(ep.node.state)
        a={atom.GetSymbol():inv.map_to_atom[atom.GetAtomMapNum()] for atom in mol.GetAtoms() if atom.GetSymbol() in ('C','O')}
        return dict(direction='retrosynthetic', electron_flow=[dict(source=f"the bond between atoms {a['C']} and {a['O']}",destination='atom '+a['O'],instruction='transfer')],bond_order_changes=[],charge_changes=[])
    probe=episode(); send(probe,'apply_electron_flow',args(probe))
    stock=set(chemical_identity(probe.chemistry.precursors(probe.node,probe.initial)).split('.'))
    ep=MemoryTargetOnlyRouteEpisode(empty_config(),'CO',stock,'recovered',17)
    send(ep,'finish_trace',{}); send(ep,'apply_electron_flow',args(ep));send(ep,'finish_trace',{})
    assert ep.status=='stock_solved'
    return ep


def test_actual_replay_builds_recovery_memory_without_claiming_route_success(tmp_path):
    from mechet.route_memory_curation import curate_episode
    from mechet.route_experience_memory import ExperienceMemory
    ep = recovery_episode()
    result = curate_episode(ep.snapshot(), task_id='train-1', public_tasks={'train-1': 'CO'},
                            protected=set(), output_dir=tmp_path)
    assert len(result['memory']['entries']) == 1
    assert result['supervision'] == []
    assert result['reference_verified'] is False
    restored = ExperienceMemory(result['memory'])
    entry = next(iter(restored.entries.values()))
    assert entry['content']['strategy_id'] == 'continue_before_finish'
    assert entry['successes'] == 1
    second = curate_episode(ep.snapshot(), task_id='train-1', public_tasks={'train-1': 'CO'},
                            protected=set(), output_dir=tmp_path)
    assert second == result


def test_verifier_rejects_forged_execution_and_heldout_learning(tmp_path):
    from mechet.route_memory_curation import curate_episode
    saved = recovery_episode().snapshot()
    kwargs = dict(task_id='train-1', public_tasks={'train-1': 'CO'}, protected=set(), output_dir=tmp_path)
    forged = deepcopy(saved); forged['records'][0]['accepted'] = True
    with pytest.raises(ValueError):
        curate_episode(forged, **kwargs)
    with pytest.raises(ValueError, match='TRAIN'):
        curate_episode(saved, **dict(kwargs, split='validation'))
    with pytest.raises(ValueError, match='PROTECTED'):
        curate_episode(saved, **dict(kwargs, protected={'CO'}))
    with pytest.raises(ValueError, match='PUBLIC_TRAIN'):
        curate_episode(saved, **dict(kwargs, public_tasks={}))


def test_unfinished_or_resource_interrupted_episode_cannot_teach_memory(tmp_path):
    from mechet.route_memory_curation import curate_episode
    ep = episode(); ep.next_decision()
    with pytest.raises(ValueError, match='COMPLETE'):
        curate_episode(ep.snapshot(), task_id='x', public_tasks={'x': 'CO'}, protected=set(), output_dir=tmp_path)

    req = ep.next_decision()
    ep.accept(dict(request_id=req.request_id, behavior_version=req.behavior_version,
                   raw='', prompt_token_ids=[1], generated_token_ids=[], behavior_logprobs=[], resource_error='OOM'))
    with pytest.raises(ValueError, match='CENSORED'):
        curate_episode(ep.snapshot(), task_id='x', public_tasks={'x': 'CO'}, protected=set(), output_dir=tmp_path)


def test_memory_supervision_is_a_real_paired_tool_row_without_future_input():
    from mechet.route_memory_curation import supervised_decision
    from mechet.route_action_feedback import tools_for_request
    from scripts.train_tool_sft import validate_conversation
    ep = episode(); req = ep.next_decision()
    send(ep, 'import_fragments', {'fragments': [dict(smiles='[Cl-]', count=1, purpose='electron_participant')]})
    row = supervised_decision(ep.records[0], tools_for_request(req), source_sha='a'*64,
                              memory_sha=ep.cfg['route_memory']['sha256'])
    counts = validate_conversation(row, require_trace_owned=False, require_tool_decision=True)
    assert counts['tool_calls'] == counts['tool_results'] == 1
    assert row['messages'][:-2] == req.messages
    assert row['messages'][-2]['content'] == ''
    assert row['metadata']['history_contains_failed_actions'] is True


def test_merely_accepted_import_is_not_successful_error_recovery(tmp_path):
    from mechet.route_memory_curation import curate_episode
    ep=unresolved_episode()
    result=curate_episode(ep.snapshot(),task_id='x',public_tasks={'x':'CO'},protected=set(),output_dir=tmp_path)
    assert result['executor_recovery_count']==0
    assert result['memory']['entries']==[]
