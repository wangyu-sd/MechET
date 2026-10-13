"""Real-executor controller tests, not chemical model performance experiments."""
import math
import pytest


def test_sampling_breadth_grows_with_new_chemical_support_not_fixed_beam():
    from mechet.dynamic_electron_tree import NoveltyStats
    high = NoveltyStats()
    low = NoveltyStats()
    for index in range(40):
        high.observe(str(index))
        low.observe('same')
    assert high.needs_more_sampling
    assert not low.needs_more_sampling
    assert high.next_batch > low.next_batch


def test_same_chemical_state_different_history_is_not_destroyed():
    from mechet.dynamic_electron_tree import context_key, chemical_key
    from scripts.run_natural_language_value_search import Node
    a = Node('CO', '[CH3:1][OH:2]', 3, visited={'CO'})
    b = Node('CO', '[CH3:1][OH:2]', 3, visited={'CO','CCl'})
    assert chemical_key(a.state) == chemical_key(b.state)
    assert context_key(a) != context_key(b)


def test_rejected_and_duplicate_samples_still_count_and_are_journaled():
    from mechet.dynamic_electron_tree import ElectronTree, Proposal
    tree = ElectronTree('case', 'CO')
    root = tree.next_context()
    assert root is not None
    p = Proposal('import_fragments', {'fragments':[{'smiles':'[Cl-]','count':1,'purpose':'electron_participant'}]}, 'raw', -1, 5)
    bad = Proposal('not_a_tool', {}, 'bad', -2, 7)
    tree.incorporate(root.key, [p,p,bad])
    assert tree.work['sampled_tokens'] == 17
    assert tree.work['raw_candidates'] == 3
    assert len(tree.edges) == 3
    assert sum(e['status']=='duplicate_successor' for e in tree.edges) == 1
    assert sum(e['status']=='rejected' for e in tree.edges) == 1
    assert len(tree.contexts) == 2


def test_more_than_40_steps_is_not_truncated_by_chemical_depth_cap():
    from mechet.dynamic_electron_tree import ElectronTree, Proposal
    tree = ElectronTree('case', 'CO')
    key = tree.root_key
    for index in range(42):
        p = Proposal('import_fragments', {'fragments':[{'smiles':'[Cl-]','count':1,'purpose':'electron_participant'}]}, 'raw', -1, 5)
        tree.incorporate(key,[p])
        key = tree.edges[-1]['child_context']
    assert len(tree.contexts[key].node.actions) == 42
    assert tree.contexts[key].node.imported['[Cl-]'] == 42


def test_finish_does_not_hide_remaining_search_contexts_or_read_gold():
    from mechet.dynamic_electron_tree import ElectronTree, Proposal
    tree = ElectronTree('case','CO')
    root = tree.next_context()
    tree.incorporate(root.key,[Proposal('finish_trace',{},'finish',-1,3)])
    assert len(tree.terminals()) == 1
    assert tree.next_context() is not None
    report=tree.to_dict()
    assert 'ground_truth' not in report
    assert 'expected_precursor' not in report


def test_unwarped_vllm_probability_contract_rejects_temperature_ranking():
    from mechet.dynamic_electron_tree import validate_probability_contract
    validate_probability_contract(temperature=1, top_p=1, repetition_penalty=1)
    with pytest.raises(ValueError, match='unwarped'):
        validate_probability_contract(temperature=.7, top_p=.95, repetition_penalty=1)


def test_truncation_increases_token_room_once_per_batch_not_per_sample():
    from mechet.dynamic_electron_tree import ElectronTree, Proposal
    tree = ElectronTree('case', 'CO')
    root = tree.next_context()
    tree.incorporate(root.key,[Proposal('',{},'cut',-1,384,truncated=True)]*8)
    assert root.max_new_tokens == 768
    assert root.stats.draws == 0


def test_malformed_import_is_a_recorded_rejection_not_a_crash():
    from mechet.dynamic_electron_tree import ElectronTree, Proposal
    tree = ElectronTree('case','CO')
    root = tree.next_context()
    tree.incorporate(root.key,[Proposal('import_fragments',{'fragments':None},'bad',-1,5)])
    assert tree.edges[0]['status'] == 'rejected'


def test_native_kekule_executor_states_have_distinct_contexts():
    from mechet.dynamic_electron_tree import context_key, chemical_key
    from scripts.run_natural_language_value_search import Node
    a = Node('c1ccccc1', '[CH:1]1=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6]1', 7)
    b = Node('c1ccccc1', '[CH:1]1[CH:2]=[CH:3][CH:4]=[CH:5][CH:6]=1', 7)
    assert chemical_key(a.state) == chemical_key(b.state)
    assert context_key(a) != context_key(b)


def test_terminal_pooling_preserves_product_atom_ancestry():
    from mechet.dynamic_electron_tree import ElectronTree, SearchContext, context_key, chemical_key, digest
    from scripts.run_natural_language_value_search import Node
    from scripts.infer_dynamic_electron_trees import evaluate
    tree = ElectronTree('case', 'CO')
    for state, score in [('[CH3:1][OH:2].[OH2:3]', -1), ('[CH3:1][OH:3].[OH2:2]', -2)]:
        node = Node('CO', state, 4, terminal=True, logprob=score, tokens=1)
        key = context_key(node)
        tree.contexts[key] = SearchContext(key, node, digest(chemical_key(state)))
    assert len(tree.terminals()) == 2
    result = evaluate(tree, {'structural_precursor':'CO.O'})
    assert result['any_endpoint_hit']
    assert len(result['ranked_candidates']) == 2


def test_structural_ranking_deduplicates_private_address_permutations():
    from mechet.dynamic_electron_tree import ElectronTree, SearchContext, context_key, chemical_key, digest
    from scripts.run_natural_language_value_search import Node
    from scripts.infer_dynamic_electron_trees import evaluate
    tree = ElectronTree('case', 'CCO')
    for state, score in [('[CH4:1].[CH3:2][OH:3]', -1), ('[CH4:2].[CH3:1][OH:3]', -2)]:
        node = Node('CCO', state, 4, terminal=True, logprob=score, tokens=1)
        key = context_key(node)
        tree.contexts[key] = SearchContext(key, node, digest(chemical_key(state)))
    result = evaluate(tree, {'structural_precursor':'C.CO'})
    assert len(result['ranked_candidates']) == 1
    assert result['top1']
    assert result['ranked_candidates'][0]['mean_nll'] == 1


def test_infinite_import_count_is_recorded_rejection():
    from mechet.dynamic_electron_tree import ElectronTree, Proposal
    tree = ElectronTree('case', 'CO')
    tree.incorporate(tree.root_key, [Proposal('import_fragments',
        {'fragments':[{'smiles':'O','count':float('inf'),'purpose':'electron_participant'}]}, 'bad', -1, 5)])
    assert tree.edges[0]['status'] == 'rejected'


def test_empty_wave_continues_after_context_closures_not_after_budget_stop():
    from scripts.infer_dynamic_electron_trees import continue_empty_wave
    assert continue_empty_wave(has_queue=True, budget_blocked=False)
    assert not continue_empty_wave(has_queue=False, budget_blocked=False)
    assert not continue_empty_wave(has_queue=True, budget_blocked=True)


def test_memory_guard_is_physical_and_reserves_room_for_persistence():
    from scripts.infer_dynamic_electron_trees import HostMemoryGuard
    guard = HostMemoryGuard(world=8, initial_rss=100, available=8000)
    assert not guard.exhausted(rss=699, available=8000)
    assert guard.exhausted(rss=700, available=8000)
    assert guard.exhausted(rss=100, available=10)


def test_memory_probe_respects_container_limit(monkeypatch):
    from pathlib import Path
    from mechet.dynamic_electron_tree import available_host_memory
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=100000))
    files = {'/sys/fs/cgroup/memory.max':'1000', '/sys/fs/cgroup/memory.current':'800'}
    def read(path):
        if str(path) not in files:
            raise FileNotFoundError
        return files[str(path)]
    monkeypatch.setattr(Path, 'read_text', read)
    assert available_host_memory() == 200


def test_fragment_allocation_uses_actual_memory_and_fragment_size(monkeypatch):
    import mechet.dynamic_electron_tree as module
    monkeypatch.setattr(module, 'available_host_memory', lambda: 10_000_000)
    tree = module.ElectronTree('case','CO')
    tree.incorporate(tree.root_key,[module.Proposal('import_fragments',
        {'fragments':[{'smiles':'C'*100,'count':1,'purpose':'electron_participant'}]}, 'large', -1, 5)])
    assert tree.edges[0]['error'] == 'STATE_ALLOCATION_RESOURCE_FAILURE'
    assert tree.work['executor_calls'] == 0
