import pytest
from types import SimpleNamespace


def test_missing_nll_is_not_silently_ranked_as_zero():
    from scripts.infer_single_reaction_mcts import decode_request
    output=SimpleNamespace(request_id='x',num_cached_tokens=0,outputs=[SimpleNamespace(
        token_ids=[1],cumulative_logprob=None,finish_reason='stop')])
    with pytest.raises(RuntimeError,match='NLL'):
        decode_request(output,SimpleNamespace(decode=lambda *a,**k:'x'),lambda *a:('finish_trace',{},None))


def test_raw_request_record_keeps_tokens_scores_and_finish_reason():
    from scripts.infer_single_reaction_mcts import decode_request
    output=SimpleNamespace(request_id='x',num_cached_tokens=2,outputs=[SimpleNamespace(
        token_ids=[5,6],cumulative_logprob=-3.,finish_reason='length')])
    record=decode_request(output,SimpleNamespace(decode=lambda *a,**k:'raw'),lambda *a:('finish_trace',{},None))
    assert record['generations'][0]['logprob']==-3.
    assert record['generations'][0]['token_ids']==[5,6]
    assert record['generations'][0]['truncated'] is True


def test_row_projection_never_passes_gold_fields_to_search(tmp_path):
    from scripts.infer_single_reaction_mcts import make_controller
    from scripts.single_reaction_mcts import SingleReactionMCTSStore
    row={'id':'test','product_unmapped':'CO','reactants':'INVALID GOLD','reference':'do not expose'}
    c=make_controller(row,mcts_store=SingleReactionMCTSStore(tmp_path,{}),shared_store=None,
                      checkpoint_path=None,program_goal=10)
    assert c.target=='CO' and c.case_id=='test'
    assert not hasattr(c,'reference') and not hasattr(c,'reactants')
    c.close()


def test_resource_limited_reports_do_not_seal_a_complete_search():
    from scripts.infer_single_reaction_mcts import search_is_complete
    assert search_is_complete([{'stop_reason':'MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED'}],1)
    assert not search_is_complete([{'stop_reason':'CONTEXT_MEMORY_BUDGET'}],1)
    assert not search_is_complete([],1)


def test_short_truncated_completion_does_not_pretend_to_exhaust_full_context():
    from scripts.infer_single_reaction_mcts import completion_hit_context_limit
    from mechet.dynamic_electron_tree import Proposal
    truncated=[Proposal('','', 'partial',-1.,384,truncated=True)]
    assert not completion_hit_context_limit({'max_tokens':384,'prompt_ids':[1]*1000},truncated,8192)
    assert completion_hit_context_limit({'max_tokens':32,'prompt_ids':[1]*8160},truncated,8192)
