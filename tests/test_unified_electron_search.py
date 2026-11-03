"""No-GPU semantic tests for electron-only route search and stock AND closure."""
import pytest
from types import SimpleNamespace

from mechet.unified_electron_search import ElectronicFrontier, OneStepElectronicSearch
from mechet.reverse_route_network import stock_closure
from mechet.agentic_hf import STAGE2


def fake_ctx(key="root", score=-1.0, terminal=False, batches=0):
    node=SimpleNamespace(terminal=terminal,policy_score=score,actions=[])
    return SimpleNamespace(key=key,node=node,resource_closed="",
                           stats=SimpleNamespace(needs_more_sampling=True,batches=batches))


def test_only_flower_stage2_is_training_source():
    assert STAGE2["flower"]["sha256"] == "2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453"


def test_finish_marker_is_not_a_widening_successor():
    tree=SimpleNamespace(contexts={"root":fake_ctx()},
            edges=[{"parent_context":"root", "child_context":"finish",
                    "child_chemical":"same", "action":{"name":"finish_trace"}}])
    f=ElectronicFrontier()
    assert f.successors(tree,"root")==set()
    assert f.select({"P":tree})[1].key=="root"


def test_progressive_widening_is_limited_by_chemical_successors():
    tree=SimpleNamespace(contexts={"root":fake_ctx()},
          edges=[{"parent_context":"root","child_context":f"n{i}",
                  "child_chemical":f"chem{i}","action":{"name":"apply_electron_flow"}}
                 for i in range(2)])
    f=ElectronicFrontier(widen_k=1.0)
    assert f.select({"P":tree}) is None
    f.visits["root"]=4
    assert f.select({"P":tree})[1].key=="root"


def test_import_is_charged_and_kept_as_chemical_successor():
    tree=SimpleNamespace(contexts={"root":fake_ctx()},
          edges=[{"parent_context":"root","child_context":"child",
                  "child_chemical":"imported","action":{"name":"import_fragments"}}])
    assert len(ElectronicFrontier.successors(tree,"root"))==1


def test_stock_closure_requires_all_precursors():
    edge={"id":"r","product":"P","precursors":["A","B"],"cost":1.0,"tokens":3}
    assert "P" not in stock_closure({"A"},[edge])
    assert "P" in stock_closure({"A","B"},[edge])


def test_positive_backup_only_on_full_route():
    f=ElectronicFrontier()
    f.parents={"finish":"flow", "flow":"root"}
    partial={"route_key":"bad","complete_reaction_edges":[]}
    assert not f.credit_complete_route(partial)
    assert f.rewards=={}
    complete={"route_key":"ok","complete_reaction_edges":[
         {"program":{"terminal_context":"finish"}}]}
    assert f.credit_complete_route(complete)
    assert f.rewards["flow"]==1.0 and f.rewards["root"]==1.0
    assert "finish" not in f.rewards
    assert not f.credit_complete_route(complete)


def test_frontier_parameter_bounds():
    with pytest.raises(ValueError):
        ElectronicFrontier(widen_alpha=1.0)
    with pytest.raises(ValueError):
        ElectronicFrontier(c_puct=0)


def test_score_not_relying_on_gold_precursors():
    search=OneStepElectronicSearch.__dict__["run"]
    assert "gold" not in search.__code__.co_varnames


def test_real_executable_flow_and_stock_closure(tmp_path):
    """Real executor, fake policy; demonstrates FINISH as a marker."""
    import json
    from test_pr96_history import state_rows
    from test_main_product_training_data import legacy_sn2
    from scripts.build_main_product_training_data import prepare_reference
    from mechet.endpoints import split_precursor_endpoints
    from mechet.dynamic_electron_tree import chemical_key
    from mechet.unified_electron_search import UnifiedRouteSearch

    source=prepare_reference(legacy_sn2())
    actions=[r["messages"][2]["tool_calls"][0]["function"] for r in state_rows()]
    assert [a["name"] for a in actions]==[
        "import_fragments","apply_electron_flow","finish_trace"]
    core=split_precursor_endpoints(
        source["full_precursor_state"],source["target_smiles"]).structural
    class FakeActor:
        def __init__(self): self.position=0
        def generate(self,messages,*,max_new_tokens,seed):
            if self.position>=len(actions):
                raise AssertionError("unexpected fourth call")
            call=actions[self.position]
            self.position+=1
            return {"raw":"<tool_call>"+json.dumps(call)+"</tool_call>",
                    "generated_token_ids":[42],"truncated":False,"logprob":-1.0}
    agent=FakeActor()
    run=UnifiedRouteSearch(case_id="synthetic-real-executor",
          product=source["target_smiles"],stock={chemical_key(core)},
          stock_sha="fixture-stock",actor=agent,actor_sha="fixture-actor",
          output=tmp_path,max_calls=3,max_tokens=1000,max_new_tokens=512)
    result=run.run(stop_on_first=True)
    assert result["stock_solved"] is True
    assert result["counters"]["electron_events"]==1
    assert result["counters"]["import_preparations"]==1
    assert result["counters"]["finish_markers"]==1
    assert result["counters"]["successful_route_backups"]==1
    assert not any("finish" in k.lower() for k in result["puct_statistics"]["parents"])
    assert (tmp_path/"report.json").is_file()
