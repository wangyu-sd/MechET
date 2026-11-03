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
    assert not f.credit_complete_route(complete)


def test_frontier_parameter_bounds():
    with pytest.raises(ValueError):
        ElectronicFrontier(widen_alpha=1.0)
    with pytest.raises(ValueError):
        ElectronicFrontier(c_puct=0)


def test_score_not_relying_on_gold_precursors():
    search=OneStepElectronicSearch.__dict__["run"]
    assert "gold" not in search.__code__.co_varnames
