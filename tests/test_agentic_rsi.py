"""RSI provenance, AND closure and decision-change regression tests.

These use deterministic fixtures only, never a frozen Stage-II checkpoint.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mechet.agentic_rsi import (
    AgentSession, ExperienceMemory, checksum, electron_signature,
    harvest_candidate, load_memory, promote, verify_promotion,
)
from mechet.reverse_route_network import stock_closure
from mechet.shared_electron_search import durable_json


ACTOR="a"*64
STOCK="s"*64


def memory_entry(product="CCO"):
    item={"actor_sha":ACTOR,"stock_sha":STOCK,"product":product,
          "source_split":"TRAIN","route_solved":True,
          "flow_actions":[],"evidence":{"case_id":"train-01",
                    "route_key":"r1","edge_id":"e1","program_sha256":"p1"}}
    item["id"]=checksum(item)
    return item


def test_and_stock_closure_never_counts_partially_available_route():
    edges=[{"id":"r1","product":"target","precursors":["A","B"],
            "cost":1.0,"tokens":10}]
    assert "target" not in stock_closure({"A"},edges)
    edges.append({"id":"r2","product":"B","precursors":["A"],
                  "cost":1.0,"tokens":5})
    closure=stock_closure({"A"},edges)
    assert closure["target"]["edge"]=="r1"
    assert closure["target"]["tokens"]==15


def test_electron_signature_comes_from_executed_mapped_state_delta():
    before="[CH3:1][CH2:2][OH:3]"
    after="[CH3:1].[CH2:2][OH:3]"
    actions=[{"name":"apply_electron_flow","state_before":before},
             {"name":"finish_trace","state_before":after}]
    signature=electron_signature(actions,after)
    assert len(signature)==1
    assert "C-C:-1" in signature[0]


def test_harvest_needs_actual_complete_stock_closed_route():
    run={"id":"train-01","route_key":"route-a","stock_solved":True,
         "complete_reaction_edges":[{"id":"edge1","product":"CCO",
             "program":{"actions":[
                 {"name":"apply_electron_flow",
                  "state_before":"[CH3:1][CH2:2][OH:3]"},
                 {"name":"finish_trace",
                  "state_before":"[CH3:1].[CH2:2][OH:3]"}],
             "full_terminal_state":"[CH3:1].[CH2:2][OH:3]"}}]}
    entries=harvest_candidate(run,actor_sha=ACTOR,stock_sha=STOCK)
    assert len(entries)==1
    assert entries[0]["evidence"]["case_id"]=="train-01"
    assert entries[0]["flow_actions"][0]
    assert harvest_candidate({**run,"stock_solved":False},
                             actor_sha=ACTOR,stock_sha=STOCK)==[]
    with pytest.raises(ValueError,match="TRAIN_ONLY"):
        harvest_candidate(run,actor_sha=ACTOR,stock_sha=STOCK,source_split="EVAL")


def dev_pairs(n=8,*,gain=True,changed=2):
    pairs=[]
    for i in range(n):
        base={"budget":{"max_tokens":12000,"max_actions":128},
              "seed":17,"real_executor":True,"route_solved":False,
              "memory_decision_changes":0}
        candidate={**base,"route_solved":bool(gain and i==0),
                   "memory_decision_changes":changed if i==0 else 0}
        pairs.append({"id":f"dev-{i}","split":"DEV","actor_sha":ACTOR,
                      "stock_sha":STOCK,"baseline":base,"candidate":candidate})
    return pairs


def test_promotion_requires_true_route_gain_and_memory_caused_decision_change():
    entry=memory_entry()
    approved=verify_promotion([entry],dev_pairs(),actor_sha=ACTOR,stock_sha=STOCK)
    assert approved["verified"] and approved["wins"]==1 and approved["losses"]==0
    with pytest.raises(ValueError,match="NO_VERIFIED"):
        verify_promotion([entry],dev_pairs(gain=False),actor_sha=ACTOR,stock_sha=STOCK)
    with pytest.raises(ValueError,match="NO_VERIFIED"):
        verify_promotion([entry],dev_pairs(changed=0),actor_sha=ACTOR,stock_sha=STOCK)
    with pytest.raises(ValueError,match="INSUFFICIENT"):
        verify_promotion([entry],dev_pairs(n=7),actor_sha=ACTOR,stock_sha=STOCK)
    bad=dev_pairs()
    bad[0]["id"]="train-01"
    with pytest.raises(ValueError,match="LEAKED"):
        verify_promotion([entry],bad,actor_sha=ACTOR,stock_sha=STOCK)


def test_only_promoted_verified_memory_can_enter_eval(tmp_path):
    item=memory_entry()
    src=tmp_path/"candidate.json"
    durable_json(src,{"schema":"mechet_rsi_candidate_v1","entries":[item]})
    with pytest.raises(ValueError,match="UNVERIFIED"):
        load_memory(src,actor_sha=ACTOR,stock_sha=STOCK)
    raw=tmp_path/"dev.jsonl"
    raw.write_text("".join(json.dumps(p)+"\n" for p in dev_pairs()))
    dest=tmp_path/"promoted.json"
    promoted=promote(src,raw,dest,actor_sha=ACTOR,stock_sha=STOCK)
    assert promoted["verified"]
    accepted=load_memory(dest,actor_sha=ACTOR,stock_sha=STOCK)
    assert accepted.entries[0]["id"]==item["id"]
    with pytest.raises(ValueError,match="PROVENANCE"):
        load_memory(dest,actor_sha=ACTOR,stock_sha="other")


def test_rsi_changes_actual_frontier_not_only_explanation():
    def ctx(key,score):
        return SimpleNamespace(
            key=key, node=SimpleNamespace(terminal=False,
                           policy_score=score,actions=[],state="[CH3:1]"),
            stats=SimpleNamespace(needs_more_sampling=True,unseen_upper=1.0),
            resource_closed="")
    a,b=ctx("high-policy",0.0),ctx("memory-transfer",-0.2)
    tree_a=SimpleNamespace(contexts={a.key:a},work={"node_expansions":0})
    tree_b=SimpleNamespace(contexts={b.key:b},work={"node_expansions":0})
    net=SimpleNamespace(trees={"C":tree_a,"CCO":tree_b},
                        owners={a.key:"C",b.key:"CCO"},
                        next_context=lambda:a)
    agent=AgentSession.__new__(AgentSession)
    agent.network=net
    agent.memory=ExperienceMemory([memory_entry()],actor_sha=ACTOR,stock_sha=STOCK)
    agent.mode="agent_rsi"
    chosen,baseline,memory_id=agent._choose()
    assert baseline.key=="high-policy"
    assert chosen.key=="memory-transfer"
    assert memory_id==agent.memory.entries[0]["id"]
    agent.mode="agent"
    chosen,baseline,none=agent._choose()
    assert chosen.key==baseline.key=="high-policy"
    assert none is None


def test_real_executor_route_closure_from_native_import_flow_finish(tmp_path):
    """CPU full-route integration: stub predictor only, REAL chemistry executor."""
    from test_pr96_history import state_rows
    from test_main_product_training_data import legacy_sn2
    from scripts.build_main_product_training_data import prepare_reference
    from mechet.dynamic_electron_tree import chemical_key
    from mechet.endpoints import split_precursor_endpoints

    source=prepare_reference(legacy_sn2())
    decisions=[r["messages"][2]["tool_calls"][0]["function"] for r in state_rows()]
    assert [x["name"] for x in decisions]==[
        "import_fragments","apply_electron_flow","finish_trace"]
    precursors=split_precursor_endpoints(
        source["full_precursor_state"],source["target_smiles"]).structural
    stock={chemical_key(precursors)}
    assert stock
    class ReplayActor:
        def __init__(self):
            self.index=0
        def generate(self,messages,*,max_new_tokens,seed):
            if self.index>=len(decisions):
                raise AssertionError("actor ran beyond expected program")
            call=decisions[self.index]
            self.index+=1
            return {"raw":"<tool_call>"+json.dumps(call)+"</tool_call>",
                    "generated_token_ids":[42],
                    "truncated":False,"logprob":-1.0}
    actor=ReplayActor()
    session=AgentSession(case_id="cpu-executor-integration",
                         product=source["target_smiles"],stock=stock,
                         stock_sha=STOCK,actor_sha=ACTOR,actor=actor,
                         memory=ExperienceMemory(actor_sha=ACTOR,stock_sha=STOCK),
                         output=tmp_path,mode="independent",
                         max_actions=4,max_tokens=1000,max_new_tokens=128,seed=17)
    result=session.run()
    assert result["stock_solved"] is True
    assert result["model_actions"]==3
    assert len(result["complete_reaction_edges"])==1
    assert result["complete_reaction_edges"][0]["program"]["actions"][-1]["name"]=="finish_trace"
    assert (tmp_path/"report.json").is_file()
