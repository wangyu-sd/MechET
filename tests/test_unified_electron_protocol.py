from mechet.unified_electron_search import ElectronicFrontier
from scripts.eval_unified_electron_search import aggregate_routes,score_one_step,choose_pilot


def test_same_route_outputs_define_r2_not_new_inference():
    rows=[{"id":f"X:{i}","cohort":"retrocast_mkt_cnv160","solved":i==0,
           "already_in_stock":False,"reference_depth":2 if i==0 else 3,
           "composition_stratum":"Unknown","independent_chemistry_verified":False}
          for i in range(3)]
    result=aggregate_routes(rows,expected=3)
    assert result["r2_same_r1_outputs"]["population"]==3
    assert result["r2_same_r1_outputs"]["composition_stratum"]=={"Unknown":3}
    assert result["solved_nontrivial"]==1


def test_one_step_gold_is_scored_after_inference():
    out={"id":"x","model_calls":4,"generated_tokens":10,
         "unique_structural_precursors":1,
         "candidates":[{"precursor":"CC","score":-0.2}]}
    assert score_one_step(out,"CC")["top1_hit"]
    assert not score_one_step(out,"CCC")["top1_hit"]
    assert score_one_step(out,None)["top1_hit"] is None
