#!/usr/bin/env python3
"""Evaluate frozen FlowER Qwen3.5-0.8B with executable electronic-event search.

R1: external stock-closed multistep routes (all four frozen cohorts).
R2: chemistry/route-depth evidence on SAME R1 outputs; do not resample.
Simple retro: one-step executable precursor candidates and source-exact scores.
None of these protocols uses an endpoint reference to guide model inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.agentic_hf import FrozenStageIIActor, STAGE2
from mechet.dynamic_electron_tree import Proposal, chemical_key
from mechet.frozen_route_assets import file_sha256
from mechet.pr96_evaluation import messages_for, parsed
from mechet.reverse_route_network import RouteNetwork
from mechet.shared_electron_search import durable_json, read_record, SearchJournal
from mechet.unified_electron_search import (
    UnifiedRouteSearch, OneStepElectronicSearch, deterministic_seed, proposal_from_actor,
)


COHORTS = ("ursa_expert100", "ursa_drugs100",
           "retrocast_mkt_cnv160", "paroutes_derived_n5_100")
PAPER_TARGET_COUNTS = dict(zip(COHORTS, (100,100,160,100)))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def frozen_assets(folder: Path) -> tuple[dict,list[dict]]:
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest.get("schema_version")!="mechet_frozen_route_assets_v1":
        raise ValueError("UNKNOWN_ROUTE_ASSET_SCHEMA")
    if not manifest.get("stock_preparation_complete"):
        raise ValueError("STOCK_REQUIRED_FOR_ROUTE_SOLVED")
    if manifest.get("full_target_rows")!=460:
        raise ValueError("R1_R2_EXPECT_460_FROZEN_TARGETS")
    for name,n in PAPER_TARGET_COUNTS.items():
        if manifest.get("cohorts",{}).get(name,{}).get("rows")!=n:
            raise ValueError("FROZEN_COHORT_COUNT_MISMATCH:"+name)
    target_path=Path(manifest["full_targets_path"])
    if file_sha256(target_path)!=manifest["full_targets_sha256"]:
        raise ValueError("FROZEN_TARGET_HASH_MISMATCH")
    rows=list(read_jsonl(target_path))
    if len(rows)!=460 or len({r["id"] for r in rows})!=460:
        raise ValueError("MISSING_OR_DUPLICATE_FROZEN_TARGETS")
    for name,item in manifest["stocks"].items():
        if file_sha256(Path(item["canonical_path"]))!=item["canonical_sha256"]:
            raise ValueError("FROZEN_STOCK_HASH_MISMATCH:"+name)
    for name,metadata in manifest["cohorts"].items():
        if set(metadata["target_ids"])!={r["id"] for r in rows if r["cohort"]==name}:
            raise ValueError("FROZEN_COHORT_IDS_CHANGED:"+name)
    return manifest,rows


def choose_pilot(rows:list[dict], seed:int)->list[dict]:
    chosen=[]
    for cohort in COHORTS:
        pool=[r for r in rows if r["cohort"]==cohort]
        pool.sort(key=lambda r:(hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest(),r["id"]))
        chosen.extend(pool[:4])
    if len(chosen)!=16 or len({x["id"] for x in chosen})!=16:
        raise ValueError("PILOT16_SELECTION_INVALID")
    return chosen


def run_legacy_route(*, case_id, product, stock, actor, actor_sha, stock_sha, out,
                     method, seed, max_calls, max_tokens, max_new_tokens,
                     stop_on_first, resume):
    """Matched historical greedy/heuristic baseline; NOT rebranded as MCTS."""
    if method not in ("greedy","heuristic"):
        raise ValueError("UNKNOWN_LEGACY_SEARCH")
    out.mkdir(parents=True,exist_ok=True)
    contract={"actor_sha":actor_sha,"stock_sha":stock_sha,
              "method":method,"seed":seed,"max_calls":max_calls,
              "max_tokens":max_tokens,"max_new_tokens":max_new_tokens}
    journal=SearchJournal(out/"journal",case_id,contract)
    if journal.sequence:
        if not resume: raise ValueError("LEGACY_JOURNAL_EXISTS_USE_RESUME")
        net=journal.restore(stock=stock)
        if net.mode!=("greedy" if method=="greedy" else "tree") or net.target!=chemical_key(product):
            raise ValueError("LEGACY_ROUTE_RESUME_MISMATCH")
    else:
        net=RouteNetwork(case_id,product,stock,mode="greedy" if method=="greedy" else "tree")
        journal.initialize(net)
    work=net.work
    calls,tokens=int(work["raw_candidates"]),int(work["sampled_tokens"])
    reason="BUDGET_EXHAUSTED"
    while calls<max_calls and tokens<max_tokens:
        if net.already_in_stock:
            reason="ALREADY_IN_STOCK";break
        ctx=net.next_context()
        if ctx is None:
            reason="FRONTIER_EXHAUSTED";break
        room=min(max_new_tokens,max_tokens-tokens)
        p,_=proposal_from_actor(actor,ctx,budget=room,
                                seed=deterministic_seed(seed,case_id,ctx.key,ctx.stats.draws))
        edges=net.incorporate(ctx.key,[p])
        journal.record(net,ctx.key,edges)
        calls+=1;tokens+=p.tokens
        if stop_on_first and net._certificate() is not None:
            reason="STOCK_SOLVED";break
    net.stop_reason=reason
    report=net.report()
    report.update(search_kind="legacy_"+method,
                  boundary_semantics="FINISH_IS_MARKER",
                  actor_sha=actor_sha,real_executor=True,
                  model_calls=calls,generated_tokens=tokens,
                  budget={"max_calls":max_calls,"max_tokens":max_tokens})
    durable_json(out/"report.json",report)
    return report


def run_routes(args):
    folder=Path(args.assets)
    manifest,all_rows=frozen_assets(folder)
    rows=choose_pilot(all_rows,args.seed) if args.pilot16 else all_rows
    if args.cohort:
        rows=[r for r in rows if r["cohort"] in args.cohort]
    out=Path(args.output)/"routes"/args.method
    # Evaluate the same frozen FlowER actor for every method and cohort.
    actor=FrozenStageIIActor("flower",device=args.device,log_path=out/"actor_calls.jsonl")
    if actor.actor_sha!=STAGE2["flower"]["sha256"]:
        raise ValueError("FLOWER_STAGE_II_PIN_FAILED")
    reports=[]
    for row in rows:
        case_id=row["id"]
        folder=out/row["cohort"]/hashlib.sha256(case_id.encode()).hexdigest()[:20]
        stock_spec=manifest["stocks"][row["stock"]]
        stock=set(Path(stock_spec["canonical_path"]).read_text().splitlines())
        if args.method=="event_puct":
            controller=UnifiedRouteSearch(case_id=case_id,product=row["product_unmapped"],
                stock=stock,stock_sha=stock_spec["canonical_sha256"],
                actor=actor,actor_sha=actor.actor_sha,output=folder,
                max_calls=args.max_calls,max_tokens=args.max_tokens,
                max_new_tokens=args.max_new_tokens,seed=args.seed,resume=args.resume,
                c_puct=args.c_puct,widen_k=args.widen_k,widen_alpha=args.widen_alpha)
            report=controller.run(stop_on_first=args.stop_on_first)
        else:
            report=run_legacy_route(case_id=case_id,product=row["product_unmapped"],
                stock=stock,stock_sha=stock_spec["canonical_sha256"],
                actor=actor,actor_sha=actor.actor_sha,out=folder,method=args.method,
                max_calls=args.max_calls,max_tokens=args.max_tokens,
                max_new_tokens=args.max_new_tokens,seed=args.seed,resume=args.resume,
                stop_on_first=args.stop_on_first)
        reports.append({"id":case_id,"cohort":row["cohort"],
            "solved":bool(report["stock_solved"]),"already_in_stock":bool(report["already_in_stock"]),
            "first_solution":report.get("first_solution"),
            "work":report["work"],"route_key":report.get("route_key"),
            "reference_depth":row.get("reference_depth"),
            "composition_stratum":row.get("composition_stratum","Unknown"),
            "independent_chemistry_verified":False})
        print(json.dumps({"id":case_id,"solved":report["stock_solved"],
                          "work":report["work"]},ensure_ascii=False),flush=True)
    summary=aggregate_routes(reports,expected=len(rows))
    summary.update(actor="flower_stage2_qwen35_0.8b",actor_sha=actor.actor_sha,
                   method=args.method,selected_ids=[r["id"] for r in rows],
                   protocol="R1_AND_stock_R2_same_targets_no_new_inference",
                   benchmark="pilot16" if args.pilot16 else "frozen460")
    durable_json(out/"summary.json",summary)
    with (out/"per_target.jsonl").open("w",encoding="utf-8") as handle:
        for row in reports:handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    return summary


def aggregate_routes(rows, *, expected):
    if len(rows)!=expected:
        raise ValueError("ROUTE_REPORTS_INCOMPLETE")
    cohorts={}
    for name in COHORTS:
        selected=[x for x in rows if x["cohort"]==name]
        if selected:
            cohorts[name]={"count":len(selected),
                "solved_nontrivial":sum(x["solved"] for x in selected),
                "already_stock":sum(x["already_in_stock"] for x in selected),
                "unknown_or_censored":sum(not x["solved"] and not x["already_in_stock"]
                                          for x in selected)}
    # RetroCast reference depths come from source routes; these are NOT model
    # electron-mechanism annotations, so unknown composition remains unknown.
    retrocast=[x for x in rows if x["cohort"]=="retrocast_mkt_cnv160"]
    r2={"population":len(retrocast),
        "reference_depth":{"%s"%k:{"n":sum(x["reference_depth"]==k for x in retrocast),
             "solved":sum(x["reference_depth"]==k and x["solved"] for x in retrocast)}
             for k in (2,3,4,5)},
        "composition_stratum":dict(Counter(x["composition_stratum"] for x in retrocast)),
        "independently_verified_mechanisms":sum(x["independent_chemistry_verified"]
                                                 for x in retrocast),
        "limitation":"Without verified reference electronic programs, C0/C1/C2 remains Unknown."}
    return {"n":len(rows),"solved_nontrivial":sum(x["solved"] for x in rows),
            "already_in_stock":sum(x["already_in_stock"] for x in rows),
            "by_cohort":cohorts,"r2_same_r1_outputs":r2,
            "chemical_feasibility":"not_independently_verified"}


def gold_for_one_step(row:dict, product:str):
    """Only called AFTER generation; reference is never exposed to actor."""
    reference=(row.get("gold_structural_precursors") or row.get("gold_precursors")
               or row.get("structural_precursors"))
    if not reference and row.get("full_precursor_state"):
        from mechet.endpoints import split_precursor_endpoints
        from mechet.in_place_grounded_flow import map_unmapped_fragment
        from mechet.dynamic_electron_tree import product_only_private_state
        reference=split_precursor_endpoints(row["full_precursor_state"],
                                           product_only_private_state(product)).structural
    if reference:
        if isinstance(reference,list):
            reference=".".join(reference)
        return chemical_key(reference)
    return None


def score_one_step(result:dict, gold:str|None) -> dict:
    predictions=[x["precursor"] for x in result["candidates"]]
    return {"id":result["id"],"gold_available":gold is not None,
            "top1_hit":predictions[:1]==[gold] if gold else None,
            "top5_hit":gold in predictions[:5] if gold else None,
            "top10_hit":gold in predictions[:10] if gold else None,
            "model_calls":result["model_calls"],
            "generated_tokens":result["generated_tokens"],
            "unique_precursor_candidates":result["unique_structural_precursors"]}


def run_one_step(args):
    rows=list(read_jsonl(Path(args.one_step_jsonl)))
    if not rows or len({str(row.get("id","")) for row in rows})!=len(rows):
        raise ValueError("ONE_STEP_ROWS_MISSING_OR_DUPLICATE_IDS")
    if any(not str(row.get("id","")) for row in rows):
        raise ValueError("ONE_STEP_MISSING_ID")
    out=Path(args.output)/"simple_retro"
    actor=FrozenStageIIActor("flower",device=args.device,log_path=out/"actor_calls.jsonl")
    summary=[]
    for row in rows:
        product=(row.get("product_unmapped") or row.get("target_smiles")
                 or row.get("product_smiles"))
        if not product:raise ValueError("ONE_STEP_MISSING_PRODUCT")
        case_id=str(row["id"])
        search=OneStepElectronicSearch(case_id=case_id,product=product,actor=actor,
              max_calls=args.max_calls,max_tokens=args.max_tokens,
              max_new_tokens=args.max_new_tokens,seed=args.seed)
        result=search.run()
        gold=gold_for_one_step(row,product)
        scored=score_one_step(result,gold)
        folder=out/hashlib.sha256(case_id.encode()).hexdigest()[:20]
        durable_json(folder/"prediction.json",result)
        summary.append(scored)
        print(json.dumps(scored,ensure_ascii=False),flush=True)
    known=[x for x in summary if x["gold_available"]]
    totals={"n":len(summary),"n_scored":len(known),"source_split":args.split,
        "actor":"flower_stage2_qwen35_0.8b","actor_sha":actor.actor_sha,
        "top1_nll_surrogate_hit_count":sum(x["top1_hit"] for x in known),
        "top5_hit_count":sum(x["top5_hit"] for x in known),
        "top10_hit_count":sum(x["top10_hit"] for x in known),
        "total_generated_tokens":sum(x["generated_tokens"] for x in summary),
        "note":"TopK uses normalized action-logprob candidate ranking, NOT original PR96 NLL ranking."}
    durable_json(out/"summary.json",totals)
    with (out/"per_target.jsonl").open("w",encoding="utf-8") as handle:
        for row in summary:handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",choices=("routes","simple_retro"),required=True)
    p.add_argument("--assets",type=Path,help="Source-verified frozen460 assets (routes)")
    p.add_argument("--one-step-jsonl",type=Path,help="One-step product-only input rows")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--method",choices=("event_puct","heuristic","greedy"),default="event_puct")
    p.add_argument("--cohort",choices=COHORTS,action="append")
    p.add_argument("--pilot16",action="store_true")
    p.add_argument("--split",choices=("VALID","TEST"),default="VALID")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-calls",type=int,default=128)
    p.add_argument("--max-tokens",type=int,default=12000)
    p.add_argument("--max-new-tokens",type=int,default=512)
    p.add_argument("--c-puct",type=float,default=1.25)
    p.add_argument("--widen-k",type=float,default=1.5)
    p.add_argument("--widen-alpha",type=float,default=0.5)
    p.add_argument("--stop-on-first",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.suite=="routes":
        if not args.assets or args.one_step_jsonl: p.error("routes needs --assets and forbids --one-step-jsonl")
        output=run_routes(args)
    else:
        if not args.one_step_jsonl or args.assets or args.cohort:
            p.error("simple_retro needs --one-step-jsonl and forbids route options")
        output=run_one_step(args)
    print(json.dumps(output,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
