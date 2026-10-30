#!/usr/bin/env python3
"""Run real frozen Stage-II electron-tool routes; compile paired DEV RSI evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

from mechet.agentic_hf import STAGE2,file_sha256
from mechet.agentic_rsi import AgentSession,checksum,harvest_candidate,load_memory,promote
from mechet.shared_electron_search import durable_json,read_record

COHORTS=("ursa_expert100","ursa_drugs100","retrocast_mkt_cnv160",
         "paroutes_derived_n5_100")

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def load_assets(root):
    manifest=json.loads((Path(root)/"manifest.json").read_text())
    if not manifest.get("stock_preparation_complete") or manifest["full_target_rows"]!=460:
        raise ValueError("FROZEN_ASSETS_INCOMPLETE")
    source=Path(manifest["full_targets_path"])
    if file_sha256(source)!=manifest["full_targets_sha256"]:
        raise ValueError("FROZEN_TARGET_SHA_MISMATCH")
    rows=list(read_jsonl(source))
    if len(rows)!=460 or len({r["id"] for r in rows})!=460:
        raise ValueError("INVALID_FROZEN_460")
    for item in manifest["stocks"].values():
        if file_sha256(item["canonical_path"])!=item["canonical_sha256"]:
            raise ValueError("STOCK_HASH_MISMATCH")
    return manifest,rows

def freeze(assets,selection,seed=17):
    manifest,rows=load_assets(assets)
    picked={}
    for cohort in COHORTS:
        pool=[r for r in rows if r["cohort"]==cohort]
        if len(pool)!=manifest["cohorts"][cohort]["rows"]:
            raise ValueError("COHORT_DENOMINATOR_MISMATCH")
        pool.sort(key=lambda r:(hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest(),r["id"]))
        picked[cohort]=[r["id"] for r in pool[:4]]
    record={"schema":"mechet_agentic_pilot_v1","seed":seed,
            "frozen_targets_sha256":manifest["full_targets_sha256"],
            "config_sha256":manifest["config_sha256"],"selection":picked}
    selection=Path(selection)
    if selection.exists():
        if read_record(selection)!=record:raise ValueError("EXISTING_SELECTION_LOCK_DIFFERS")
    else:durable_json(selection,record)
    return record

def pilot(assets,selection):
    manifest,rows=load_assets(assets)
    plan=read_record(selection)
    if plan.get("schema")!="mechet_agentic_pilot_v1":raise ValueError("BAD_SELECTION")
    if freeze(assets,selection,plan["seed"])!=plan:raise ValueError("SELECTION_CHANGED")
    ids={rid for group in plan["selection"].values() for rid in group}
    if len(ids)!=16:raise ValueError("EXPECTED_SIXTEEN_PILOT_TARGETS")
    return manifest,[r for r in rows if r["id"] in ids]

def run_one(row,*,args,actor,memory,stock,stock_sha):
    root=Path(args.output)/args.track/args.mode/checksum(row["id"])[:20]
    root.mkdir(parents=True,exist_ok=True)
    result_path=root/"report.json"
    if result_path.exists():
        if not args.resume:raise ValueError("EXISTING_RESULT_REQUIRES_RESUME")
        result=read_record(result_path)
        if result["id"]!=row["id"] or result["actor_sha"]!=actor.actor_sha:
            raise ValueError("RESUMED_RESULT_BINDING_MISMATCH")
        return result
    actions=tokens=episodes=0
    solved=False
    latest=None
    change_count=0
    while actions<args.max_actions and tokens<args.max_tokens:
        suffix=f"independent_{episodes}" if args.mode=="independent" else "search"
        session=AgentSession(case_id=f"{row['id']}:{suffix}",
                             product=row["product_unmapped"],stock=stock,stock_sha=stock_sha,
                             actor_sha=actor.actor_sha,actor=actor,memory=memory,
                             output=root/suffix,mode=args.mode,
                             max_actions=args.max_actions-actions,max_tokens=args.max_tokens-tokens,
                             max_new_tokens=args.max_new_tokens,seed=args.seed+episodes,
                             resume=args.resume)
        report=session.run()
        actions+=report["model_actions"]
        tokens+=report["generated_tokens"]
        change_count+=report["memory_decision_changes"]
        episodes+=1
        latest=report
        if report["stock_solved"]:
            solved=True
            break
        if args.mode!="independent" or report["model_actions"]==0:break
    if latest is None:raise ValueError("NO_ROUTE_ATTEMPT")
    result={**latest,"id":row["id"],"track":args.track,"split":args.split,
            "actor_sha":actor.actor_sha,"stock_sha":stock_sha,"stock_name":row["stock"],
            "stock_solved":solved,"total_episodes":episodes,
            "model_actions":actions,"generated_tokens":tokens,
            "budget":{"max_actions":args.max_actions,"max_tokens":args.max_tokens},
            "real_executor":True,"seed":args.seed,"memory_decision_changes":change_count}
    durable_json(result_path,result)
    return result

def run(args):
    if args.split=="EVAL":
        if args.targets_jsonl:raise ValueError("EVAL_MUST_USE_FROZEN_SELECTION")
        manifest,rows=pilot(args.assets,args.selection)
    else:
        if not args.targets_jsonl:raise ValueError("TRAIN_DEV_REQUIRE_EXPLICIT_TARGETS")
        manifest,eval_rows=load_assets(args.assets)
        rows=list(read_jsonl(args.targets_jsonl))
        forbidden_ids={r["id"] for r in eval_rows}
        from mechet.dynamic_electron_tree import chemical_key
        forbidden_structures={chemical_key(r["product_unmapped"]) for r in eval_rows}
        if not rows or any(r["cohort"] in COHORTS or r["id"] in forbidden_ids
                           or chemical_key(r["product_unmapped"]) in forbidden_structures
                           for r in rows):
            raise ValueError("TRAIN_DEV_MUST_BE_STRUCTURE_DISJOINT_FROM_EVAL")
        if len({r["id"] for r in rows})!=len(rows):raise ValueError("DUPLICATE_TARGETS")
    if args.allow_candidate and args.split!="DEV":
        raise ValueError("CANDIDATE_MEMORY_DEV_ONLY")
    if args.mode=="agent_rsi" and not args.memory:raise ValueError("RSI_MEMORY_REQUIRED")
    from mechet.agentic_hf import FrozenStageIIActor
    actor=FrozenStageIIActor(args.track,device=args.device,
                             log_path=Path(args.output)/args.track/args.mode/"raw_model_calls.jsonl")
    completed=[]
    for row in rows:
        stock_info=manifest["stocks"].get(row["stock"])
        if stock_info is None:raise ValueError("STOCK_NOT_IN_FROZEN_MANIFEST")
        memory_path=(args.memory / f"{row['stock']}.json"
                     if args.memory is not None and args.memory.is_dir() else args.memory)
        memory=load_memory(memory_path,actor_sha=actor.actor_sha,
                           stock_sha=stock_info["canonical_sha256"],
                           allow_candidate=args.allow_candidate)
        if any(e["evidence"]["case_id"]==row["id"] for e in memory.entries):
            raise ValueError("TARGET_EXPERIENCE_LEAK")
        stock=set(Path(stock_info["canonical_path"]).read_text().splitlines())
        result=run_one(row,args=args,actor=actor,memory=memory,
                       stock=stock,stock_sha=stock_info["canonical_sha256"])
        completed.append(result)
        print(json.dumps({"id":row["id"],"method":args.mode,
                          "stock_solved":result["stock_solved"],
                          "actions":result["model_actions"],
                          "tokens":result["generated_tokens"]}),flush=True)
    summary={"track":args.track,"method":args.mode,"split":args.split,
             "n":len(completed),"solved":sum(r["stock_solved"] for r in completed),
             "already_in_stock":sum(r["already_in_stock"] for r in completed),
             "generated_tokens":sum(r["generated_tokens"] for r in completed),
             "targets":[r["id"] for r in completed]}
    durable_json(Path(args.output)/args.track/args.mode/"summary.json",summary)
    return summary

def harvest(args):
    reports=[read_record(p) for p in sorted(Path(args.reports).glob("*/report.json"))]
    if not reports or any(r.get("split")!="TRAIN" or r.get("real_executor") is not True
                          for r in reports):
        raise ValueError("HARVEST_REQUIRES_REAL_TRAIN_ROUTE_REPORTS")
    if len({r["actor_sha"] for r in reports})!=1 or len({r["stock_sha"] for r in reports})!=1:
        raise ValueError("HARVEST_ONE_ACTOR_AND_ONE_STOCK_PER_BANK")
    entries=[]
    for r in reports:
        entries.extend(harvest_candidate(r,actor_sha=r["actor_sha"],
                                        stock_sha=r["stock_sha"]))
    unique={e["id"]:e for e in entries}
    durable_json(args.candidate,{"schema":"mechet_rsi_candidate_v1",
                                 "entries":list(unique.values())})
    return {"train_reports":len(reports),"candidate_strategies":len(unique)}

def pair_dev(args):
    def load(root):
        records=[read_record(p) for p in Path(root).glob("*/report.json")]
        if len(records)!=len({r["id"] for r in records}):raise ValueError("DEV_DUPLICATE_ID")
        return {r["id"]:r for r in records}
    before,after=load(args.baseline),load(args.candidate_runs)
    if not before or before.keys()!=after.keys():raise ValueError("UNPAIRED_DEV_TARGETS")
    pairs=[]
    for rid in sorted(before):
        a,b=before[rid],after[rid]
        if a["split"]!="DEV" or b["split"]!="DEV":
            raise ValueError("NOT_DEV_RESULTS")
        if any(a[k]!=b[k] for k in ("actor_sha","stock_sha","seed","budget")):
            raise ValueError("UNMATCHED_DEV_CONTRACT")
        def arm(x):
            return {"budget":x["budget"],"seed":x["seed"],
                    "route_solved":x["stock_solved"],"real_executor":x["real_executor"],
                    "model_actions":x["model_actions"],"generated_tokens":x["generated_tokens"],
                    "memory_decision_changes":x["memory_decision_changes"]}
        pairs.append({"id":rid,"split":"DEV","actor_sha":a["actor_sha"],
                      "stock_sha":a["stock_sha"],"baseline":arm(a),"candidate":arm(b)})
    args.pairs.parent.mkdir(parents=True,exist_ok=True)
    if args.pairs.exists():raise ValueError("DEV_PAIR_FILE_ALREADY_EXISTS")
    args.pairs.write_text("".join(json.dumps(p,sort_keys=True)+"\n" for p in pairs))
    return {"n_dev":len(pairs),"sha256":file_sha256(args.pairs)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze")
    f.add_argument("--assets",type=Path,required=True)
    f.add_argument("--selection",type=Path,required=True)
    f.add_argument("--seed",type=int,default=17)
    r=sub.add_parser("run")
    for key in ("assets","output"):
        r.add_argument("--"+key,type=Path,required=True)
    r.add_argument("--selection",type=Path)
    r.add_argument("--targets-jsonl",type=Path)
    r.add_argument("--split",choices=["EVAL","TRAIN","DEV"],default="EVAL")
    r.add_argument("--track",choices=sorted(STAGE2),required=True)
    r.add_argument("--mode",choices=["independent","tree","agent","agent_rsi"],required=True)
    r.add_argument("--memory",type=Path)
    r.add_argument("--allow-candidate",action="store_true")
    r.add_argument("--device",default="cuda:0")
    r.add_argument("--seed",type=int,default=17)
    r.add_argument("--max-actions",type=int,default=128)
    r.add_argument("--max-tokens",type=int,default=12000)
    r.add_argument("--max-new-tokens",type=int,default=512)
    r.add_argument("--resume",action="store_true")
    h=sub.add_parser("harvest")
    h.add_argument("--reports",type=Path,required=True)
    h.add_argument("--candidate",type=Path,required=True)
    d=sub.add_parser("pair-dev")
    d.add_argument("--baseline",type=Path,required=True)
    d.add_argument("--candidate-runs",type=Path,required=True)
    d.add_argument("--pairs",type=Path,required=True)
    v=sub.add_parser("promote")
    v.add_argument("--candidate",type=Path,required=True)
    v.add_argument("--dev-pairs",type=Path,required=True)
    v.add_argument("--promoted",type=Path,required=True)
    v.add_argument("--track",choices=sorted(STAGE2),required=True)
    v.add_argument("--stock-sha",required=True)
    v.add_argument("--min-dev-pairs",type=int,default=8)
    a=p.parse_args()
    if a.command=="freeze":result=freeze(a.assets,a.selection,a.seed)
    elif a.command=="run":result=run(a)
    elif a.command=="harvest":result=harvest(a)
    elif a.command=="pair-dev":result=pair_dev(a)
    else:result=promote(a.candidate,a.dev_pairs,a.promoted,
                        actor_sha=STAGE2[a.track]["sha256"],
                        stock_sha=a.stock_sha,min_pairs=a.min_dev_pairs)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
