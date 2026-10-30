"""Evidence-gated inference-time RSI over native MechET reverse-electron trees."""
from __future__ import annotations
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import TypedDict
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .dynamic_electron_tree import Proposal
from .pr96_evaluation import messages_for, parsed
from .reverse_route_network import RouteNetwork
from .shared_electron_search import SearchJournal, durable_json, read_record

def checksum(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

@lru_cache(maxsize=20000)
def fingerprint(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError("INVALID_EXPERIENCE_SMILES")
    return rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(mol)

def electron_signature(actions, terminal_state):
    """Map-invariant bond delta signature of actual executed FLOW steps."""
    def bonds(state):
        mol=Chem.MolFromSmiles(state)
        if mol is None:return {}
        atoms={a.GetAtomMapNum():a.GetSymbol() for a in mol.GetAtoms()}
        if not atoms or any(not k for k in atoms):return {}
        return {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum()))):
                (float(b.GetBondTypeAsDouble()),"-".join(sorted((
                   b.GetBeginAtom().GetSymbol(),b.GetEndAtom().GetSymbol()))))
                for b in mol.GetBonds()}
    signatures=[]
    for i,action in enumerate(actions):
        if action.get("name")!="apply_electron_flow":continue
        a=bonds(action["state_before"])
        later=actions[i+1]["state_before"] if i+1<len(actions) else terminal_state
        b=bonds(later)
        delta=[]
        for key in sorted(a.keys()|b.keys()):
            old=a.get(key,(0.0,b.get(key,(0.0,""))[1]))
            new=b.get(key,(0.0,old[1]))
            if abs(new[0]-old[0])>0.01:
                delta.append(f"{new[1]}:{new[0]-old[0]:+g}")
        signatures.append("|".join(sorted(delta)))
    return signatures

class ExperienceMemory:
    """Transferable strategy priors, never stored reactants or tool replay."""
    def __init__(self,entries=(),*,actor_sha,stock_sha,kind="promoted"):
        self.entries=list(entries)
        self.kind=kind
        for e in self.entries:
            if (e["actor_sha"]!=actor_sha or e["stock_sha"]!=stock_sha
                or e["source_split"]!="TRAIN" or not e["route_solved"]):
                raise ValueError("EXPERIENCE_PROVENANCE_MISMATCH")
    def priority(self,molecule,actions,current_state):
        query=fingerprint(molecule)
        prefix=electron_signature(actions,current_state)
        best=(0.0,None)
        for e in self.entries:
            sim=DataStructs.TanimotoSimilarity(query,fingerprint(e["product"]))
            if sim<0.45:continue
            learned=e["flow_actions"]
            matched=sum(x==y for x,y in zip(prefix,learned))
            if prefix and learned and not matched:sim*=0.75
            benefit=1.75*sim+0.08*matched
            if benefit>best[0]:best=(benefit,e["id"])
        return best

def harvest_candidate(report,*,actor_sha,stock_sha,source_split="TRAIN"):
    if source_split!="TRAIN":raise ValueError("TRAIN_ONLY_EXPERIENCE")
    if not report.get("stock_solved") or not report.get("complete_reaction_edges"):return []
    out=[]
    for edge in report["complete_reaction_edges"]:
        p=edge["program"]
        if not any(a.get("name")=="apply_electron_flow" for a in p["actions"]):continue
        item={"actor_sha":actor_sha,"stock_sha":stock_sha,"product":edge["product"],
              "source_split":"TRAIN","route_solved":True,
              "flow_actions":electron_signature(p["actions"],p["full_terminal_state"]),
              "evidence":{"case_id":report["id"],"route_key":report["route_key"],
                          "edge_id":edge["id"],"program_sha256":checksum(p["actions"])}}
        item["id"]=checksum(item)
        out.append(item)
    return out

def verify_promotion(entries,dev_pairs,*,actor_sha,stock_sha,min_pairs=8):
    if not entries:raise ValueError("EMPTY_CANDIDATE_MEMORY")
    mem=ExperienceMemory(entries,actor_sha=actor_sha,stock_sha=stock_sha,kind="candidate")
    if len(dev_pairs)<min_pairs:raise ValueError("INSUFFICIENT_DEV_PAIRS")
    seen=set()
    sources={e["evidence"]["case_id"] for e in mem.entries}
    wins=losses=changes=0
    for pair in dev_pairs:
        tid=pair["id"]
        if tid in seen or tid in sources or pair["split"]!="DEV":
            raise ValueError("LEAKED_OR_DUPLICATED_DEV_ID")
        if pair["actor_sha"]!=actor_sha or pair["stock_sha"]!=stock_sha:
            raise ValueError("DEV_ACTOR_STOCK_MISMATCH")
        a,b=pair["baseline"],pair["candidate"]
        if a["budget"]!=b["budget"] or a.get("seed")!=b.get("seed"):
            raise ValueError("DEV_BUDGET_SEED_UNMATCHED")
        if any(x.get("real_executor") is not True or type(x.get("route_solved")) is not bool
               for x in (a,b)):raise ValueError("NOT_REAL_EXECUTOR_DEV_PAIR")
        wins+=int(b["route_solved"] and not a["route_solved"])
        losses+=int(a["route_solved"] and not b["route_solved"])
        changes+=int(b.get("memory_decision_changes",0))
        seen.add(tid)
    if wins<=losses or wins==0 or changes==0:
        raise ValueError("NO_VERIFIED_DEV_ROUTE_GAIN")
    return {"verified":True,"n_dev":len(seen),"wins":wins,"losses":losses,
            "memory_decision_changes":changes,"source":"real_paired_route_dev"}

def promote(candidate_path,dev_pairs_path,output,*,actor_sha,stock_sha,min_pairs=8):
    record=read_record(candidate_path)
    if record.get("schema")!="mechet_rsi_candidate_v1":
        raise ValueError("INVALID_CANDIDATE_SCHEMA")
    pairs=[json.loads(s) for s in Path(dev_pairs_path).read_text().splitlines() if s.strip()]
    result=verify_promotion(record["entries"],pairs,actor_sha=actor_sha,
                            stock_sha=stock_sha,min_pairs=min_pairs)
    durable_json(output,{"schema":"mechet_rsi_promoted_v1","entries":record["entries"],
                        "verification":result,"candidate_sha256":checksum(record),
                        "dev_pairs_sha256":hashlib.sha256(Path(dev_pairs_path).read_bytes()).hexdigest()})
    return result

def load_memory(path,*,actor_sha,stock_sha,allow_candidate=False):
    if path is None:return ExperienceMemory(actor_sha=actor_sha,stock_sha=stock_sha)
    obj=read_record(path)
    expected="mechet_rsi_candidate_v1" if allow_candidate else "mechet_rsi_promoted_v1"
    if obj.get("schema")!=expected or (not allow_candidate and not obj.get("verification",{}).get("verified")):
        raise ValueError("UNVERIFIED_MEMORY_REJECTED")
    return ExperienceMemory(obj["entries"],actor_sha=actor_sha,stock_sha=stock_sha,
                            kind="candidate" if allow_candidate else "promoted")

class GraphState(TypedDict):
    actions:int
    tokens:int
    done:bool
    stop_reason:str

class AgentSession:
    """LangGraph controls search. The native executor alone produces precursors."""
    def __init__(self,*,case_id,product,stock,stock_sha,actor_sha,actor,memory,output,
                 mode="agent",max_actions=128,max_tokens=12000,max_new_tokens=512,seed=17,resume=False):
        if mode not in {"independent","tree","agent","agent_rsi"}:raise ValueError("BAD_MODE")
        if mode=="agent_rsi" and not memory.entries:raise ValueError("VERIFIED_MEMORY_REQUIRED")
        self.case_id,self.actor,self.memory,self.mode=case_id,actor,memory,mode
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.seed,self.max_actions,self.max_tokens=seed,max_actions,max_tokens
        self.max_new_tokens=max_new_tokens
        contract={"actor_sha":actor_sha,"stock_sha":stock_sha,"mode":mode,
                  "seed":seed,"max_actions":max_actions,"max_tokens":max_tokens,
                  "max_new_tokens":max_new_tokens,"memory_ids":sorted(e["id"] for e in memory.entries)}
        self.journal=SearchJournal(self.output/"journal",case_id,contract)
        if self.journal.sequence:
            if not resume:raise ValueError("JOURNAL_EXISTS_USE_RESUME")
            self.network=self.journal.restore(stock=stock)
            from .dynamic_electron_tree import chemical_key
            if self.network.target!=chemical_key(product):raise ValueError("RESTORE_TARGET_MISMATCH")
        else:
            self.network=RouteNetwork(case_id,product,stock,
                                      mode="greedy" if mode=="independent" else "tree")
            self.journal.initialize(self.network)
        self.pending=None
        self.memory_decision_changes=0

    def _choose(self):
        net=self.network
        baseline=net.next_context()
        if baseline is None or self.mode in ("independent","tree"):
            return baseline,baseline,None
        valid=[ctx for tree in net.trees.values() for ctx in tree.contexts.values()
               if not ctx.node.terminal and not ctx.resource_closed and ctx.stats.needs_more_sampling]
        if not valid:return None,baseline,None
        options=[]
        for ctx in valid:
            molecule=net.owners[ctx.key]
            base=(ctx.node.policy_score+math.log(max(1e-9,ctx.stats.unseen_upper))
                  -math.log1p(net.trees[molecule].work["node_expansions"]))
            complexity=0.4/(1+Chem.MolFromSmiles(molecule).GetNumHeavyAtoms())
            bonus,eid=(self.memory.priority(molecule,ctx.node.actions,ctx.node.state)
                       if self.mode=="agent_rsi" else (0.0,None))
            options.append((base+complexity+bonus,ctx.key,ctx,eid))
        chosen=max(options,key=lambda x:(x[0],x[1]))
        return chosen[2],baseline,chosen[3]

    def _select(self,state:GraphState):
        if self.network.target in self.network._closure() and not self.network.already_in_stock:
            return {**state,"done":True,"stop_reason":"STOCK_SOLVED"}
        if self.network.already_in_stock:
            return {**state,"done":True,"stop_reason":"ALREADY_IN_STOCK"}
        if state["actions"]>=self.max_actions or state["tokens"]>=self.max_tokens:
            return {**state,"done":True,"stop_reason":"BUDGET_EXHAUSTED"}
        choice,baseline,eid=self._choose()
        if choice is None:return {**state,"done":True,"stop_reason":"FRONTIER_EXHAUSTED"}
        self.pending=(choice,baseline.key if baseline else None,eid)
        return {**state,"done":False}

    def _act(self,state:GraphState):
        context,without,eid=self.pending
        changed=without is not None and context.key!=without
        self.memory_decision_changes+=int(changed and self.mode=="agent_rsi")
        room=self.max_tokens-state["tokens"]
        seed=int(checksum([self.seed,self.case_id,context.key,
                           context.stats.draws,state["actions"]])[:8],16)
        messages=messages_for(context.node,True)
        output=self.actor.generate(messages,max_new_tokens=min(self.max_new_tokens,room),seed=seed)
        name,args,error=parsed(output)
        p=Proposal(name=name,arguments=args,raw=output.get("raw",""),
                   logprob=float(output.get("logprob",0.0)),
                   tokens=len(output.get("generated_token_ids",[])),
                   error=error,truncated=bool(output.get("truncated")),
                   token_ids=output.get("generated_token_ids",[]))
        edges=self.network.incorporate(context.key,[p])
        self.journal.record(self.network,context.key,edges)
        event={"id":self.case_id,"decision":state["actions"],"context":context.key,
               "baseline_context":without,"memory_id":eid,
               "memory_changed_decision":bool(changed and self.mode=="agent_rsi"),
               "messages":messages,"model_output":output,
               "predicted_name":name,"arguments":args,"parse_error":error,
               "executor_edges":edges,"state_before":context.node.state,
               "stock_solved":self.network.target in self.network._closure()}
        with (self.output/"calls.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(event,ensure_ascii=False)+"\n")
            handle.flush()
        return {**state,"actions":state["actions"]+1,"tokens":state["tokens"]+p.tokens}

    def run(self):
        from langgraph.graph import END,START,StateGraph
        graph=StateGraph(GraphState)
        graph.add_node("choose",self._select)
        graph.add_node("electron_tool",self._act)
        graph.add_node("finish",lambda s:s)
        graph.add_edge(START,"choose")
        graph.add_conditional_edges("choose",lambda s:"finish" if s["done"] else "electron_tool")
        graph.add_edge("electron_tool","choose")
        graph.add_edge("finish",END)
        stats=self.network.work
        initial={"actions":int(stats["raw_candidates"]),"tokens":int(stats["sampled_tokens"]),
                 "done":False,"stop_reason":""}
        state=graph.compile().invoke(initial,config={"recursion_limit":2*self.max_actions+10})
        self.network.stop_reason=state["stop_reason"]
        report=self.network.report()
        report.update({"agent_mode":self.mode,"agent_actor_frozen":True,
                       "rsi_memory_kind":self.memory.kind if self.mode=="agent_rsi" else "none",
                       "memory_decision_changes":self.memory_decision_changes,
                       "model_actions":state["actions"],"generated_tokens":state["tokens"],
                       "real_executor":True,
                       "budget":{"max_actions":self.max_actions,"max_tokens":self.max_tokens}})
        durable_json(self.output/"report.json",report)
        return report
