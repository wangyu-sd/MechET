"""Account for exact precursor obligations in every verified C0 development attempt."""
from argparse import Namespace
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys

from scripts.pr117_controllers import C0
from scripts.run_pr117_reference import bind_runtime, write_new


def rejection(error):
    if error.startswith("PR117_REQUEST_BUDGET"):
        return "request_cap"
    lower = error.lower()
    for token, name in (("route_cycle", "route_ancestor_cycle"), ("state_cycle", "electronic_state_cycle"),
                        ("import_fragment_invalid", "invalid_import"), ("chemical_state_invalid", "invalid_electronic_state"),
                        ("nonproductive_finish", "nonproductive_finish")):
        if token in lower:
            return name
    if "bond" in lower:
        if "destination" in lower:
            return "destination_bond"
        if "source" in lower:
            return "source_bond"
    raise ValueError("Unclassified terminal executor reason")


def obligations(target, reactions, stock):
    open_nodes = {"0": (target, ())}
    for event in reactions:
        product, ancestors = open_nodes.pop(event["occurrence"])
        if product != event["product"] or not event["precursors"]:
            raise ValueError("Committed attempt has inconsistent structural obligation")
        for i, child in enumerate(event["precursors"]):
            if child in (*ancestors, product):
                raise ValueError("Committed attempt contains an ancestor cycle")
            key = f"{event['occurrence']}.{i}"
            open_nodes[key] = (child, (*ancestors, product))
    return Counter(v[0] for v in open_nodes.values() if v[0] in stock), Counter(v[0] for v in open_nodes.values() if v[0] not in stock)


def run(root, runtime):
    from rdkit import Chem

    panel_path = root / "reference_panel.json"
    _, panel, stocks, verified = bind_runtime(Namespace(runtime_contract=runtime, panel=panel_path))
    audited_path = root / "stage_a_final_verified/receipt_audit.json"
    sealed_path = root / "stage_a_final_verified/complete.json"
    seal = json.loads(sealed_path.read_bytes())
    audit_raw = audited_path.read_bytes()
    audit = json.loads(audit_raw)
    assert sha256(audit_raw).hexdigest() == seal["analysis_sha256"]["receipt_audit.json"]
    assert audit["fully_audited"] and audit["final_summary_verified"] and audit["binding"] == verified
    by_id = {(a["cohort"], a["target_id"]): a for a in audit["rows"] if a["arm"] == C0}
    assert len(by_id) == len(panel["targets"]) == 24
    source_dir = root / "launch_1a100_hf_02/results/targets"
    by_cohort, all_attempts, selected = {}, [], []
    for case in panel["targets"]:
        key = case["cohort"], case["target_id"]
        p = source_dir / f"{key[0]}-{key[1]}-{C0}.json"
        raw = p.read_bytes()
        source = json.loads(raw)
        verified_row = by_id[key]
        assert sha256(raw).hexdigest() == verified_row["source_sha256"]
        assert source["target"] == case["target"] and source["stock"] == case["stock"]
        stock = stocks[case["stock"]]
        episodes = source["result"]["independent_attempts"]
        expected_calls = [x["request_id"] for x in source["result"]["decision_receipts"] if x["generated_token_ids"]]
        actual_calls = [x["request_id"] for report in episodes for x in report["records"] if x["generated_token_ids"]]
        assert expected_calls == actual_calls and len(actual_calls) == verified_row["gpu_requests"]
        attempts = []
        for num, ep in enumerate(episodes, 1):
            in_stock, unresolved = obligations(case["target"], ep["reactions"], stock)
            assert not Counter(ep["stock_leaves"]) - in_stock
            heavy = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in unresolved]
            attempts.append(dict(cohort=key[0],target_id=key[1],source_sha256=verified_row["source_sha256"],
                attempt=num,native_calls=sum(bool(x["generated_token_ids"]) for x in ep["records"]),
                committed_reactions=len(ep["reactions"]),maximum_depth=max((x["depth"] for x in ep["reactions"]),default=0),
                terminal_status=ep["status"],cause=rejection(ep["error"]),stock_leaf_occurrences=in_stock.total(),
                unresolved_occurrences=unresolved.total(),distinct_unresolved_structures=len(unresolved),
                largest_unresolved_heavy_atoms=max(heavy,default=None)))
        all_attempts.extend(attempts)
        selected.append(max(attempts,key=lambda t:t["committed_reactions"]))
    assert sum(a["native_calls"] for a in all_attempts)==sum(a["gpu_requests"] for a in by_id.values())==720
    for cohort in sorted({row["cohort"] for row in selected}):
        rows=[row for row in selected if row["cohort"]==cohort]
        by_cohort[cohort]=dict(targets=len(rows),two_reactions=sum(a["committed_reactions"]>=2 for a in rows),
                               median_unresolved=statistics.median(a["unresolved_occurrences"] for a in rows))
    return dict(scope="Complete archived independent-route development condition. No new inference.",
                targets=len(by_id),attempt_count=len(all_attempts),native_calls=720,
                attempt_stop_causes=dict(Counter(row["cause"] for row in all_attempts)),
                rejections_after_committed_reaction=sum(row["committed_reactions"]>0 and row["cause"]!="request_cap" for row in all_attempts),
                request_cap_attempts_with_unresolved_structural_leaf=sum(row["cause"]=="request_cap" and row["unresolved_occurrences"]>0 for row in all_attempts),
                candidate_complete_structural_routes=sum(row["unresolved_occurrences"]==0 for row in all_attempts),
                deepest_attempt_selection="max committed reactions, first in order on ties",
                deepest_attempt_targets_achieving_stock_leaf=sum(row["stock_leaf_occurrences"]>0 for row in selected),
                deepest_attempt_targets_with_one_unresolved=sum(row["unresolved_occurrences"]==1 for row in selected),
                deepest_attempt_median_unresolved=statistics.median(row["unresolved_occurrences"] for row in selected),
                deepest_attempt_maximum_depth=max(row["maximum_depth"] for row in selected),by_cohort=by_cohort,
                panel_sha256=verified["panel_sha256"],actor_sha256=verified["actor_sha256"],
                receipt_audit_sha256=sha256(audit_raw).hexdigest(),rows=all_attempts,selected_rows=selected)


if __name__ == "__main__":
    root,runtime=map(Path,sys.argv[1:])
    output=root/"reference_c0_terminal_frontiers_20261122.json"
    data=run(root,runtime)
    write_new(output,data)
    print(json.dumps({k:v for k,v in data.items() if k not in ("rows","selected_rows")}))
