"""Evaluate genuine CbyD receipts against separate literature and stock endpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.pr117_cbyd import add_inputs, bind_cbyd
from scripts.pr117_confirmation import read_pinned
from scripts.run_pr117_reference import CEPH, sha, write_new


def summarize_scores(rows):
    summary = dict(targets=len(rows), productive_first_reaction=sum(r["has_productive_root_commit"] for r in rows),
                   exact_first_disconnection=sum(r["exact_first_commit"] for r in rows),
                   exact_disconnection_found=sum(r["any_exact_match"] for r in rows),
                   full_inventory_closed=sum(r["stock_closed"] for r in rows))
    if any("first_root_structure" in row for row in rows):
        summary["first_root_structure_counts"] = dict(Counter(row["first_root_structure"] for row in rows))
        summary["root_connectivity_found"] = sum(row["first_connectivity_match_call"] is not None for row in rows)
    return summary


def stereo_free_precursors(precursors):
    from rdkit import Chem

    identities = []
    for smiles in precursors:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Cannot resolve an executor precursor for stereochemistry diagnosis")
        Chem.RemoveStereochemistry(molecule)
        identities.append(Chem.MolToSmiles(molecule, isomericSmiles=True))
    return sorted(identities)


def source_connectivity(evaluator, tree_path):
    from cbyd_first_disconnection import precursor_signature
    from mechet.route_training_contract import chemical_identity

    trees = read_pinned(dict(path=tree_path, sha256=evaluator["source_tree_sha256"]))
    refs = {row["target_id"]: row for row in evaluator["references"]}
    results = {}
    for line in trees.splitlines():
        tree = json.loads(line)
        uid = tree["id"]
        if uid not in refs:
            continue
        edge, = tree["children"]
        precursors = [chemical_identity(child["smiles"]) for child in edge["children"]]
        reference = refs[uid]
        if (uid in results or chemical_identity(tree["target"]) != reference["target"]
                or precursor_signature(precursors) != reference["precursor_signature"]):
            raise ValueError("Stereo analysis requires the exact preregistered source root structures")
        results[uid] = stereo_free_precursors(precursors)
    if results.keys() != refs.keys():
        raise ValueError("Incomplete stereo reference extraction")
    return results


def root_stereo_diagnostic(reference, commits, request_ids, cap, exact):
    positions = {uid: index for index, uid in enumerate(request_ids, 1)}
    matching = [positions[c["receipt"]] for c in commits if c["occurrence"] == "0" and positions[c["receipt"]] <= cap
                and stereo_free_precursors(c["precursors"]) == reference]
    first_match = min(matching, default=None)
    first_commit = exact["first_root_commit_call"]
    if first_commit is None:
        status = "no_productive_root"
    elif exact["exact_first_commit"]:
        status = "exact"
    elif first_commit == first_match:
        status = "same_connectivity_different_stereochemistry"
    else:
        status = "different_connectivity"
    if exact["first_exact_match_call"] is not None and exact["first_exact_match_call"] not in matching:
        raise ValueError("An exact literature match must also match source connectivity")
    return dict(first_root_structure=status, first_connectivity_match_call=first_match)


def references_for(panel, evaluator, mapping, *, evaluator_sha, source_lock_sha):
    if (evaluator["target_lock_sha256"] != source_lock_sha or evaluator["model_calls"]
            or mapping["evaluator_sha256"] != evaluator_sha):
        raise ValueError("Literature reference provenance changed")
    references = {r["target_id"]: r for r in evaluator["references"]}
    source = mapping["source_target_id_by_policy_target_id"]
    if (len(references) != evaluator["target_count"] or len(references) != len(evaluator["references"])
            or len(source) != len(references) or set(source.values()) != set(references)):
        raise ValueError("Evaluator mapping must be one-to-one and cover all registered references")
    matched = {row["target_id"]: references[source[row["target_id"]]] for row in panel["targets"]}
    if any(matched[row["target_id"]]["target"] != row["target"] for row in panel["targets"]):
        raise ValueError("Literature reference target differs from the native policy input")
    return matched


def score(args):
    from audit_reference_receipts import audit_arm, journal_index
    from cbyd_first_disconnection import score_root_commits

    cfg, panel, stocks, binding = bind_cbyd(args)
    evaluator = json.loads(read_pinned({"path": args.evaluator, "sha256": args.evaluator_sha256}))
    mapping = json.loads(read_pinned({"path": args.mapping, "sha256": args.mapping_sha256}))
    refs = references_for(panel, evaluator, mapping, evaluator_sha=args.evaluator_sha256,
                          source_lock_sha=binding["cbyd_source_target_lock_sha256"])
    tree_path = getattr(args, "source_trees", None)
    stereo_refs = source_connectivity(evaluator, tree_path) if tree_path is not None else None
    results = args.results.resolve()
    admission = json.loads((results / "admission.json").read_bytes())
    summary_raw = (results / "screen_result.json").read_bytes()
    summary = json.loads(summary_raw)
    if admission["inference_backend"] != "hf" or any(
            doc.get(k) != value for doc in (admission, summary) for k, value in binding.items()):
        raise ValueError("Only the admitted CbyD worker and frozen policy can supply results")
    arm, = panel["arms"]
    expected = {row["target_id"]: row for row in panel["targets"]}
    worker_rows = {row["target_id"]: row for row in summary["rows"]}
    paths = {row["target_id"]: results / "targets" / f"cbyd-{row['target_id']}-{arm}.json"
             for row in panel["targets"]}
    if (set(worker_rows) != set(expected) or len(worker_rows) != len(summary["rows"])
            or set(paths.values()) != set((results / "targets").glob("*.json"))):
        raise ValueError("Missing or duplicated CbyD target outcomes")

    raw_rows, requests = {}, set()
    for uid, path in paths.items():
        raw = path.read_bytes()
        if not raw.endswith(b"\n"):
            raise ValueError("Partially published CbyD target outcome")
        row = json.loads(raw)
        tabular = {k: v for k, v in worker_rows[uid].items() if k != "native_generation_requests"}
        row_case, stock = expected[uid], expected[uid]["stock"]
        if ({k: v for k, v in row.items() if k != "result"} != tabular
                or (row["target_id"], row["cohort"], row["arm"], row["target"], row["stock"], row["uid"])
                != (uid, "cbyd", arm, row_case["target"], stock, f"pr117/cbyd/{uid}/{row_case['seed']}/{arm}")):
            raise ValueError("Raw CbyD target outcome differs from the registered worker summary")
        ids = [decision["request_id"] for decision in row["result"]["decision_receipts"]]
        if len(set(ids)) != len(ids) or requests.intersection(ids):
            raise ValueError("A model decision receipt was reused")
        requests.update(ids)
        raw_rows[uid] = dict(row, source_sha256=sha(raw)), ids

    requested, returned, receipts, journal = journal_index(results / "native_generation.jsonl", requests, "hf")
    cap = binding["per_target_arm_request_cap"]
    cfg = dict(cfg, behavior_version="validation000000:" + binding["actor_sha256"], seed=panel["seed"])
    rows = []
    for uid, (row, ids) in raw_rows.items():
        audited = audit_arm(row, expected[uid], cfg, stocks[row["stock"]], cap, (requested, returned, receipts))
        physical_ids = [key for key in ids if key in requested]
        if len(physical_ids) != audited["gpu_requests"] or len(physical_ids) != worker_rows[uid]["native_generation_requests"]:
            raise ValueError("Physical CbyD call accounting disagrees with native replay")
        scored = score_root_commits(refs[uid], audited["commits"], physical_ids, cap)
        if stereo_refs is not None:
            scored.update(root_stereo_diagnostic(stereo_refs[refs[uid]["target_id"]], audited["commits"], physical_ids, cap, scored))
        rows.append(dict(target_id=uid, source_target_id=refs[uid]["target_id"], status=audited["status"],
                         stock_closed=audited["solved"], actual_native_calls=len(physical_ids),
                         generated_tokens=audited["generated_tokens"], source_sha256=row["source_sha256"], **scored))

    physical = sum(row["actual_native_calls"] for row in rows)
    if (physical != summary["actual_native_generation_requests"] or physical > binding["total_request_ceiling"]
            or any(physical != journal.get(k, 0) for k in ("gpu_requests", "gpu_returns", "behavior_receipts"))
            or {arm: sum(row["stock_closed"] for row in rows)} != summary["solved"]):
        raise ValueError("Final CbyD native journal does not cover the frozen evaluation exactly")
    output = dict(binding=binding, evaluator_sha256=args.evaluator_sha256, mapping_sha256=args.mapping_sha256,
                  native_summary_sha256=sha(summary_raw), actual_native_calls=physical,
                  totals=summarize_scores(rows), rows=rows)
    if stereo_refs is not None:
        output["source_tree_sha256"] = evaluator["source_tree_sha256"]
    return output


def merge_cbyd_scores(parts, cumulative, source_by_policy):
    if set(parts) != {"initial", "additional"} or len(source_by_policy) != cumulative["targets"]:
        raise ValueError("Expected both CbyD cohorts and their full cumulative identity binding")
    view_fields = {"panel_sha256", "cbyd_policy_sha256", "targets", "total_request_ceiling"}
    common = {k: v for k, v in cumulative.items() if k not in view_fields}
    provenance = {key: parts["initial"].get(key) for key in
                  ("evaluator_sha256", "mapping_sha256", "source_tree_sha256")}
    rows, stages = {}, {}
    cap = cumulative["per_target_arm_request_cap"]
    for name, part in parts.items():
        binding = part["binding"]
        if ({k: v for k, v in binding.items() if k not in view_fields} != common
                or any(part.get(key) != value for key, value in provenance.items())):
            raise ValueError("The CbyD cohorts must share one frozen evaluation")
        cases = part["rows"]
        calls = sum(row["actual_native_calls"] for row in cases)
        if (len(cases) != binding["targets"] or calls != part["actual_native_calls"]
                or any(not 0 <= row["actual_native_calls"] <= cap for row in cases)
                or binding["total_request_ceiling"] != len(cases) * cap):
            raise ValueError("CbyD cohort counts do not match its physical work and ceiling")
        for row in cases:
            target_id = row["target_id"]
            if target_id in rows:
                raise ValueError("Duplicate CbyD outcome across the two cohorts")
            if source_by_policy.get(target_id) != row["source_target_id"]:
                raise ValueError("CbyD outcome differs from the cumulative source identity")
            rows[target_id] = dict(row, cohort_stage=name)
        stages[name] = dict(binding=binding, native_summary_sha256=part["native_summary_sha256"],
                            actual_native_calls=calls, totals=summarize_scores(cases))
    if rows.keys() != source_by_policy.keys():
        raise ValueError("CbyD outcomes must cover the full registered cumulative cohort")
    ordered = [rows[target_id] for target_id in source_by_policy]
    return dict(binding=cumulative, **{k: v for k, v in provenance.items() if v is not None}, stages=stages,
                actual_native_calls=sum(row["actual_native_calls"] for row in ordered),
                totals=summarize_scores(ordered), rows=ordered)


def prepare_cbyd_merge(args):
    plan = json.loads(read_pinned(dict(path=args.merge_prepared, sha256=args.merge_prepared_sha256)))
    views = plan["views"]
    if (views["initial"]["policy_sha256"] != args.cbyd_policy_sha256
            or plan["reference_confirmation_registration_sha256"] != args.confirmation_registration_sha256
            or plan["separate_private_evaluator_sha256"] != args.evaluator_sha256
            or plan["separate_evaluator_id_mapping_sha256"] != args.mapping_sha256):
        raise ValueError("CbyD reporting inputs differ from the registered split")
    root = args.merge_prepared.parent
    policies = dict(initial=args.cbyd_policy, additional=root / views["additional"]["policy"],
                    cumulative=args.cumulative_policy)
    rows, seeds = {}, set()
    for name, path in policies.items():
        view = json.loads(read_pinned(dict(path=path, sha256=views[name]["policy_sha256"])))
        seeds.add(view["seed"])
        rows[name] = {row["target_id"]: row for row in view["targets"]}
        if len(rows[name]) != len(view["targets"]) or len(rows[name]) != views[name]["targets"]:
            raise ValueError("Unexpected identity or denominator in the registered CbyD split")
    if (len(seeds) != 1 or rows["initial"].keys() & rows["additional"].keys()
            or (rows["initial"] | rows["additional"]) != rows["cumulative"]):
        raise ValueError("CbyD policy views must retain the original identities, molecules and seeds")
    cumulative_args = argparse.Namespace(**vars(args))
    cumulative_args.cbyd_policy = policies["cumulative"]
    cumulative_args.cbyd_policy_sha256 = views["cumulative"]["policy_sha256"]
    _, panel, _, binding = bind_cbyd(cumulative_args)
    if any(len(part) * binding["per_target_arm_request_cap"] != views[name]["calls_ceiling"] for name, part in rows.items()):
        raise ValueError("CbyD cohort call budgets changed after registration")
    evaluator = json.loads(read_pinned(dict(path=args.evaluator, sha256=args.evaluator_sha256)))
    mapping = json.loads(read_pinned(dict(path=args.mapping, sha256=args.mapping_sha256)))
    references = references_for(panel, evaluator, mapping, evaluator_sha=args.evaluator_sha256,
                                source_lock_sha=binding["cbyd_source_target_lock_sha256"])
    additional_args = argparse.Namespace(**vars(args))
    additional_args.cbyd_policy = policies["additional"]
    additional_args.cbyd_policy_sha256 = views["additional"]["policy_sha256"]
    additional_args.results = args.additional_results
    return additional_args, binding, {key: ref["target_id"] for key, ref in references.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    for name in ("evaluator", "mapping", "results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("evaluator-sha256", "mapping-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--source-trees", type=Path, help="Pinned evaluator-only trees for the stereochemistry diagnostic")
    for name in ("merge-prepared", "additional-results", "cumulative-policy"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--merge-prepared-sha256")
    args = parser.parse_args()
    output = args.output.resolve()
    merge_inputs = (args.merge_prepared, args.merge_prepared_sha256, args.additional_results, args.cumulative_policy)
    if any(merge_inputs) and not all(merge_inputs):
        parser.error("A cumulative report requires the registered split, its hash, the cumulative policy and both results")
    raw_dirs = [path.resolve() for path in (args.results, args.additional_results) if path is not None]
    if not output.is_relative_to(CEPH) or any(output.is_relative_to(path) for path in raw_dirs):
        parser.error("Write evaluator results to shared storage outside the raw policy outputs")
    if args.merge_prepared is None:
        result = score(args)
    else:
        additional, binding, identities = prepare_cbyd_merge(args)
        parts = {"initial": score(args), "additional": score(additional)}
        result = merge_cbyd_scores(parts, binding, identities)
        result["split_plan_sha256"] = args.merge_prepared_sha256
    write_new(output, result)
    print(json.dumps(result["totals"]))


if __name__ == "__main__":
    main()
