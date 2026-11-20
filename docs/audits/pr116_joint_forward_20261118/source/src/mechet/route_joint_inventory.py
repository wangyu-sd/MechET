"""Extract one trusted PR112 committed edge for a charge-preserving FlowER scorer.

Supply a single rollout (or an individual ReFIT suffix report). Dense indices
are one-based. Graph hydrogens retain their executor provenance; new hydrogens
added for the teacher point to their executor-map parent. A teacher wrapper
must bypass reactant_process(), which would remap this already prepared input.
"""
from collections import Counter
from hashlib import sha256
import json

from rdkit import Chem
from mechet.agent_inference import parse_tool_calls

CONTRACT = "pr116_pr112_full_endpoint_dense_h_project_stereo_v2"


def _require(ok, why):
    if not ok:
        raise ValueError(why)


def _molecule(value):
    _require(isinstance(value, str) and value.strip(), "empty_state")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(value, params)
    _require(mol is not None and mol.GetNumAtoms() > 0, "invalid_state")
    return mol


def _maps(mol):
    result = {a.GetAtomMapNum(): (a.GetAtomicNum(), a.GetIsotope()) for a in mol.GetAtoms()}
    _require(len(result) == mol.GetNumAtoms() and min(result) > 0, "incomplete_or_duplicate_map")
    return result


def _unmapped(mol, collapse_h=False):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if collapse_h:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _teacher_projection(mol):
    _require(not any(atom.GetIsotope() for atom in mol.GetAtoms()),
             "teacher_isotope_unsupported")
    projected = Chem.Mol(mol)
    Chem.RemoveStereochemistry(projected)
    return projected


def _digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def _check_import(before, after, args):
    # Frozen append_mapped_fragments_verbatim preserves the old bytes.
    _require(after.startswith(before + "."), "import_changed_prior_state")
    suffix = after[len(before) + 1:]
    declared = Counter()
    for item in args["fragments"]:
        count = int(item["count"])
        _require(count >= 1, "invalid_import_count")
        for component in _unmapped(_molecule(item["smiles"])).split("."):
            declared[component] += count
    actual = Counter(_unmapped(_molecule(suffix)).split("."))
    _require(declared == actual, "import_inventory_mismatch")
    return suffix


def _prepare_teacher(full):
    # Start in a deterministic canonical order even if the state is reordered.
    mol = Chem.AddHs(_molecule(full))
    executor_maps, generated = {}, {}
    for atom in mol.GetAtoms():
        old = atom.GetAtomMapNum()
        dense = atom.GetIdx() + 1
        if old:
            executor_maps[old] = dense
        else:
            _require(atom.GetAtomicNum() == 1 and atom.GetDegree() == 1, "unbound_added_h")
            parent = atom.GetNeighbors()[0].GetAtomMapNum()
            _require(parent in executor_maps, "unbound_added_h")
            generated[dense] = parent
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    teacher_smi = Chem.MolToSmiles(mol, allHsExplicit=True, isomericSmiles=True)
    return teacher_smi, executor_maps, generated


def _accepted_chain(report, edge):
    occurrence = edge["occurrence"]
    _require(isinstance(occurrence, str) and occurrence, "missing_occurrence")
    _require(sum(e["occurrence"] == occurrence for e in report["reactions"]) == 1,
             "ambiguous_committed_occurrence")
    records = [(i, r) for i, r in enumerate(report["records"])
               if r["occurrence"] == occurrence and r.get("accepted") is True]
    actions = edge["actions"]
    _require(records and len(records) == len(actions), "accepted_action_count_mismatch")
    version = report['behavior_version']
    _require(isinstance(version, str) and version, "missing_policy_version")
    request_ids = set()
    current = edge["initial_state"]
    additions, has_flow = [], False
    for ordinal, ((_, record), action) in enumerate(zip(records, actions)):
        rid = record["request_id"]
        _require(isinstance(rid, str) and rid and rid not in request_ids, "invalid_request_id")
        request_ids.add(rid)
        # A ReFIT suffix can inherit genuine prefix receipts from a different
        # episode/reaction id. Occurrence, version and bytewise states bind them.
        _require(record["behavior_version"] == version and not record["error"] and
                 not record.get("truncated") and not record.get("resource_error"),
                 "untrusted_accepted_receipt")
        _require(current == record["state_before"] == action["state_before"], "state_chain_mismatch")
        _require(action["result"]["ok"] is True and action["result"]["code"] == "PASS",
                 "action_result_not_pass")
        name, args = action["name"], action["arguments"]
        parsed = parse_tool_calls(record["raw"])
        _require(len(parsed) == 1 and (parsed[0].name, parsed[0].arguments) == (name, args),
                 "sampled_action_mismatch")
        after = record["state_after"]
        before_mol, after_mol = _molecule(current), _molecule(after)
        before_maps, after_maps = _maps(before_mol), _maps(after_mol)
        if name == "import_fragments":
            additions.append(_check_import(current, after, args))
        elif name == "apply_electron_flow":
            _require(before_maps == after_maps, "flow_map_inventory_changed")
            _require(Chem.GetFormalCharge(before_mol) == Chem.GetFormalCharge(after_mol),
                     "flow_charge_inventory_changed")
            _require(Chem.AddHs(before_mol).GetNumAtoms() == Chem.AddHs(after_mol).GetNumAtoms(),
                     "flow_hydrogen_inventory_changed")
            has_flow = True
        elif name == "finish_trace":
            _require(ordinal == len(actions) - 1 and not args and current == after,
                     "invalid_finish_transition")
        else:
            raise ValueError("unknown_accepted_tool")
        current = after
    _require(has_flow and actions[-1]["name"] == "finish_trace" and current == edge["final_state"],
             "missing_committed_finish")
    return records, additions


def _extract(report, index):
    _require(type(index) is int and index >= 0, "invalid_reaction_index")
    edge = report["reactions"][index]
    initial, target = _molecule(edge["initial_state"]), _molecule(edge["public_product"])
    _require(all(a.GetAtomMapNum() == 0 for a in target.GetAtoms()) and
             len(Chem.GetMolFrags(target)) == 1, "invalid_public_main_product")
    _require(_unmapped(initial, True) == _unmapped(target, True), "initial_public_product_mismatch")
    projected_target = _teacher_projection(target)
    records, additions = _accepted_chain(report, edge)
    final = _molecule(edge["final_state"])
    projected_final = _teacher_projection(final)
    maps_initial, maps_final = _maps(initial), _maps(final)
    assembled = _molecule(".".join([edge["initial_state"], *additions]))
    _require(_maps(assembled) == maps_final, "net_map_inventory_mismatch")
    full = Chem.MolToSmiles(final, isomericSmiles=True)
    _require(Chem.MolToSmiles(assembled, isomericSmiles=True) != full, "no_net_edit")
    heavy = sorted(a.GetAtomMapNum() for a in initial.GetAtoms() if a.GetAtomicNum() > 1)
    _require(heavy, "missing_target_heavy_maps")
    projected = Chem.MolToSmiles(projected_final, isomericSmiles=True)
    projected_start = Chem.MolToSmiles(_teacher_projection(assembled), isomericSmiles=True)
    _require(projected_start != projected, "no_net_constitutional_edit")
    teacher, lut, generated = _prepare_teacher(projected)
    index_finish, finish = records[-1]
    transaction = [{k: row[k] for k in ("request_id", "raw", "state_before", "state_after",
                                         "occurrence", "accepted", "behavior_version")}
                   for _, row in records]
    return dict(
        status="eligible", contract=CONTRACT, full_reactants_mapped=full,
        target_product=_unmapped(projected_target, True), teacher_reactants_mapped=teacher,
        executor_to_teacher_map=lut, target_heavy_teacher_maps=sorted(lut[m] for m in heavy),
        generated_hydrogen_parents=generated, finish_index=index_finish,
        finish_request_id=finish["request_id"],
        provenance=dict(occurrence=edge["occurrence"], accepted_action_count=len(records),
                        imported_executor_maps=sorted(maps_final.keys() - maps_initial.keys()),
                        public_product_sha256=_digest(edge["public_product"]),
                        initial_state_sha256=_digest(edge["initial_state"]),
                        final_state_sha256=_digest(edge["final_state"]),
                        transaction_sha256=_digest(json.dumps(transaction, sort_keys=True,
                                                             separators=(",", ":")))),
    )


def extract_forward_input(report, reaction_index):
    """Return an eligible input or an unavailable reason; neither contains a score.

    Supply the trusted executor report and index of its committed reaction.
    The caller owns checkpoint/code pinning, teacher applicability (vocabulary,
    supported bond types and budget), persistent deduplication and scoring.
    """
    try:
        return _extract(report, reaction_index)
    except ValueError as exc:
        reason = str(exc)
        if not reason.isidentifier():
            reason = "malformed_executor_record"
    except (AttributeError, IndexError, KeyError, TypeError, RuntimeError, OverflowError):
        reason = "missing_or_malformed_executor_evidence"
    return dict(status="unavailable", contract=CONTRACT, reason=reason)
