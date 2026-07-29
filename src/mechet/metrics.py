"""MechET evaluation metrics aligned with ORBIT single-step / mech_cot tables."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mechet.mech_et import verify_mech_et
from mechet.sft import parse_mech_cot_output
from mechet.verifier import compute_mech_et_reward


def extract_product_from_user(content: str) -> str:
    first = (content or "").split("\n", 1)[0]
    return first.replace("TARGET:", "").strip()


def extract_gold_answer(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    if meta.get("initial_reactants"):
        return str(meta["initial_reactants"])
    messages = row.get("messages") or []
    if messages:
        parsed = parse_mech_cot_output(str(messages[-1].get("content") or ""))
        return str(parsed.get("answer") or "")
    return ""


def canonical_species(smiles: str) -> list[str]:
    """ORBIT endpoint matching: per-fragment RDKit canonical SMILES, sorted."""
    from rdkit import Chem

    out: list[str] = []
    for part in (smiles or "").split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            raise ValueError(f"unparseable fragment: {part[:80]}")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return sorted(out)


def answer_parse_ok(answer: str) -> bool:
    if not (answer or "").strip():
        return False
    try:
        canonical_species(answer)
        return True
    except ValueError:
        return False


def top1_strict_match(predicted: str, gold: str) -> bool:
    if not predicted.strip() or not gold.strip():
        return False
    try:
        return canonical_species(predicted) == canonical_species(gold)
    except ValueError:
        return False


def precursor_state_agree(parsed_mechanism: dict[str, Any], answer: str) -> bool:
    body = parsed_mechanism.get("parsed") if isinstance(parsed_mechanism, dict) else None
    if not body:
        return False
    states = body.get("states") or {}
    precursor_id = body.get("precursor_state_id")
    if not precursor_id or precursor_id not in states:
        return False
    verified = verify_mech_et(
        mechanism_body=str(parsed_mechanism.get("mechanism") or ""),
        answer=answer,
        expected_precursor=states[precursor_id],
    )
    return bool(verified.get("answer_exact"))


def score_mech_et_prediction(
    row: dict[str, Any],
    prediction: str,
    *,
    mode: str = "model",
) -> dict[str, Any]:
    """Score one prediction against row metadata.

    mode:
      - ``model``: compare to gold ``initial_reactants`` (real eval)
      - ``gold_audit``: compare answer to itself (data QC only)
    """
    messages = row.get("messages") or []
    product = ""
    for msg in messages:
        if msg.get("role") == "user" and str(msg.get("content") or "").startswith("TARGET:"):
            product = extract_product_from_user(str(msg["content"]))
            break

    gold = extract_gold_answer(row)
    parsed = parse_mech_cot_output(prediction)
    mechanism = str(parsed.get("mechanism") or "")
    answer = str(parsed.get("answer") or "").strip()

    expected = gold if mode == "model" else (answer or gold)
    verified = verify_mech_et(
        mechanism_body=mechanism,
        answer=answer,
        main_product=product or None,
        expected_precursor=expected or None,
    )
    reward = compute_mech_et_reward(
        prediction if "<mechanism>" in prediction.lower() else f"<mechanism>\n{mechanism}\n</mechanism>\n<answer>\n{answer}\n</answer>",
        product,
        expected_precursor=expected if mode == "model" else gold,
    )

    meta = row.get("metadata") or {}
    case = {
        "id": row.get("id"),
        "topology": meta.get("topology"),
        "et_signature": meta.get("et_signature"),
        "format_ok": bool(parsed.get("format_ok") and verified.get("format_ok")),
        "reachability_ok": bool(verified.get("reachability_ok")),
        "be_delta_exact": bool(verified.get("be_delta_exact")),
        "electron_conserved": bool(verified.get("electron_conserved")),
        "main_product_ok": bool(verified.get("main_product_ok")),
        "answer_exact": bool(verified.get("answer_exact")),
        "answer_top1": bool(gold) and top1_strict_match(answer, gold) if mode == "model" else bool(verified.get("answer_exact")),
        "top1_strict": bool(gold) and top1_strict_match(answer, gold) if mode == "model" else False,
        "answer_parse_ok": answer_parse_ok(answer),
        "state_agree": precursor_state_agree(parsed, answer) if parsed.get("format_ok") else False,
        "reward_total": float(reward.get("total") or 0.0),
        "hard_fail": bool(reward.get("hard_fail")),
        "n_states": int(verified.get("n_states") or 0),
        "n_edges": int(verified.get("n_edges") or 0),
    }
    return case


def aggregate_rates(cases: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "format_ok",
        "reachability_ok",
        "be_delta_exact",
        "electron_conserved",
        "main_product_ok",
        "answer_exact",
        "answer_top1",
        "top1_strict",
        "answer_parse_ok",
        "state_agree",
    ]
    n = len(cases)
    totals = {k: sum(int(bool(c.get(k))) for c in cases) for k in keys}
    totals["n"] = n
    totals["reward_mean"] = sum(float(c.get("reward_total") or 0.0) for c in cases) / n if n else 0.0
    rates = {f"{k}_rate": (totals[k] / n if n else 0.0) for k in keys}
    rates["reward_mean"] = totals["reward_mean"]
    return {"totals": totals, "rates": rates}


def aggregate_by_topology(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        buckets[str(case.get("topology") or "unknown")].append(case)
    return {topo: aggregate_rates(rows) for topo, rows in sorted(buckets.items())}


def single_step_table_row(
    method: str,
    *,
    backbone: str,
    train_n: int | str,
    test_n: int | str,
    agg: dict[str, Any],
    status: str = "COMPLETED",
) -> dict[str, str]:
    """One row compatible with ORBIT ``single_step_main_table.tsv`` columns."""
    rates = agg.get("rates") or {}
    return {
        "method": method,
        "backbone": backbone,
        "train_examples": str(train_n),
        "test_examples": str(test_n),
        "top1": f"{rates.get('top1_strict_rate', 0.0):.4f}",
        "top5": "NOT_EXECUTED",
        "valid_precursors": f"{rates.get('answer_parse_ok_rate', 0.0):.4f}",
        "parse_rate": f"{rates.get('format_ok_rate', 0.0):.4f}",
        "compile_rate": f"{rates.get('reachability_ok_rate', 0.0):.4f}",
        "execute_rate": f"{rates.get('be_delta_exact_rate', 0.0):.4f}",
        "endpoint_rate": f"{rates.get('answer_top1_rate', 0.0):.4f}",
        "repair_success": "N/A",
        "status": status,
    }
