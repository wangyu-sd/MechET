"""MechET evaluation metrics aligned with flower_completion strict EM baselines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from mechet.mech_et import parse_mech_et_body, verify_mech_et
from mechet.sft import parse_mech_cot_output
from mechet.verifier import compute_mech_et_reward

TOPK_LEVELS = (1, 3, 5, 10)


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


def extract_answer_from_prediction(prediction: str) -> str:
    return str(parse_mech_cot_output(prediction).get("answer") or "").strip()


def canonical_species(smiles: str) -> list[str]:
    """ORBIT endpoint matching: per-fragment RDKit canonical SMILES, sorted."""
    from rdkit import Chem

    out: list[str] = []
    for part in (smiles or "").replace(";", ".").split("."):
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


def canonical_multiset(smiles: str) -> frozenset[str] | None:
    try:
        return frozenset(canonical_species(smiles))
    except ValueError:
        return None


def answer_parse_ok(answer: str) -> bool:
    return canonical_multiset(answer) is not None and bool((answer or "").strip())


def top1_strict_match(predicted: str, gold: str) -> bool:
    pred = canonical_multiset(predicted)
    gt = canonical_multiset(gold)
    return bool(pred and gt and pred == gt)


def largest_fragment_canonical(smiles: str) -> str | None:
    from rdkit import Chem

    best: str | None = None
    best_heavy = -1
    for part in (smiles or "").replace(";", ".").split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        heavy = mol.GetNumHeavyAtoms()
        if heavy > best_heavy:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            best = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            best_heavy = heavy
    return best


def top1_main_only_match(predicted: str, gold: str) -> bool:
    pred = largest_fragment_canonical(predicted)
    gt = largest_fragment_canonical(gold)
    return bool(pred and gt and pred == gt)


def topk_strict_hit(candidates: Sequence[str], gold: str, k: int) -> bool:
    gt = canonical_multiset(gold)
    if not gt:
        return False
    for answer in candidates[:k]:
        pred = canonical_multiset(str(answer or "").strip())
        if pred and pred == gt:
            return True
    return False


def topk_main_only_hit(candidates: Sequence[str], gold: str, k: int) -> bool:
    gt = largest_fragment_canonical(gold)
    if not gt:
        return False
    for answer in candidates[:k]:
        pred = largest_fragment_canonical(str(answer or "").strip())
        if pred and pred == gt:
            return True
    return False


def normalize_candidates(prediction: str, candidates: Iterable[str] | None = None) -> list[str]:
    """Dedupe answer strings extracted from one or more model outputs."""
    seen: set[str] = set()
    out: list[str] = []
    for text in [prediction, *(candidates or [])]:
        ans = extract_answer_from_prediction(str(text or ""))
        if not ans or ans in seen:
            continue
        seen.add(ans)
        out.append(ans)
    return out


def precursor_state_agree(mechanism_body: str, answer: str) -> bool:
    parsed = parse_mech_et_body(mechanism_body)
    if not parsed.get("ok"):
        return False
    precursor_id = str(parsed.get("precursor_state_id") or "")
    states = parsed.get("states") or {}
    if not precursor_id or precursor_id not in states:
        return False
    verified = verify_mech_et(
        mechanism_body=mechanism_body,
        answer=answer,
        expected_precursor=str(states[precursor_id]),
    )
    return bool(verified.get("answer_exact"))


def score_mech_et_prediction(
    row: dict[str, Any],
    prediction: str,
    *,
    mode: str = "model",
    candidates: Sequence[str] | None = None,
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
    answer_candidates = normalize_candidates(prediction, candidates)

    expected = gold if mode == "model" else (answer or gold)
    verified = verify_mech_et(
        mechanism_body=mechanism,
        answer=answer,
        main_product=product or None,
        expected_precursor=expected or None,
    )
    reward = compute_mech_et_reward(
        prediction
        if "<mechanism>" in prediction.lower()
        else f"<mechanism>\n{mechanism}\n</mechanism>\n<answer>\n{answer}\n</answer>",
        product,
        expected_precursor=expected if mode == "model" else gold,
    )

    meta = row.get("metadata") or {}
    case: dict[str, Any] = {
        "id": row.get("id"),
        "topology": meta.get("topology"),
        "et_signature": meta.get("et_signature"),
        "format_ok": bool(parsed.get("format_ok") and verified.get("format_ok")),
        "reachability_ok": bool(verified.get("reachability_ok")),
        "be_delta_exact": bool(verified.get("be_delta_exact")),
        "electron_conserved": bool(verified.get("electron_conserved")),
        "main_product_ok": bool(verified.get("main_product_ok")),
        "answer_exact": bool(verified.get("answer_exact")),
        "answer_parse_ok": answer_parse_ok(answer),
        "state_agree": precursor_state_agree(mechanism, answer) if parsed.get("format_ok") else False,
        "reward_total": float(reward.get("total") or 0.0),
        "hard_fail": bool(reward.get("hard_fail")),
        "n_states": int(verified.get("n_states") or 0),
        "n_edges": int(verified.get("n_edges") or 0),
        "n_candidates": len(answer_candidates),
    }

    if mode == "model" and gold:
        case["top1_strict"] = top1_strict_match(answer, gold)
        case["top1_main_only"] = top1_main_only_match(answer, gold)
        case["answer_top1"] = case["top1_strict"]
        for k in TOPK_LEVELS:
            case[f"top{k}_strict"] = topk_strict_hit(answer_candidates, gold, k)
            case[f"top{k}_main_only"] = topk_main_only_hit(answer_candidates, gold, k)
    else:
        case["top1_strict"] = False
        case["top1_main_only"] = False
        case["answer_top1"] = bool(verified.get("answer_exact"))
        for k in TOPK_LEVELS:
            case[f"top{k}_strict"] = False
            case[f"top{k}_main_only"] = False

    return case


def aggregate_rates(cases: list[dict[str, Any]]) -> dict[str, Any]:
    bool_keys = [
        "format_ok",
        "reachability_ok",
        "be_delta_exact",
        "electron_conserved",
        "main_product_ok",
        "answer_exact",
        "answer_top1",
        "top1_strict",
        "top1_main_only",
        "answer_parse_ok",
        "state_agree",
    ]
    for k in TOPK_LEVELS:
        bool_keys.extend([f"top{k}_strict", f"top{k}_main_only"])

    n = len(cases)
    totals = {k: sum(int(bool(c.get(k))) for c in cases) for k in bool_keys}
    totals["n"] = n
    totals["reward_mean"] = sum(float(c.get("reward_total") or 0.0) for c in cases) / n if n else 0.0
    rates = {f"{k}_rate": (totals[k] / n if n else 0.0) for k in bool_keys}
    rates["reward_mean"] = totals["reward_mean"]
    return {"totals": totals, "rates": rates}


def aggregate_by_topology(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        buckets[str(case.get("topology") or "unknown")].append(case)
    return {topo: aggregate_rates(rows) for topo, rows in sorted(buckets.items())}


def build_eval_report(
    cases: list[dict[str, Any]],
    *,
    mode: str,
    data_path: str,
    predictions_path: str,
    missing_predictions: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "mode": mode,
        "data": data_path,
        "predictions": predictions_path,
        "missing_predictions": missing_predictions,
        **aggregate_rates(cases),
        "by_topology": aggregate_by_topology(cases),
    }
    if extra:
        report.update(extra)
    return report


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
    top5 = rates.get("top5_strict_rate")
    return {
        "method": method,
        "backbone": backbone,
        "train_examples": str(train_n),
        "test_examples": str(test_n),
        "top1": f"{rates.get('top1_strict_rate', 0.0):.4f}",
        "top5": f"{top5:.4f}" if top5 is not None else "NOT_EXECUTED",
        "valid_precursors": f"{rates.get('answer_parse_ok_rate', 0.0):.4f}",
        "parse_rate": f"{rates.get('format_ok_rate', 0.0):.4f}",
        "compile_rate": f"{rates.get('reachability_ok_rate', 0.0):.4f}",
        "execute_rate": f"{rates.get('be_delta_exact_rate', 0.0):.4f}",
        "endpoint_rate": f"{rates.get('top1_strict_rate', 0.0):.4f}",
        "repair_success": "N/A",
        "status": status,
    }
