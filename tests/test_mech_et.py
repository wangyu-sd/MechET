"""Tests for MECH_ET v3 (FlowER BE-delta mechanism CoT)."""

from __future__ import annotations

from pathlib import Path

from mechet.mech_et import (
    be_delta_from_mapped_smiles,
    electron_conserved,
    format_mech_et_cot,
    parse_mech_et_body,
    verify_mech_et,
)
from mechet.mech_graph import build_mechanism_graph, load_flower_graphs
from mechet.sft import convert_record_to_qwen_sft, parse_mech_et_output
from mechet.verifier import compute_mech_et_reward, compute_reward

FLOWER_VAL = Path("/aaa/fionafyang/buddy1/whaleywang/datasets/retro/data/flower_new_dataset/val.txt")


def test_be_delta_simple_and_conserved():
    # C -> CC is not a real elementary BE step; use identical then a charge/LP toy if needed.
    # Real FlowER step from val.
    graphs, _ = load_flower_graphs(FLOWER_VAL, limit=5)
    g = graphs[0]
    src, dst = g.reverse_edges()[0]
    delta = be_delta_from_mapped_smiles(g.states[src].mapped_smiles, g.states[dst].mapped_smiles)
    assert delta is not None
    # Matching self-diff is empty and conserved
    empty = be_delta_from_mapped_smiles(g.states[src].mapped_smiles, g.states[src].mapped_smiles)
    assert empty is not None
    assert empty.is_empty()
    assert electron_conserved(empty)


def test_mech_et_roundtrip_linear_tree_dag():
    assert FLOWER_VAL.exists()
    graphs, skip = load_flower_graphs(FLOWER_VAL, limit=50)
    assert not skip
    by_id = {g.trajectory_id: g for g in graphs}
    samples = [by_id["1"], by_id["16"], by_id["11"]]
    assert samples[1].topology == "tree"
    assert samples[2].topology == "dag_branch_join"
    for g in samples:
        body = format_mech_et_cot(g)
        assert body.startswith("MECH_ET v3")
        assert "BE_DELTA" in body
        assert "ET_SIGNATURE" in body
        assert "PERCEIVE" in body
        parsed = parse_mech_et_body(body)
        assert parsed["ok"]
        verified = verify_mech_et(
            mechanism_body=body,
            answer=g.compact_precursor_smiles(),
            main_product=g.compact_main_product(),
            expected_graph=g,
        )
        assert verified["format_ok"]
        assert verified["reachability_ok"]
        assert verified["be_delta_exact"]
        assert verified["graph_exact"]
        assert verified["answer_exact"]
        assert verified["main_product_ok"]


def test_sft_convert_and_reward_no_leak():
    graphs, _ = load_flower_graphs(FLOWER_VAL, limit=5)
    g = graphs[0]
    body = format_mech_et_cot(g)
    answer = g.compact_precursor_smiles()
    product = g.compact_main_product()
    row = convert_record_to_qwen_sft(
        {
            "task_type": "mech_et_cot_retro",
            "product_smiles": product,
            "mechanism": body,
            "reactants": answer,
        }
    )
    assert row["task_type"] == "mech_et_cot_retro"
    user = row["messages"][1]["content"]
    assert user.startswith("TARGET:")
    assert "topology=" not in user
    assert "n_states=" not in user
    assistant = row["messages"][-1]["content"]
    parsed = parse_mech_et_output(assistant)
    assert parsed["format_ok"]
    assert parsed["graph_ok"]
    reward = compute_mech_et_reward(
        assistant,
        product,
        expected_precursor=answer,
        expected_graph=g,
    )
    assert not reward["hard_fail"]
    assert reward["total"] > 0
    assert compute_reward(assistant, product, expected_precursors=[answer])["total"] > 0


def test_missing_be_delta_hard_fails_with_gt():
    graphs, _ = load_flower_graphs(FLOWER_VAL, limit=5)
    g = graphs[0]
    body = format_mech_et_cot(g)
    # Strip all BE_DELTA blocks but keep RETRO_EDGE lines.
    lines = []
    skip = False
    for ln in body.splitlines():
        if ln.strip() == "BE_DELTA" or ln.strip().startswith("BOND ") or ln.strip().startswith("LP ") or ln.strip().startswith("CHARGE "):
            skip = True
            continue
        if ln.startswith("RETRO_EDGE") or ln.startswith("STATE") or ln.startswith("N_") or not ln.startswith(" "):
            skip = False
        if skip and ln.startswith("  "):
            continue
        lines.append(ln)
    broken = "\n".join(lines)
    answer = g.compact_precursor_smiles()
    wrapped = f"<mechanism>\n{broken}\n</mechanism>\n<answer>\n{answer}\n</answer>"
    reward = compute_mech_et_reward(
        wrapped,
        g.compact_main_product(),
        expected_precursor=answer,
        expected_graph=g,
    )
    assert reward["hard_fail"]
