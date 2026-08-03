import importlib.util
import json
from pathlib import Path

import pytest

from mechet.knowledge_ablation import (
    condition_metrics,
    make_irrelevant_context_control,
    matched_intersection,
    strip_knowledge_messages,
    validate_alignment,
)


def _load_conversational_records():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_tool_sft.py"
    spec = importlib.util.spec_from_file_location("mechet_train_tool_sft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.conversational_records


conversational_records = _load_conversational_records()


def tool_call(name, arguments=None, call_id="call_1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments or {}),
                },
            }
        ],
    }


def tool_result(name, result, call_id="call_1"):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result),
    }


def row(identifier, target, context_text, passage_id):
    context = {
        "text": context_text,
        "passage_ids": [passage_id],
        "context_sha256": f"hash-{passage_id}",
        "n_characters": len(context_text),
        "truncated": False,
    }
    return {
        "id": identifier,
        "target_smiles": target,
        "expected_precursor": target,
        "messages": [
            {"role": "system", "content": "Use trace-owned tools."},
            {"role": "user", "content": f"TARGET: {target}"},
            tool_call(
                "retrieve_textbook_guidance",
                {"query": passage_id, "top_k": 1},
                "knowledge",
            ),
            tool_result(
                "retrieve_textbook_guidance",
                {
                    "ok": True,
                    "query": passage_id,
                    "state_smiles": target,
                    "context": context,
                    "matches": [{"passage_id": passage_id}],
                    "soft_evidence_only": True,
                    "direct_reward": False,
                },
                "knowledge",
            ),
            tool_call(
                "apply_coupled_electron_moves",
                {"moves_json": "[]"},
                "chemistry",
            ),
            tool_result(
                "apply_coupled_electron_moves",
                {"ok": True, "trace_bound": True},
                "chemistry",
            ),
            tool_call("finish_trace", {}, "finish"),
            tool_result(
                "finish_trace",
                {
                    "ok": True,
                    "formal_execute": True,
                    "endpoint_exact": True,
                    "trace_bound": True,
                },
                "finish",
            ),
        ],
        "metadata": {
            "endpoint_source": "environment_owned_trace",
            "knowledge_condition": "textbook_rag",
            "textbook_passage_ids": [passage_id],
            "textbook_context_characters": len(context_text),
        },
    }


def rows():
    return [
        row(
            "r1",
            "[CH3:1][C:2](=[O:3])[CH3:4]",
            "Carbonyl addition moves electron density toward oxygen.",
            "carbonyl",
        ),
        row(
            "r2",
            "[CH3:1][Br:2].[OH-:3]",
            "Substitution couples nucleophile attack with leaving group departure.",
            "substitution",
        ),
    ]


def textbook_result(value):
    message = next(
        message
        for message in value["messages"]
        if message.get("role") == "tool"
        and message.get("name") == "retrieve_textbook_guidance"
    )
    return json.loads(message["content"])


def test_matched_intersection_preserves_ids_and_endpoints():
    original = rows()
    stripped = [strip_knowledge_messages(item) for item in original]
    identifiers, matched = matched_intersection(
        {"textbook": original, "none": stripped}
    )
    assert identifiers == ["r1", "r2"]
    assert [item["id"] for item in matched["none"]] == identifiers


def test_alignment_rejects_target_mismatch():
    original = rows()
    changed = [dict(item) for item in original]
    changed[0] = dict(changed[0])
    changed[0]["target_smiles"] = "[CH4:1]"
    with pytest.raises(ValueError, match="matched endpoint mismatch"):
        validate_alignment({"left": original, "right": changed})


def test_strip_knowledge_retains_chemistry_trace():
    value = strip_knowledge_messages(rows()[0])
    names = []
    for message in value["messages"]:
        if message.get("role") == "tool":
            names.append(message.get("name"))
        for call in message.get("tool_calls") or []:
            names.append((call.get("function") or {}).get("name"))
    assert "retrieve_textbook_guidance" not in names
    assert "apply_coupled_electron_moves" in names
    assert "finish_trace" in names
    assert value["metadata"]["knowledge_condition"] == "none"


def test_irrelevant_control_rotates_only_text_and_matches_length():
    original = rows()
    controlled = make_irrelevant_context_control(original)
    for source, target in zip(original, controlled):
        source_result = textbook_result(source)
        target_result = textbook_result(target)
        source_context = source_result["context"]
        target_context = target_result["context"]
        assert target_context["n_characters"] == source_context["n_characters"]
        assert target_context["control_type"] == "length_matched_irrelevant"
        assert target_context["text"] != source_context["text"]
        assert target_result["query"] == source_result["query"]
        assert target_result["state_smiles"] == source_result["state_smiles"]
        assert target_result["matches"] == []
        assert target["metadata"]["textbook_control_donor_id"] != source["id"]


def test_tool_sft_records_remain_conversational():
    prepared = conversational_records(rows())
    assert prepared[0]["id"] == "r1"
    assert "messages" in prepared[0]
    assert "text" not in prepared[0]
    assert any(message.get("tool_calls") for message in prepared[0]["messages"])


def test_condition_metrics_enforce_trace_and_zero_knowledge_reward():
    metrics = condition_metrics(rows())
    assert metrics["textbook_call_rate"] == 1.0
    assert metrics["trace_bound_rate"] == 1.0
    assert metrics["execute_rate"] == 1.0
    assert metrics["endpoint_exact_rate"] == 1.0
    assert metrics["knowledge_direct_reward_violations"] == 0
