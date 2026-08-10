import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_evidence_interventions.py"
    spec = importlib.util.spec_from_file_location("build_evidence_interventions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_module()


def tool_call(name, call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": {}},
            }
        ],
    }


def tool_result(name, result, call_id):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result),
    }


def row(identifier, target, passage, text, terms):
    context = {
        "text": text,
        "passage_ids": [passage],
        "n_characters": len(text),
        "context_sha256": passage,
    }
    textbook = {
        "ok": True,
        "context": context,
        "matches": [
            {
                "passage_id": passage,
                "matched_terms": terms,
                "state_terms": terms,
            }
        ],
        "direct_reward": False,
    }
    anchors = {
        "ok": True,
        "matches": [
            {
                "primitive_id": passage,
                "warnings": ["warning survives competitor removal"],
                "competitors": ["competitor survives warning removal"],
            }
        ],
        "direct_reward": False,
    }
    return {
        "id": identifier,
        "target_smiles": target,
        "expected_precursor": target,
        "messages": [
            {"role": "system", "content": "Use evidence."},
            {"role": "user", "content": f"TARGET: {target}"},
            tool_call("retrieve_textbook_guidance", "t"),
            tool_result("retrieve_textbook_guidance", textbook, "t"),
            tool_call("retrieve_primitives", "a"),
            tool_result("retrieve_primitives", anchors, "a"),
        ],
        "metadata": {"textbook_context_characters": len(text)},
    }


def result(value, name):
    message = next(
        item
        for item in value["messages"]
        if item.get("role") == "tool" and item.get("name") == name
    )
    return json.loads(message["content"])


def rows():
    return [
        row(
            "r1",
            "[CH4:1]",
            "p1",
            "Substitution warning: elimination may compete.",
            ["substitution", "halide"],
        ),
        row(
            "r2",
            "[NH3:2]",
            "p2",
            "Alternative substitution pathway can compete.",
            ["substitution", "halide"],
        ),
    ]


def test_passage_shuffle_uses_a_different_donor_and_preserves_length():
    source = rows()
    shuffled = mod.passage_shuffle(source)
    for original, changed in zip(source, shuffled):
        assert changed["metadata"]["evidence_donor_id"] != original["id"]
        assert changed["metadata"]["textbook_context_characters"] == original["metadata"]["textbook_context_characters"]
        assert result(changed, "retrieve_textbook_guidance")["context"]["n_characters"] == original["metadata"]["textbook_context_characters"]


def test_same_topic_wrong_requires_shared_terms_and_different_passage():
    changed, quarantined = mod.same_topic_wrong(rows())
    assert quarantined == []
    assert changed[0]["metadata"]["evidence_donor_id"] == "r2"
    assert changed[0]["metadata"]["shared_retrieval_terms"]


def test_same_topic_wrong_quarantines_when_no_reviewed_donor_exists():
    single = [row("r1", "[CH4:1]", "p1", "Only text", ["unique"])]
    changed, quarantined = mod.same_topic_wrong(single)
    assert changed == []
    assert len(quarantined) == 1
    assert quarantined[0]["id"] == "r1"
    assert quarantined[0]["error_code"] == "NO_SAME_TOPIC_WRONG_PASSAGE_DONOR"


def test_warning_and_competitor_interventions_are_isolated():
    warning_removed = mod.remove_warnings(rows())[0]
    competitor_removed = mod.remove_competing_pathways(rows())[0]
    warning_anchor = result(warning_removed, "retrieve_primitives")["matches"][0]
    competitor_anchor = result(competitor_removed, "retrieve_primitives")["matches"][0]
    assert "warnings" not in warning_anchor
    assert warning_anchor["competitors"]
    assert "competitors" not in competitor_anchor
    assert competitor_anchor["warnings"]
