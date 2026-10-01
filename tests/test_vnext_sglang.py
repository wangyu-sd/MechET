import json

from mechet.electron_pointer import parse_pointer_observation
from scripts.benchmark_vnext_sglang import payload


def test_sglang_payload_changes_only_structural_constraint():
    obs = parse_pointer_observation("ANNOTATED CURRENT STATE: <A01>C<A02>O")
    base = payload("prefix", obs, mode="unconstrained", max_new_tokens=32)
    constrained = payload("prefix", obs, mode="inventory", max_new_tokens=32)
    assert base["text"] == constrained["text"] == "prefix"
    assert base["sampling_params"] == {"temperature": 0, "max_new_tokens": 32}
    schema = json.loads(constrained["sampling_params"]["json_schema"])
    assert "A01" in json.dumps(schema)
