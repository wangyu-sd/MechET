import json

import pytest

from mechet.electron_pointer import parse_pointer_observation
from mechet.vnext_structured_actions import parse_structured_action, structured_action_schema


def test_inventory_schema_contains_current_handles_not_absent_handles():
    observation = parse_pointer_observation("ANNOTATED CURRENT STATE: <A01>C<A02>O.<A03>[Cl-]")
    schema = structured_action_schema(observation, inventory_handles=True)
    event = schema["oneOf"][1]["properties"]["arguments"]["properties"]["electron_flow"]["items"]
    source = event["properties"]["source"]["enum"]
    sink = event["properties"]["destination"]["enum"]
    assert "the bond between atoms A01 and A02" in source
    assert "the bond between atoms A01 and A03" not in source
    assert "the bond to form between atoms A01 and A03" in sink
    assert all("A99" not in phrase for phrase in source + sink)
    json.dumps(schema)


def test_schema_only_and_plain_json_envelope():
    schema = structured_action_schema(None, inventory_handles=False)
    source = schema["oneOf"][1]["properties"]["arguments"]["properties"]["electron_flow"]["items"]["properties"]["source"]
    assert source == {"type": "string"}
    assert parse_structured_action('{"name":"finish_trace","arguments":{}}') == ("finish_trace", {})
    with pytest.raises(ValueError):
        parse_structured_action('{"name":"finish_trace","arguments":{},"extra":1}')
    with pytest.raises(ValueError):
        parse_structured_action('{"name":"bad","arguments":{}}')


def test_vllm_array_compatibility_preserves_cardinality_after_decode():
    schema = structured_action_schema(None, inventory_handles=False)

    def arrays(value):
        if isinstance(value, dict):
            if value.get("type") == "array":
                yield value
            for nested in value.values():
                yield from arrays(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from arrays(nested)

    assert all("minItems" not in field and "maxItems" not in field for field in arrays(schema))
    with pytest.raises(ValueError, match="at least one fragment"):
        parse_structured_action('{"name":"import_fragments","arguments":{"fragments":[]}}')
    with pytest.raises(ValueError, match="exactly two atoms"):
        parse_structured_action('{"name":"apply_electron_flow","arguments":{"bond_order_changes":[{"atoms":["A01"]}]}}')
