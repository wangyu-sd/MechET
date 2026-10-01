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
