import json

import pytest


def test_actual_json_training_entry_preserves_scientific_notation(tmp_path):
    from scripts.train_tool_sft import load_yaml
    from mechet.qwen3_continuation import continuation_config, validate_config
    expected=continuation_config(world_size=8)
    p=tmp_path/'condition.json';p.write_text(json.dumps(expected))
    actual=load_yaml(p)
    assert actual['training']['learning_rate']==5e-6
    assert isinstance(actual['training']['learning_rate'],float)
    validate_config(actual)
    assert actual==expected


def test_yaml_training_conditions_keep_their_existing_loader(tmp_path):
    from scripts.train_tool_sft import load_yaml
    p=tmp_path/'legacy.yaml';p.write_text('training:\n  learning_rate: 5.0e-6\n  bf16: true\n')
    assert load_yaml(p)=={'training':{'learning_rate':5e-6,'bf16':True}}


def test_json_filename_cannot_silently_accept_yaml_content(tmp_path):
    from scripts.train_tool_sft import load_yaml
    p=tmp_path/'condition.json';p.write_text('training: {learning_rate: 0.000005}')
    with pytest.raises(json.JSONDecodeError):load_yaml(p)
