import json
from pathlib import Path
import importlib.util
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/monitor_qwen3_task.py'
    spec=importlib.util.spec_from_file_location('qwen3_monitor',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_actual_end_failure_is_distinct_from_resource_wait_and_success():
    m=module()
    row='| 2026-11-14 12:10:00 | '+'a'*32+' | false | END | 25 | 2026-11-14 12:15:00 |'
    assert m.parse_instances(row)['success'] is False
    assert m.parse_instances(row)['platform_state']=='END'
    assert m.parse_instances(row.replace('false','true'))['success'] is True
    with pytest.raises(ValueError):m.parse_instances('API unavailable')


def test_running_platform_alone_does_not_claim_optimizer_updates(tmp_path):
    m=module();assert m.checkpoint_progress(tmp_path)['saved_optimizer_step'] is None
    directory=tmp_path/'checkpoint-500';directory.mkdir()
    assert m.checkpoint_progress(tmp_path)['saved_optimizer_step'] is None
    (directory/'trainer_state.json').write_text(json.dumps({'global_step':499}))
    result=m.checkpoint_progress(tmp_path)
    assert result['saved_optimizer_step']==499
    assert result['binding_manifest_present'] is False
