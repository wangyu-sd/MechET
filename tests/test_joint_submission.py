import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_joint_launcher_rejects_commit_drift_before_any_model_or_job():
    path = ROOT/'scripts/run_taiji_joint_import_flow_a100.sh'
    assert path.is_file(), 'joint launcher not implemented'
    import os
    env = dict(os.environ, MECHET_EXPECTED_COMMIT='0'*40,
        MECHET_JOINT_CODE_ROOT=str(ROOT))
    result = subprocess.run(['bash', str(path)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'commit mismatch' in result.stderr


def test_joint_task_uses_successful_tencent_donor_group_and_default_logs():
    path = ROOT/'configs/taiji/meteor_mechet_joint_import_flow_full_1a100_qy_20261013_01.json'
    assert path.is_file(), 'joint task config not implemented'
    config = json.loads(path.read_text())
    assert config['host_num']*config['host_gpu_num'] == 1
    assert config['business_flag'] == 'AI4LSLab_KEMO_QY_A100'
    assert config['location'] == 'qy'
    assert config['image_full_name'] == 'mirrors.tencent.com/whaleywang/metabo:taiji6'
    assert config['init_cmd'] == 'REPLACE_WITH_PRIVATE_INIT_CMD_FROM_SUCCESSFUL_TASK'
    assert config['task_flag'].startswith('meteor') and not config['is_elasticity']
    assert config['is_mount_ceph']
    assert 'taiji_run_with_heartbeat.sh' in config['start_cmd']
    assert 'TAIJI_HEARTBEAT_SECONDS=60' in config['start_cmd']
    assert '>' not in config['start_cmd']
    assert '257167' in config['task_description']
