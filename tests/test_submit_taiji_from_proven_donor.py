import json
from pathlib import Path

import pytest

from scripts.submit_taiji_from_proven_donor import validate_template


ROOT = Path(__file__).resolve().parents[1]


def test_state_sft_template_uses_proven_h20_mount_and_live_logs():
    template = json.loads(
        (ROOT / "configs/taiji/meteor_mechet_reliable_state_06b_1ep_8h20_zjk_20261006.json").read_text()
    )
    donor = {
        "common": {"business_flag": "AI4LSLab_KEMO_ZJK_H20"},
        "designated_resource": {
            "GPUName": "H20",
            "image_full_name": "mirrors.tencent.com/whaleywang/metabo:taiji7",
        },
        "job_config": {"init_cmd": "mount private Ceph, then run"},
    }
    assert validate_template(template, donor) == donor["job_config"]["init_cmd"]
    template["start_cmd"] += " > hidden.log"
    with pytest.raises(ValueError, match="redirect"):
        validate_template(template, donor)
