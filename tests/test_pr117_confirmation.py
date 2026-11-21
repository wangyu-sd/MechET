import hashlib
import json
from pathlib import Path

import pytest

from scripts import pr117_confirmation as confirmation
from scripts import pr117_controllers as controllers


def pinned_file(path, value):
    payload = json.dumps(value).encode()
    path.write_bytes(payload)
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())


def registered_fixture(tmp_path):
    cohorts = ("ref-lin-600", "ref-cnv-400", "ref-lng-84")
    rows = [dict(cohort=cohort, target_id=str(i), target=f"mol-{cohort}-{i}",
                 stock="stock", seed=i, rank_sha256=f"rank-{cohort}-{i}")
            for i in range(32) for cohort in cohorts]
    inventory = pinned_file(tmp_path / "inventory.json", {"candidates": rows})
    stocks = {"stock": {"source_sha256": "stocks", "source_path": "/stocks", "size_after_executor_identity": 1}}
    rank = [{k: row[k] for k in ("cohort", "target_id", "rank_sha256")} for row in rows]
    targets = pinned_file(tmp_path / "targets.json", dict(status="INDEPENDENT_REFERENCE_TARGETS_LOCKED",
                         model_calls=0, source_inventory_sha256=inventory["sha256"], eligible_inventory_size=96,
                         stocks=stocks, primary_targets=96, targets=rows, ranked_candidate_ids=rank,
                         candidate_order_prefix_strata={"24": {cohort: 8 for cohort in cohorts},
                                                        "96": {cohort: 32 for cohort in cohorts}}))
    method = tmp_path / "method_spec.md"
    method.write_text("Frozen method fixture")
    arms = [controllers.C0, controllers.C2, controllers.C4]
    panel = dict(targets=rows[:24], stocks=stocks, actor={"adapter_sha256": "actor"}, arms=arms,
                 request_ceiling_per_target_arm=64)
    raw = json.dumps(panel).encode()
    reg = dict(phase="reference_confirmation", panel_sha256=hashlib.sha256(raw).hexdigest(),
               runtime_contract_sha256="runtime", actor_sha256="actor", targets=24, arms=arms,
               runner_source_sha256=confirmation.runner_source_hashes(),
               selected_arm=controllers.C2, per_target_arm_request_cap=64,
               total_request_ceiling=24 * 3 * 64, target_lock=targets, inventory=inventory,
               method_spec=dict(path=str(method), sha256=hashlib.sha256(method.read_bytes()).hexdigest()))
    return panel, reg


def test_confirmation_preserves_frozen_targets_and_budget(tmp_path):
    panel, reg = registered_fixture(tmp_path)
    result = confirmation.validate_registration(reg, panel, "runtime")
    assert result["targets"] == 24
    assert result["total_request_ceiling"] == 24 * 3 * 64
    assert result["primary_arm"] == controllers.C2


def test_changing_the_target_or_stock_is_rejected(tmp_path):
    panel, reg = registered_fixture(tmp_path)
    panel["targets"][0]["target"] = "different"
    with pytest.raises(ValueError, match="registered target prefix"):
        confirmation.validate_registration(reg, panel, "runtime")


def test_unapproved_cap_or_missing_reaction_first_control_is_rejected(tmp_path):
    panel, reg = registered_fixture(tmp_path)
    panel["arms"] = reg["arms"] = [controllers.C0, controllers.C2]
    with pytest.raises(ValueError, match="reaction-first"):
        confirmation.validate_registration(reg, panel, "runtime")


def test_target_lock_cannot_follow_an_edited_inventory(tmp_path):
    _, reg = registered_fixture(tmp_path)
    Path(reg["inventory"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="digest changed"):
        confirmation.read_target_lock(reg["target_lock"], reg["inventory"])


def test_execution_builder_binds_an_explicit_method_to_a_registered_prefix(tmp_path):
    panel, reg = registered_fixture(tmp_path)
    runtime = pinned_file(tmp_path / "runtime.json", {"actor_lineage": {"adapter_sha256": "actor"}, "seed": 17})
    encoded_panel, generated = confirmation.build_execution(
        reg["target_lock"], reg["inventory"], reg["method_spec"], runtime,
        targets=24, selected_arm=controllers.C2, cap=64)
    assert json.loads(encoded_panel)["targets"] == panel["targets"]
    assert generated["panel_sha256"] == hashlib.sha256(encoded_panel).hexdigest()
    assert generated["arms"] == panel["arms"]
    assert generated["total_request_ceiling"] == 24 * 3 * 64


def test_confirmation_cap_must_stay_within_protocol(tmp_path):
    panel, reg = registered_fixture(tmp_path)
    panel["request_ceiling_per_target_arm"] = reg["per_target_arm_request_cap"] = 129
    reg["total_request_ceiling"] = 24 * 3 * 129
    with pytest.raises(ValueError, match="request allowance"):
        confirmation.validate_registration(reg, panel, "runtime")


def test_changed_controller_revision_cannot_share_old_registration(tmp_path):
    panel, reg = registered_fixture(tmp_path)
    reg["runner_source_sha256"]["pr117_controllers.py"] = "different"
    with pytest.raises(ValueError, match="controller changed"):
        confirmation.validate_registration(reg, panel, "runtime")
