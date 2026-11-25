from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import pr117_controllers as controllers
from scripts import pr117_public_registration as public
from scripts.pr117_confirmation import digest
from scripts.pr117_reference_source import PUBLIC_COHORTS


def write_ref(path, value):
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": digest(raw)}


def inputs(tmp_path):
    rows = [dict(cohort=c, target_id=str(i), target=f"m-{c}-{i}", stock="stock", seed=i)
            for c, n in PUBLIC_COHORTS.items() for i in range(n)]
    method = write_ref(tmp_path / "method.md", b"Frozen method fixture")
    actors = dict(adapter_sha256="parent")
    source = write_ref(tmp_path / "targets.json", dict(status="FULL_PUBLIC_REFERENCE_TARGETS_LOCKED",
        total_targets=1084, model_calls=0, seed=17, targets=rows,
        stocks={"stock": dict(source_path="source", source_sha256="stock", size_after_executor_identity=1)},
        cohorts={k: {"targets": n, "stock": "stock"} for k, n in PUBLIC_COHORTS.items()}))
    audit = write_ref(tmp_path / "confirmation.json", dict(
        fully_audited=True, final_summary_verified=True, audited=288, expected=288, fully_paired_targets=96,
        binding=dict(primary_arm=controllers.C2, targets=96, arms=[controllers.C0, controllers.C2, controllers.C4],
                     method_spec_sha256=method["sha256"], confirmation_registration_sha256="original registration")))
    runtime = write_ref(tmp_path / "contract.json", dict(actor_lineage=actors, seed=17, output_dir=str(tmp_path)))
    return source, method, audit, runtime


def test_full_public_registration_binds_the_exact_population_and_method(tmp_path):
    source, method, audit, runtime = inputs(tmp_path)
    payload, registration = public.build_execution(source, method, audit, runtime, cap=30)
    panel = json.loads(payload)
    admitted = public.validate_registration(registration, panel, runtime["sha256"])
    assert admitted["targets"] == 1084
    assert admitted["total_request_ceiling"] == 32520
    assert admitted["arms"] == [controllers.C2]
    assert all(set(r) == {"cohort", "target_id", "target", "stock", "seed"} for r in panel["targets"])
    assert registration["panel_sha256"] == digest(payload)


def test_partial_confirmation_cannot_register_public_evaluation(tmp_path):
    source, method, audit, runtime = inputs(tmp_path)
    value = json.loads(Path(audit["path"]).read_bytes())
    value["final_summary_verified"] = False
    audit = write_ref(Path(audit["path"]), value)
    with pytest.raises(ValueError, match="completed independent confirmation"):
        public.build_execution(source, method, audit, runtime, cap=30)


def test_public_fallback_keeps_the_three_condition_confirmation_contract(tmp_path):
    source, method, audit, runtime = inputs(tmp_path)
    value = json.loads(Path(audit["path"]).read_bytes())
    value["binding"]["primary_arm"] = controllers.C0
    audit = write_ref(Path(audit["path"]), value)
    _, registration = public.build_execution(source, method, audit, runtime, cap=30)
    assert registration["arms"] == [controllers.C0]
    value["binding"]["arms"] = [controllers.C0, controllers.C4]
    value["audited"] = value["expected"] = 192
    audit = write_ref(Path(audit["path"]), value)
    with pytest.raises(ValueError, match="completed independent confirmation"):
        public.build_execution(source, method, audit, runtime, cap=30)


def test_truncated_or_substituted_public_panel_is_rejected(tmp_path):
    source, method, audit, runtime = inputs(tmp_path)
    payload, registration = public.build_execution(source, method, audit, runtime, cap=30)
    panel = json.loads(payload)
    shortened = deepcopy(panel)
    shortened["targets"].pop()
    with pytest.raises(ValueError, match="all official targets"):
        public.validate_registration(registration, shortened, runtime["sha256"])
    panel["targets"][71]["stock"] = "other"
    with pytest.raises(ValueError, match="all official targets"):
        public.validate_registration(registration, panel, runtime["sha256"])


def test_trained_actor_requires_a_completed_original_bound_update(tmp_path):
    source, method, audit, runtime = inputs(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "round-000019"
    checkpoint.mkdir(parents=True)
    weight = write_ref(checkpoint / "adapter_model.safetensors", b"adapter")
    cfg = write_ref(checkpoint / "adapter_config.json", b"config")
    done = dict(optimizer_step=19, contract_sha256=runtime["sha256"],
                payload_sha256={"adapter_model.safetensors": weight["sha256"], "adapter_config.json": cfg["sha256"]},
                round_metrics=[{"optimizer_updated": True}])
    complete = write_ref(checkpoint / "complete.json", done)
    payload, registration = public.build_execution(source, method, audit, runtime, cap=30, checkpoint=complete)
    assert json.loads(payload)["actor"]["adapter_sha256"] == weight["sha256"]
    assert public.trained_actor(registration["trained_checkpoint"], runtime["sha256"], tmp_path)["checkpoint"] == str(checkpoint)
    Path(weight["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        public.trained_actor(complete, runtime["sha256"], tmp_path)


def test_finalized_comparison_handoff_requires_registered_rows_and_receipts(tmp_path):
    launch = tmp_path / "launch"
    inputs_folder = launch / "inputs"
    inputs_folder.mkdir(parents=True)
    frozen_rows = [dict(cohort="linear", target_id="one"), dict(cohort="long", target_id="two")]
    panel = write_ref(inputs_folder / "panel.json", {"targets": frozen_rows})
    arm = controllers.C0
    registration = write_ref(inputs_folder / "confirmation_registration.json", dict(
        targets=2, arms=[arm], actor_sha256="parent", panel_sha256=panel["sha256"],
        per_target_arm_request_cap=64, method_spec={"sha256": "method"}))
    write_ref(launch / "prepared.json", {"confirmation_registration_sha256": registration["sha256"]})
    binding = dict(targets=2, arms=[arm], actor_sha256="parent", panel_sha256=panel["sha256"],
                   confirmation_registration_sha256=registration["sha256"], method_spec_sha256="method",
                   per_target_arm_request_cap=64)
    out = tmp_path / "evidence"
    out.mkdir()
    rows = [dict(row, arm=arm, gpu_requests=3) for row in frozen_rows]
    audit = write_ref(out / "receipt_audit.json", dict(binding=binding, rows=rows))
    snapshot = dict(status="COMPLETE_REPLAY_VERIFIED", binding=binding, expected_arms=2, completed_arms=2,
                    fully_paired_targets=2, issued_native_calls=6, returned_native_calls=6, audited_native_calls=6,
                    artifact_sha256={"receipt_audit.json": audit["sha256"]})
    record = write_ref(out / "evidence.json", snapshot)
    assert public.finalized_confirmation_audit(launch, record) == audit
    fake_rows = [rows[0], rows[0]]
    changed = write_ref(out / "receipt_audit.json", dict(binding=binding, rows=fake_rows))
    snapshot["artifact_sha256"]["receipt_audit.json"] = changed["sha256"]
    with pytest.raises(ValueError, match="registered target arms"):
        public.finalized_confirmation_audit(launch, write_ref(out / "evidence.json", snapshot))


def test_trained_public_activation_uses_the_exact_paired_checkpoint(tmp_path):
    source, method, audit, runtime = inputs(tmp_path)
    confirmation = json.loads(Path(audit["path"]).read_bytes())
    confirmation["binding"].update(primary_arm=controllers.C0, actor_sha256="parent", panel_sha256="panel",
                                   runtime_contract_sha256=runtime["sha256"], per_target_arm_request_cap=64,
                                   confirmation_target_lock_sha256="paired-targets", stocks={"stock": 1})
    cohort = "linear"
    confirmation["rows"] = [dict(cohort=cohort, target_id=str(i), arm=arm)
                            for i in range(96) for arm in confirmation["binding"]["arms"]]
    audit = write_ref(Path(audit["path"]), confirmation)

    checkpoint = tmp_path / "checkpoints" / "round-000019"
    checkpoint.mkdir(parents=True)
    adapter = write_ref(checkpoint / "adapter_model.safetensors", b"trained")
    cfg = write_ref(checkpoint / "adapter_config.json", b"configuration")
    complete = write_ref(checkpoint / "complete.json", dict(
        optimizer_step=19, contract_sha256=runtime["sha256"],
        payload_sha256={"adapter_model.safetensors": adapter["sha256"], "adapter_config.json": cfg["sha256"]},
        round_metrics=[{"optimizer_updated": True}]))
    payload, registration = public.build_execution(source, method, audit, runtime, cap=64, checkpoint=complete)
    panel = json.loads(payload)
    registration["confirmation_finalizer"] = {"sha256": "sealed-finalizer"}
    with pytest.raises(ValueError, match="completed paired actor evaluation"):
        public.validate_registration(registration, panel, runtime["sha256"])

    matched = confirmation["binding"]
    learned = dict(
        primary_arm=controllers.C0, arms=[controllers.C0], targets=96, per_target_arm_request_cap=64,
        runtime_contract_sha256=runtime["sha256"], method_spec_sha256=method["sha256"],
        parent_actor_sha256=matched["actor_sha256"], parent_panel_sha256=matched["panel_sha256"],
        parent_confirmation_registration_sha256=matched["confirmation_registration_sha256"],
        confirmation_target_lock_sha256=matched["confirmation_target_lock_sha256"], stocks={"stock": 1},
        actor_sha256=adapter["sha256"], trained_checkpoint_complete_sha256=complete["sha256"], optimizer_step=19,
    )
    paired = dict(parent=dict(binding=matched, final_summary_verified=True),
                  trained=dict(binding=learned, final_summary_verified=True),
                  target_pairs=96, same_initial_native_input_pairs=96, rows=[])
    for i in range(96):
        first = dict(first_request_id=f"r{i}", first_prompt_sha256=f"p{i}")
        paired["rows"].append(dict(cohort=cohort, target_id=str(i), parent=first, trained=dict(first)))
    ref = write_ref(tmp_path / "paired.json", paired)
    registration["paired_actor_evidence"] = ref
    admitted = public.validate_registration(registration, panel, runtime["sha256"])
    assert admitted["paired_actor_evidence_sha256"] == ref["sha256"]

    learned["trained_checkpoint_complete_sha256"] = "other-complete"
    registration["paired_actor_evidence"] = write_ref(tmp_path / "other-paired.json", paired)
    with pytest.raises(ValueError, match="checkpoint"):
        public.validate_registration(registration, panel, runtime["sha256"])

    learned["trained_checkpoint_complete_sha256"] = complete["sha256"]
    paired["rows"][-1]["trained"]["first_prompt_sha256"] = "different"
    registration["paired_actor_evidence"] = write_ref(tmp_path / "other-input.json", paired)
    with pytest.raises(ValueError, match="matched native inputs"):
        public.validate_registration(registration, panel, runtime["sha256"])
