from argparse import Namespace
from hashlib import sha256
import json
from pathlib import Path

import pytest


def write_ref(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": sha256(raw).hexdigest()}


def test_prepared_public_workers_bind_disjoint_batches_and_the_one_finalized_comparison(tmp_path, monkeypatch):
    from scripts import pr117_public_package as package

    monkeypatch.setattr(package, "CEPH", tmp_path)
    launch = tmp_path / "confirmation"
    inputs = launch / "inputs"
    inputs.mkdir(parents=True)
    parent_panel = write_ref(inputs / "panel.json", {"targets": [{"cohort": "linear", "target_id": "b"}]})
    registration = write_ref(inputs / "confirmation_registration.json", {
        "targets": 1, "arms": ["C0_original"], "actor_sha256": "parent", "panel_sha256": parent_panel["sha256"],
        "per_target_arm_request_cap": 64, "method_spec": {"sha256": "method"},
    })
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    (inputs / "runtime_contract.path").write_text(str(contract) + "\n")
    deployment = {"task_flag": "meteor_mechet_pr117_reference96_test",
                  "shape": {"host_num": 1, "host_gpu_num": 1, "GPUName": "A100"}}
    write_ref(inputs / "deployment.json", deployment)
    manifest = "".join(f"{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(launch)}\n" for p in sorted(inputs.iterdir()))
    (launch / "inputs.sha256").write_text(manifest)
    write_ref(launch / "prepared.json", {"task_flag": deployment["task_flag"],
              "manifest_sha256": sha256(manifest.encode()).hexdigest(),
              "confirmation_registration_sha256": registration["sha256"]})
    binding = {"targets": 1, "arms": ["C0_original"], "actor_sha256": "parent",
               "panel_sha256": parent_panel["sha256"], "per_target_arm_request_cap": 64,
               "method_spec_sha256": "method", "confirmation_registration_sha256": registration["sha256"]}
    audit = write_ref(tmp_path / "receipt_audit.json", {"binding": binding, "rows": [
        {"cohort": "linear", "target_id": "b", "arm": "C0_original", "gpu_requests": 2}]})
    finalizer = write_ref(tmp_path / "confirmation_evidence.json", {
        "status": "COMPLETE_REPLAY_VERIFIED", "expected_arms": 1, "completed_arms": 1, "fully_paired_targets": 1,
        "issued_native_calls": 2, "returned_native_calls": 2, "audited_native_calls": 2, "binding": binding,
        "artifact_sha256": {"receipt_audit.json": audit["sha256"]},
    })
    public_panel = write_ref(tmp_path / "panel.json", {"targets": [{"target_id": str(i)} for i in range(5)]})
    public = write_ref(tmp_path / "public_registration.json", {
        "phase": "reference_full_public", "targets": 5, "arms": ["C0_original"], "actor_sha256": "parent",
        "panel_sha256": public_panel["sha256"], "runtime_contract_sha256": sha256(contract.read_bytes()).hexdigest(),
        "per_target_arm_request_cap": 64, "total_request_ceiling": 320,
        "confirmation_finalizer": finalizer, "confirmation_audit": audit, "target_lock": {"sha256": "full-lock"},
    })
    plan = write_ref(tmp_path / "batch_plan.json", {"public_target_lock_sha256": "full-lock",
                     "registered_targets": 5, "target_indexes": [[0, 2, 4], [1, 3]]})

    def validate_command(command, **kwargs):
        if command[0] == "git":
            assert command[-2:] == ["rev-parse", "HEAD"]
            return "execution-commit\n"
        assert command[0] == "bash" and command[-1] == "--validate"
        box = Path(command[1]).parent
        batch = int((box / "inputs/public_batch_index.txt").read_text())
        size = {1: 3, 2: 2}[batch]
        return json.dumps({"binding": {
            "public_batch_index": batch-1, "full_population_targets": 5, "targets": size,
            "total_request_ceiling": size*64, "public_registration_sha256": public["sha256"],
            "public_batch_plan_sha256": plan["sha256"], "actor_sha256": "parent",
        }}).encode()

    monkeypatch.setattr(package.subprocess, "check_output", validate_command)
    output = tmp_path / "public-launch"
    args = Namespace(parent_launch=launch, public_registration=Path(public["path"]),
                     public_registration_sha256=public["sha256"], batch_plan=Path(plan["path"]),
                     batch_plan_sha256=plan["sha256"], output=output)
    receipt = package.prepare(args)
    assert receipt["targets"] == 5 and receipt["max_native_calls"] == 320
    assert receipt["gpu_jobs_submitted"] == 0
    assert [d["targets"] for d in receipt["batches"].values()] == [3, 2]
    for name in ("batch-01", "batch-02"):
        worker = output / name
        data = package.verified_deployment(worker)
        assert data["start_command"].endswith(f"{name}/worker.sh")
        assert (worker / "inputs/public_results_root.path").read_text().strip() == str(output / "results")
    with pytest.raises(FileExistsError):
        package.prepare(args)
