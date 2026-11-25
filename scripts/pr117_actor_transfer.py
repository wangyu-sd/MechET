"""Run a genuine trained adapter on the already registered independent targets."""
import argparse
import json
from pathlib import Path

from scripts.pr117_controllers import C0
from scripts.pr117_public_registration import trained_actor
from scripts.run_pr117_reference import bind_runtime, run_live, sha


def matched_actor_panel(baseline, primary_arm, adapter_sha):
    if primary_arm != C0 or primary_arm not in baseline["arms"]:
        raise ValueError("The trained and parent conditions must use the same controller for independent routes")
    if adapter_sha == baseline["actor"]["adapter_sha256"]:
        raise ValueError("The trained actor must use a distinct adapter checkpoint")
    return dict(baseline, actor=dict(adapter_sha256=adapter_sha), arms=[primary_arm])


def normalized_adapter_config(config):
    modules = config["target_modules"]
    return dict(config, target_modules=sorted(modules) if isinstance(modules, list) else modules)


def tensor_schema(path):
    from safetensors import safe_open

    with safe_open(str(path / "adapter_model.safetensors"), framework="numpy", device="cpu") as f:
        schema = {}
        for key in sorted(f.keys()):
            record = f.get_slice(key)
            schema[key] = {"dtype": record.get_dtype(), "shape": record.get_shape()}
    return schema


def bind_trained_comparison(args):
    cfg, baseline, stocks, original = bind_runtime(args)
    reference = dict(path=str(args.trained_checkpoint.resolve()), sha256=args.trained_checkpoint_sha256)
    trained = trained_actor(reference, original["runtime_contract_sha256"], cfg["output_dir"],
                            cfg.get("role_partition", {}).get("learner", 0))
    arm = original["primary_arm"]
    panel = matched_actor_panel(baseline, arm, trained["adapter_sha256"])
    baseline_path, trained_path = Path(cfg["parent"]), Path(trained["checkpoint"])
    baseline_shape = tensor_schema(baseline_path)
    trained_shape = tensor_schema(trained_path)
    baseline_cfg = normalized_adapter_config(json.loads((baseline_path / "adapter_config.json").read_bytes()))
    trained_cfg = normalized_adapter_config(json.loads((trained_path / "adapter_config.json").read_bytes()))
    if not baseline_shape or baseline_shape != trained_shape or baseline_cfg != trained_cfg:
        raise ValueError("Trained actor and original actor use different adapter semantics")

    cfg = dict(cfg, parent=trained["checkpoint"], parent_sha256=trained["adapter_sha256"])
    cap = original["per_target_arm_request_cap"]
    proof = dict(
        scope="trained actor with the original independent-route controller on paired held-out Reference targets",
        actor_condition="trained_refit", panel_sha256=sha(json.dumps(panel, sort_keys=True).encode()),
        actor_sha256=trained["adapter_sha256"], parent_actor_sha256=original["actor_sha256"],
        parent_panel_sha256=original["panel_sha256"], runtime_contract_sha256=original["runtime_contract_sha256"],
        confirmation_target_lock_sha256=original["confirmation_target_lock_sha256"],
        method_spec_sha256=original["method_spec_sha256"],
        parent_confirmation_registration_sha256=original["confirmation_registration_sha256"],
        optimizer_step=trained["optimizer_step"], trained_checkpoint_complete_sha256=reference["sha256"],
        adapter_config_sha256=trained["adapter_files"]["adapter_config.json"],
        adapter_tensor_schema_sha256=sha(json.dumps(trained_shape, sort_keys=True).encode()),
        adapter_tensors=len(trained_shape), arms=[arm], primary_arm=arm, targets=original["targets"],
        per_target_arm_request_cap=cap, total_request_ceiling=original["targets"]*cap, stocks=original["stocks"],
        sampling_seed=panel["seed"], inference_workers=6,
    )
    return cfg, panel, stocks, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-contract", "panel", "confirmation-registration", "trained-checkpoint"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("confirmation-registration-sha256", "trained-checkpoint-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--allocation-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.backend = "hf"
    args.vllm_native_driver = False
    args.workers = 6
    cfg, panel, stocks, proof = bind_trained_comparison(args)
    if args.validate:
        print(json.dumps(proof))
        return
    if args.allocation_receipt is None or args.output is None:
        parser.error("Live inference requires a new output and a separate GPU allocation receipt")
    run_live(args, cfg, panel, stocks, proof)


if __name__ == "__main__":
    main()
