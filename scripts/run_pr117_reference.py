"""Run PR117's preregistered Reference screen using one frozen native actor."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts import pr117_controllers as controllers
from scripts.pr117_actor_backend import loaded_actor

CEPH = Path("/aaa/fionafyang/buddy1")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, data):
    with path.open("x") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def verify_closure(target, stock, report):
    """Independently traverse committed occurrences and check their exact leaves."""
    if report["status"] != "stock_solved":
        return False
    if not report["reactions"] or target in stock:
        raise ValueError("a zero-step route cannot be counted")
    open_nodes = {"0": (target, ())}
    for reaction in report["reactions"]:
        occurrence = reaction["occurrence"]
        product, ancestors = open_nodes.pop(occurrence)
        if product != reaction["product"] or product in stock or not reaction["precursors"]:
            raise ValueError("invalid committed reaction")
        for i, child in enumerate(reaction["precursors"]):
            if child in (*ancestors, product):
                raise ValueError("committed ancestor cycle")
            open_nodes[f"{occurrence}.{i}"] = (child, (*ancestors, product))
    leaves = [product for product, _ in open_nodes.values()]
    if not leaves or any(leaf not in stock for leaf in leaves) or Counter(leaves) != Counter(report["stock_leaves"]):
        raise ValueError("stock obligations disagree with executor report")
    return True


def bind_runtime(args):
    cfg_raw = args.runtime_contract.read_bytes()
    cfg = json.loads(cfg_raw)
    runtime = Path(cfg["code_root"]).resolve()
    pins = cfg["code_files_sha256"]
    for name in ("route_policy_episode", "route_refit", "route_native_observation",
                 "route_vllm_backend", "route_vllm_protocol", "external_route_actor"):
        rel = f"src/mechet/{name}.py"
        if sha((runtime / rel).read_bytes()) != pins[rel]:
            raise ValueError(f"Runtime source differs: {rel}")
    sys.path.insert(0, str(runtime / "src"))
    sys.path.append(str(runtime))

    from mechet.external_route_actor import verify_stage2_actor
    from mechet.route_training_contract import chemical_identity
    actor = verify_stage2_actor(cfg["parent"])
    reg_path = getattr(args, "confirmation_registration", None)
    reg_sha = getattr(args, "confirmation_registration_sha256", None)
    if (reg_path is None) != (reg_sha is None):
        raise ValueError("Confirmation requires both the registration and its SHA256")
    if reg_path is None:
        lock_file = Path(__file__).resolve().parents[1] / "docs/goals/PR117_INITIAL_REFERENCE_PANEL_LOCK.json"
        lock = json.loads(lock_file.read_bytes())
    else:
        from scripts.pr117_confirmation import read_pinned
        lock = json.loads(read_pinned({"path": reg_path, "sha256": reg_sha}))
    payload = args.panel.read_bytes()
    if sha(payload) != lock["panel_sha256"]:
        raise ValueError("Reference panel changed after preregistration")
    panel = json.loads(payload)
    if actor["adapter_sha256"] != lock["actor_sha256"] or actor["adapter_sha256"] != panel["actor"]["adapter_sha256"]:
        raise ValueError("Wrong frozen actor")
    cap = panel["request_ceiling_per_target_arm"]
    if reg_path is None:
        arms = (controllers.C0, controllers.C1, controllers.C2, controllers.C4)
        if tuple(panel["arms"]) != arms or len(panel["targets"]) != lock["targets"]:
            raise ValueError("Panel arm or population mismatch")
        if len(arms) * len(panel["targets"]) * cap != lock["total_primary_screen_cap"]:
            raise ValueError("Panel request ceiling mismatch")
        scope = dict(targets=len(panel["targets"]), arms=list(arms), total_request_ceiling=lock["total_primary_screen_cap"])
    else:
        from scripts.pr117_confirmation import validate_registration
        scope = validate_registration(lock, panel, sha(cfg_raw))
        scope["confirmation_registration_sha256"] = reg_sha
    stocks = {}
    for name, info in panel["stocks"].items():
        raw = Path(info["source_path"]).read_bytes()
        if sha(raw) != info["source_sha256"]:
            raise ValueError(f"Official stock changed: {name}")
        stocks[name] = {chemical_identity(s) for s in gzip.decompress(raw).decode().splitlines() if s}
        if len(stocks[name]) != info["size_after_executor_identity"]:
            raise ValueError(f"Stock identity projection changed: {name}")
    for row in panel["targets"]:
        if row["target"] in stocks[row["stock"]]:
            raise ValueError("Frozen evaluation target must require synthesis")
    verification = dict(panel_sha256=sha(payload), runtime_contract_sha256=sha(cfg_raw),
                        actor_sha256=actor["adapter_sha256"], per_target_arm_request_cap=cap,
                        stocks={n: len(s) for n,s in stocks.items()}, **scope)
    return cfg, panel, stocks, verification


def admit_gpu(receipt_file):
    """Use a separately allocated GPU recorded by the platform."""
    if receipt_file is None:
        raise ValueError("A scheduler allocation receipt is required for live inference")
    receipt = json.loads(receipt_file.read_bytes())
    if (not receipt["task_flag"].startswith("meteor_mechet_pr117_")
            or not receipt["instance_id"] or receipt["hostname"] != socket.gethostname()):
        raise ValueError("Allocation is not bound to the PR117 worker")
    uuid = receipt["gpu_uuid"]
    if not uuid.startswith("GPU-"):
        raise ValueError("Expected an explicit allocated GPU UUID")
    command = ["nvidia-smi", "-i", uuid]
    actual = subprocess.check_output(
        command + ["--query-gpu=uuid,name", "--format=csv,noheader"], text=True).strip()
    gpu_uuid, model = map(str.strip, actual.split(",", 1))
    if gpu_uuid != uuid or not any(device in model for device in ("H20", "A100")):
        raise ValueError("A separate H20/A100 allocation is required by PR117")
    process_report = subprocess.check_output(
        command + ["--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if process_report:
        raise ValueError("The allocated GPU already has active compute processes")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return dict(receipt, gpu_model=model, scheduler_receipt_sha256=sha(receipt_file.read_bytes()))


def run_live(args, cfg, panel, stocks, verification):
    from mechet.route_action_feedback import semantic_action
    from mechet.route_policy_episode import NativeChemistry
    from mechet.route_policy_sampling import read_owned_journal
    from mechet.route_refit import RefitRouteEpisode, grounded_action_key

    def action(raw):
        parsed = semantic_action(raw)
        return parsed[1]["name"] if parsed else None

    class CountedChemistry(NativeChemistry):
        def __init__(self):
            super().__init__()
            self.executor_calls = 0

        def step(self, node, receipt):
            self.executor_calls += 1
            return super().step(node, receipt)

    output = args.output.resolve()
    if not output.is_relative_to(CEPH):
        raise ValueError("PR117 output must reside on shared storage")
    allocation = admit_gpu(args.allocation_receipt)
    output.mkdir(exist_ok=False)
    analysis_scope = verification.get("scope", "exploratory preregistered screen")
    write_new(output / "admission.json", dict(verification, allocation=allocation,
              chemistry_teacher="off for all paired arms", inference_backend=args.backend,
              analysis=analysis_scope))
    (output / "targets").mkdir()
    cfg = dict(cfg, output_dir=str(output), _sha=verification["panel_sha256"])
    cfg["vllm_runtime"] = dict(cfg["vllm_runtime"], cache_root=str(output / "engine-caches"))
    cfg["vllm_runtime"].pop("shared_triton_cache_root", None)
    # The one-device inference pilot has fewer active requests than the
    # original H20 training role. Keep its cache and concurrency isolated.
    cfg["vllm_runtime"]["kv_cache_memory_bytes"] = min(cfg["vllm_runtime"]["kv_cache_memory_bytes"], 4 << 30)
    if args.vllm_native_driver:
        from scripts.pr117_native_cuda import bind_native_driver
        cfg["vllm_runtime"], driver = bind_native_driver(cfg["vllm_runtime"], output, allocation["gpu_uuid"])
        write_new(output / "native_driver_receipt.json", driver)
    cfg["generation_batch_size"] = args.workers
    cfg["seed"] = panel["seed"]
    version = "validation000000:" + verification["actor_sha256"]
    cfg["behavior_version"] = version
    route_context_episode = controllers.contextual_episode_type(RefitRouteEpisode)
    began = time.monotonic()
    with loaded_actor(args.backend, cfg, output, version) as (sampler, journal_path, dispatched_event):
        def solve_target(row):
            answers = []
            arms = panel["arms"]
            order = row["seed"] % len(arms)
            for arm in arms[order:] + arms[:order]:
                target = row["target"]
                stock = stocks[row["stock"]]
                uid = f"pr117/{row['cohort']}/{row['target_id']}/{row['seed']}/{arm}"
                chemistry = CountedChemistry()
                episode_type = RefitRouteEpisode if arm == controllers.C0 else route_context_episode
                episode = episode_type(cfg, target, stock, uid, row["seed"], chemistry=chemistry)
                start = time.monotonic()
                result = controllers.solve(
                    arm, episode, lambda request: sampler.submit(request).result(),
                    cap=panel["request_ceiling_per_target_arm"], action=action, grounded_key=grounded_action_key)
                # The controller owns budget censoring; the Executor owns chemistry.
                solved = result["status"] == "stock_solved" and verify_closure(target, stock, result["report"])
                tokens = result["generated_tokens"]
                value = dict(target_id=row["target_id"], target=target, cohort=row["cohort"],
                             stock=row["stock"], arm=arm, uid=uid, solved=solved, status=result["status"],
                             request_attempts=result["requests"], generated_tokens=tokens,
                             executor_calls=chemistry.executor_calls, seconds=time.monotonic()-start)
                write_new(output / "targets" / f"{row['cohort']}-{row['target_id']}-{arm}.json",
                          dict(value, result=result))
                answers.append(value)
            return answers

        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            rows = [row for result in workers.map(solve_target, panel["targets"]) for row in result]
    duration = time.monotonic()-began
    native = Counter()
    for event in read_owned_journal(journal_path):
        if event["event"] != dispatched_event:
            continue
        requests = event["requests"] if args.backend == "hf" else [event["request"]]
        for request in requests:
            request_id = request["request_id"]
            native[request_id.split("/r0/a",1)[0].split("/pr117-",1)[0]] += 1
    assert len(rows) == verification["targets"] * len(verification["arms"])
    assert sum(native.values()) <= sum(row["request_attempts"] for row in rows) <= verification["total_request_ceiling"]
    for row in rows:
        row["native_generation_requests"] = native[row["uid"]]
    totals = {arm: sum(r["solved"] for r in rows if r["arm"] == arm) for arm in verification["arms"]}
    write_new(output / "screen_result.json", dict(verification, solved=totals, rows=rows,
              actual_native_generation_requests=sum(native.values()), single_gpu_wall_seconds=duration,
              gpu_hours=duration/3600, scope=analysis_scope))
    print(json.dumps(dict(denominator=verification["targets"], solved=totals,
                          actual_native_generation_requests=sum(native.values()), gpu_hours=duration/3600)))
    primary = verification.get("primary_arm", controllers.C2)
    print(f"METRIC solved_targets={totals[primary]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-contract", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--confirmation-registration", type=Path)
    parser.add_argument("--confirmation-registration-sha256")
    parser.add_argument("--validate", action="store_true", help="verify inputs without loading the model or GPU")
    parser.add_argument("--allocation-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--vllm-native-driver", action="store_true",
                        help="bind the allocated R580+ host driver to the fixed vLLM CUDA 12.9 runtime")
    args = parser.parse_args()
    if args.vllm_native_driver and args.backend != "vllm":
        parser.error("--vllm-native-driver requires --backend vllm")
    cfg, panel, stocks, verification = bind_runtime(args)
    if args.validate:
        print(json.dumps(verification))
        return
    if args.output is None or not 1 <= args.workers <= 16:
        parser.error("live inference needs --output and 1–16 executor workers")
    run_live(args, cfg, panel, stocks, verification)


if __name__ == "__main__":
    main()
