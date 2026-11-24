"""Generate and validate the Day 3-4 8xH20 all-public-test evaluation & elevation manifest (Reference 1084, URSA 200, CbyD 240, SAR 40, AI Dark Space 250)."""
import argparse
import hashlib
import json
from pathlib import Path

PUBLIC_DATASETS = {
    "reference_1084": {
        "description": "RetroCast Reference 1,084 full public test set (Linear 600 / Convergent 400 / Long 84)",
        "targets_lock": "/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/diagnostics/retroe_pr117_autoresearch_20261121/run-20261121T201833-RxzzqSNC/reference_1084_full_target_lock_20261121.json",
        "stock": "/aaa/fionafyang/buddy1/whaleywang/MechET/data/route_benchmarks/frozen_r1_r2_20261013/stocks/buyables_publication.txt",
        "figures": ["Fig2a", "Fig2b", "Fig3c", "Fig3d", "Fig3e", "Fig4a", "Fig4b", "Fig4c", "Fig4d", "Fig5a", "Fig6a", "Fig6b"],
    },
    "ursa_expert100": {
        "description": "URSA-expert-2026 (100 expert-curated medicinal chemistry targets)",
        "targets_lock": "/aaa/fionafyang/buddy1/whaleywang/MechET/data/route_benchmarks/frozen_r1_r2_20261013/ursa_expert100/targets.jsonl",
        "stock": "/aaa/fionafyang/buddy1/whaleywang/MechET/data/external_benchmarks/r1r2_stock_rdkit2026_20261103_01/ursa_bbs_v1.guarded.stock.smi",
        "figures": ["Fig2c", "Fig2d", "Fig2e", "Fig3c", "Fig6a", "Fig6b"],
    },
    "ursa_drugs100": {
        "description": "URSA-drugs-clinicals-2026 (100 approved/clinical drugs)",
        "targets_lock": "/aaa/fionafyang/buddy1/whaleywang/MechET/data/route_benchmarks/frozen_r1_r2_20261013/ursa_drugs100/targets.jsonl",
        "stock": "/aaa/fionafyang/buddy1/whaleywang/MechET/data/external_benchmarks/r1r2_stock_rdkit2026_20261103_01/ursa_bbs_v1.guarded.stock.smi",
        "figures": ["Fig2c", "Fig2d", "Fig2e", "Fig3c", "Fig6a", "Fig6b"],
    },
    "cbyd_240": {
        "description": "CbyD 240 complex natural products stratified 80/80/80 across LLS 5-9 / 10-14 / >=15",
        "targets_lock": "/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/diagnostics/nmi_day34_eval_suite_20261124/frozen_elevation_panels/cbyd_stratified_240.jsonl",
        "stock": "/aaa/fionafyang/buddy1/whaleywang/MechET/data/route_benchmarks/frozen_r1_r2_20261013/stocks/buyables_publication.txt",
        "figures": ["Fig5a", "Fig5b", "Fig5c", "Fig5d", "Fig5e"],
    },
    "steerable_sar_40": {
        "description": "Elevation (4): 5 clinical drug scaffolds x 8 analogs = 40 molecules for constraint steering & common-intermediate SAR divergence",
        "targets_lock": "/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/diagnostics/nmi_day34_eval_suite_20261124/frozen_elevation_panels/steerable_sar_40.jsonl",
        "stock": "/aaa/fionafyang/buddy1/whaleywang/MechET/data/external_benchmarks/r1r2_stock_rdkit2026_20261103_01/ursa_bbs_v1.guarded.stock.smi",
        "figures": ["Fig6c", "ED_Fig4"],
    },
}


def build_day34_queue(
    *,
    sft_checkpoint: str,
    short_horizon_checkpoint: str,
    retroe_plan_checkpoint: str,
    out_dir: Path,
):
    """Build the full Day 3-4 8xH20 all-actor evaluation queue and verify every frozen public asset exists."""
    models = {
        "stage2_sft": sft_checkpoint,
        "short_horizon_204": short_horizon_checkpoint,
        "retroe_plan": retroe_plan_checkpoint,
    }
    verified_datasets = {}
    for name, spec in PUBLIC_DATASETS.items():
        tp = Path(spec["targets_lock"])
        sp = Path(spec["stock"])
        verified_datasets[name] = {
            **spec,
            "targets_exists": tp.exists(),
            "targets_sha256": hashlib.sha256(tp.read_bytes()).hexdigest() if tp.exists() else None,
            "stock_exists": sp.exists(),
        }

    jobs = [
        # Day 3A: Reference 1,084 three-stage model ablation (8xH20 all-actor)
        {"job_id": "d3_ref1084_sft_c2", "dataset": "reference_1084", "model": "stage2_sft", "controller": "C2_electron_backjump", "budget": 256, "split_role": "public_test"},
        {"job_id": "d3_ref1084_204_c2", "dataset": "reference_1084", "model": "short_horizon_204", "controller": "C2_electron_backjump", "budget": 256, "split_role": "public_test"},
        {"job_id": "d3_ref1084_plan_c2", "dataset": "reference_1084", "model": "retroe_plan", "controller": "C2_electron_backjump", "budget": 256, "split_role": "public_test"},
        # Day 3B: Reference 1,084 full controller pairing (C0 vs C4 vs C2) + Elevation (1) Test-Time Scaling B=8..256
        {"job_id": "d3_ref1084_plan_c0", "dataset": "reference_1084", "model": "retroe_plan", "controller": "C0_native_independent", "budget": 256, "split_role": "public_test"},
        {"job_id": "d3_ref1084_plan_c4", "dataset": "reference_1084", "model": "retroe_plan", "controller": "C4_reaction_level", "budget": 256, "split_role": "public_test"},
        # Day 4A: URSA Expert 100 + Drugs 100 (followed by official ChemCensor + 8-GPU FlowER forward cross-validation)
        {"job_id": "d4_ursa_expert100_plan_c2", "dataset": "ursa_expert100", "model": "retroe_plan", "controller": "C2_electron_backjump", "budget": 256, "split_role": "public_test"},
        {"job_id": "d4_ursa_drugs100_plan_c2", "dataset": "ursa_drugs100", "model": "retroe_plan", "controller": "C2_electron_backjump", "budget": 256, "split_role": "public_test"},
        {"job_id": "d4_ursa_expert100_204_c2", "dataset": "ursa_expert100", "model": "short_horizon_204", "controller": "C2_electron_backjump", "budget": 256, "split_role": "public_test"},
        {"job_id": "d4_ursa_drugs100_204_c2", "dataset": "ursa_drugs100", "model": "short_horizon_204", "controller": "C2_electron_backjump", "budget": 256, "split_role": "public_test"},
        # Day 4B: CbyD 240 complex natural products + Elevation (4) SAR-40 steerable planning
        {"job_id": "d4_cbyd240_plan_c2", "dataset": "cbyd_240", "model": "retroe_plan", "controller": "C2_electron_backjump", "budget": 256, "split_role": "public_test"},
        {"job_id": "d4_sar40_plan_steerable", "dataset": "steerable_sar_40", "model": "retroe_plan", "controller": "C2_steerable_sar", "budget": 256, "split_role": "public_test"},
    ]

    manifest = {
        "policy": "100% public_test only for main paper Figures 2-6; zero internal_valid leakage",
        "models": models,
        "datasets": verified_datasets,
        "jobs": jobs,
        "post_analysis_pipeline": [
            "scripts/prepare_nmi_day34_eval_suite.py --split-role public_test",
            "scripts/export_retroe_to_ursa.py",
            "scripts/analyze_emergent_strategy.py --budgets 8,16,32,64,128,256",
            "URSA ChemCensor v1.4.0 (Ceph venv) + 8-GPU FlowER forward cross-validation",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "day34_8h20_eval_queue.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-ckpt", required=True)
    parser.add_argument("--short-ckpt", required=True)
    parser.add_argument("--plan-ckpt", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_day34_queue(
        sft_checkpoint=args.sft_ckpt,
        short_horizon_checkpoint=args.short_ckpt,
        retroe_plan_checkpoint=args.plan_ckpt,
        out_dir=args.out_dir,
    )
    missing = [k for k, v in manifest["datasets"].items() if not v["targets_exists"] or not v["stock_exists"]]
    print(json.dumps({"jobs": len(manifest["jobs"]), "missing_assets": missing, "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
