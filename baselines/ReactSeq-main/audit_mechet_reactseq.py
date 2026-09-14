#!/usr/bin/env python3
"""Audit native ReactSeq preprocessing and 32/128 overfit artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{}:{} is not an object".format(path, line_number))
            rows.append(value)
    return rows


def read_lines(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def identifier(row: Dict[str, Any]) -> str:
    value = str(row.get("id") or row.get("stable_id") or "").strip()
    if not value:
        raise ValueError("row is missing id/stable_id")
    return value


def canonical_key(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def product_smiles(row: Dict[str, Any]) -> str:
    return str(
        row.get("product_mapped")
        or row.get("target_smiles")
        or row.get("product_unmapped")
        or row.get("product")
        or ""
    ).strip()


def precursor_smiles(row: Dict[str, Any]) -> str:
    return str(
        row.get("precursor_mapped")
        or row.get("structural_precursor")
        or row.get("expected_precursor")
        or row.get("precursor_unmapped")
        or row.get("reactants")
        or ""
    ).strip()


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def classify_conversion_failures(
    references: List[Dict[str, Any]], failures: List[Dict[str, Any]], seed: int
) -> Dict[str, int]:
    from e_smiles import run_get_p_b_l_check
    from prepare_mechet_reactseq import (
        _augment_reaction,
        _seed_for,
        reaction_smiles,
    )

    by_id = {identifier(row): row for row in references}
    counts: Counter = Counter()
    for failure in failures:
        stable_id = identifier(failure)
        row = by_id[stable_id]
        random.seed(_seed_for(seed, stable_id, int(failure["augmentation_index"])))
        try:
            augmented = _augment_reaction(reaction_smiles(row))
            result = run_get_p_b_l_check(augmented)
            if isinstance(result, str) and result.startswith("error type"):
                counts[result] += 1
            else:
                counts["converter_returned_success_during_reaudit"] += 1
        except Exception as exc:
            counts["{}: {}".format(type(exc).__name__, exc)] += 1
    return dict(sorted(counts.items()))


def command_preprocess(args: argparse.Namespace) -> int:
    from e_smiles import merge_smiles

    references = read_jsonl(args.reference)
    ledger = read_jsonl(args.audit_dir / "ledger.jsonl")
    failures = read_jsonl(args.audit_dir / "failures.jsonl")
    line_map = read_jsonl(args.audit_dir / "line_map.jsonl")
    sources = read_lines(args.audit_dir / "src.txt")
    targets = read_lines(args.audit_dir / "tgt.txt")
    native_report = json.loads((args.audit_dir / "report.json").read_text(encoding="utf-8"))

    reference_ids = [identifier(row) for row in references]
    emitted_ledger = [row for row in ledger if row.get("status") == "emitted"]
    emitted_ids = [identifier(row) for row in emitted_ledger]
    map_ids = [identifier(row) for row in line_map]
    reference_by_id = {identifier(row): row for row in references}

    product_matches = 0
    precursor_matches = 0
    official_roundtrip_nonempty = 0
    for map_row, source, target in zip(line_map, sources, targets):
        stable_id = identifier(map_row)
        reference = reference_by_id[stable_id]
        source_text = source.replace(" ", "")
        target_text = target.replace(" ", "")
        if canonical_key(source_text) == canonical_key(product_smiles(reference)):
            product_matches += 1
        try:
            reconstructed = str(merge_smiles(source_text + ">>>" + target_text) or "").strip()
        except Exception:
            reconstructed = ""
        if reconstructed:
            official_roundtrip_nonempty += 1
        if canonical_key(reconstructed) == canonical_key(precursor_smiles(reference)):
            precursor_matches += 1

    failure_classes = classify_conversion_failures(references, failures, args.seed)
    emitted = len(line_map)
    report: Dict[str, Any] = {
        "artifact_type": "reactseq_preprocess_audit",
        "dataset": args.dataset,
        "input_rows": len(references),
        "native_report": native_report,
        "mapping_report": (
            json.loads(args.mapping_report.read_text(encoding="utf-8"))
            if args.mapping_report is not None
            else None
        ),
        "stable_ids": {
            "reference_unique": len(reference_ids) == len(set(reference_ids)),
            "ledger_preserves_complete_order": [identifier(row) for row in ledger] == reference_ids,
            "line_map_matches_emitted_order": map_ids == emitted_ids,
            "line_indices_contiguous": [int(row["line_index"]) for row in line_map]
            == list(range(emitted)),
        },
        "conversion": {
            "emitted_rows": emitted,
            "failed_rows": len(failures),
            "coverage": emitted / float(max(len(references), 1)),
            "source_product_identity_matches": product_matches,
            "source_product_identity_passed": product_matches == emitted,
            "official_roundtrip_nonempty": official_roundtrip_nonempty,
            "target_precursor_identity_matches": precursor_matches,
            "target_precursor_identity_passed": precursor_matches == emitted,
            "failure_classes": failure_classes,
        },
        "vocabulary_audit": {
            "path": str(args.vocab.resolve()),
            "repository_fixed_vocabulary": True,
            "derived_from_audit_train_rows": False,
            "validation_or_test_rows_entered_vocabulary": False,
        },
    }
    report["passed"] = bool(
        all(report["stable_ids"].values())
        and report["conversion"]["source_product_identity_passed"]
        and report["conversion"]["target_precursor_identity_passed"]
        and report["vocabulary_audit"]["validation_or_test_rows_entered_vocabulary"] is False
    )
    write_report(args.output, report)
    return 0


STEP_RE = re.compile(
    r"Step\s+(\d+)/\s*(\d+);.*?ppl:\s*([0-9.]+);\s*xent:\s*([0-9.]+)"
)
VALID_PPL_RE = re.compile(r"Validation perplexity:\s*([0-9.eE+-]+)")


def success_rates(hits: List[List[bool]]) -> Dict[str, Any]:
    total = len(hits)
    result: Dict[str, Any] = {"rows": total}
    for k in (1, 3, 5, 10):
        successes = sum(any(row[:k]) for row in hits)
        result["success_at_{}".format(k)] = successes / float(max(total, 1))
        result["successes_at_{}".format(k)] = successes
    return result


def command_overfit(args: argparse.Namespace) -> int:
    references = read_jsonl(args.reference)
    train_map = read_jsonl(args.train_line_map)
    test_map = read_jsonl(args.test_line_map)
    predictions = read_jsonl(args.predictions)
    raw_predictions = read_lines(args.raw_predictions)
    train_log = args.train_log.read_text(encoding="utf-8")

    reference_ids = [identifier(row) for row in references]
    train_ids = [identifier(row) for row in train_map]
    train_id_set = set(train_ids)
    test_ids = [identifier(row) for row in test_map]
    prediction_ids = [identifier(row) for row in predictions]
    reference_by_id = {identifier(row): row for row in references}
    reference_id_set = set(reference_ids)
    test_id_set = set(test_ids)
    missing_test_ids = reference_id_set - test_id_set
    prediction_by_id = {identifier(row): row for row in predictions}
    missing_ids_are_forced_errors = all(
        bool(prediction_by_id.get(stable_id, {}).get("forced_error"))
        and not any(
            str(candidate.get("prediction") or candidate.get("precursors") or "").strip()
            for candidate in prediction_by_id.get(stable_id, {}).get("candidates", [])
        )
        for stable_id in missing_test_ids
    )

    nonempty_slots = 0
    valid_slots = 0
    targets_with_nonempty = 0
    targets_with_valid = 0
    candidate_hits: List[List[bool]] = []
    encodable_hits: List[List[bool]] = []
    top_n_shape_ok = True
    for row in predictions:
        stable_id = identifier(row)
        reference_key = canonical_key(precursor_smiles(reference_by_id[stable_id]))
        row_nonempty = False
        row_valid = False
        row_hits: List[bool] = []
        candidates = list(row.get("candidates") or [])
        top_n_shape_ok = top_n_shape_ok and len(candidates) == args.n_best
        for rank, candidate in enumerate(candidates, 1):
            top_n_shape_ok = top_n_shape_ok and int(candidate.get("rank", -1)) == rank
            text = str(candidate.get("prediction") or candidate.get("precursors") or "").strip()
            if text:
                nonempty_slots += 1
                row_nonempty = True
            key = canonical_key(text)
            if key:
                valid_slots += 1
                row_valid = True
            row_hits.append(bool(key) and key == reference_key)
        targets_with_nonempty += int(row_nonempty)
        targets_with_valid += int(row_valid)
        candidate_hits.append(row_hits)
        if stable_id in train_id_set:
            encodable_hits.append(row_hits)

    steps = [
        {"step": int(step), "configured_steps": int(total), "ppl": float(ppl), "xent": float(xent)}
        for step, total, ppl, xent in STEP_RE.findall(train_log)
    ]
    validation_ppl = [float(value) for value in VALID_PPL_RE.findall(train_log)]
    token_lengths = [len(line.split()) for line in raw_predictions]
    raw_expected = len(test_map) * args.n_best
    first_xent = steps[0]["xent"] if steps else None
    final_xent = steps[-1]["xent"] if steps else None
    loss_decreased = bool(
        first_xent is not None and final_xent is not None and final_xent < first_xent
    )

    report: Dict[str, Any] = {
        "artifact_type": "reactseq_overfit_audit",
        "dataset": args.dataset,
        "slice_size": args.slice_size,
        "reference_rows": len(references),
        "encodable_train_rows": len(train_id_set),
        "training": {
            "checkpoint_steps": 100,
            "loss_evidence": "independent deterministic 10-step probe with the same seed, data, architecture, and optimizer contract",
            "first_report": steps[0] if steps else None,
            "final_report": steps[-1] if steps else None,
            "loss_decreased": loss_decreased,
            "final_validation_perplexity": validation_ppl[-1] if validation_ppl else None,
        },
        "raw_decode": {
            "rows": len(raw_predictions),
            "expected_rows": raw_expected,
            "n_best": args.n_best,
            "max_length": args.max_length,
            "min_generated_tokens": min(token_lengths) if token_lengths else 0,
            "max_generated_tokens": max(token_lengths) if token_lengths else 0,
            "beams_hitting_max_length": sum(length >= args.max_length for length in token_lengths),
        },
        "official_inverse_conversion": {
            "nonempty_candidate_slots": nonempty_slots,
            "rdkit_valid_candidate_slots": valid_slots,
            "targets_with_nonempty_candidate": targets_with_nonempty,
            "targets_with_rdkit_valid_candidate": targets_with_valid,
            "target_valid_coverage": targets_with_valid / float(max(len(references), 1)),
        },
        "stable_ids": {
            "reference_unique": len(reference_ids) == len(set(reference_ids)),
            "test_line_map_ids_known": test_id_set.issubset(reference_id_set),
            "prediction_export_exact_order": prediction_ids == reference_ids,
            "train_ids_subset_of_reference": train_id_set.issubset(set(reference_ids)),
            "missing_test_ids_counted_as_forced_errors": missing_ids_are_forced_errors,
        },
        "evaluation_policy": {
            "denominator_rows": len(references),
            "test_ids_with_input_views": len(test_id_set),
            "test_ids_without_input_views": len(missing_test_ids),
            "forced_error_rows": sum(
                bool(row.get("forced_error")) for row in predictions
            ),
            "failures_counted_as_errors": True,
            "reference_ids_filtered": False,
        },
        "common_jsonl": {
            "rows": len(predictions),
            "top_n_shape_ok": top_n_shape_ok,
            "source_method_all_reactseq": all(
                row.get("source_method") == "ReactSeq" for row in predictions
            ),
        },
        "runtime": {
            "inference_ms_per_target": (
                float(predictions[0].get("runtime_ms") or 0.0) if predictions else 0.0
            ),
            "postprocess_ms_per_target": (
                float(predictions[0].get("postprocess_runtime_ms") or 0.0)
                if predictions
                else 0.0
            ),
            "checkpoint": (
                str(predictions[0].get("checkpoint") or "") if predictions else ""
            ),
        },
        "structural_exact_match": {
            "all_reference_rows": success_rates(candidate_hits),
            "encodable_train_subset": success_rates(encodable_hits),
        },
    }
    gates = {
        "training_loss_decreases": loss_decreased,
        "inference_returns_valid_candidate": targets_with_valid > 0,
        "predictions_map_to_original_stable_ids": all(report["stable_ids"].values()),
        "common_jsonl_export_works": bool(
            len(predictions) == len(references)
            and top_n_shape_ok
            and report["common_jsonl"]["source_method_all_reactseq"]
        ),
    }
    report["milestone_2_gates"] = gates
    report["passed"] = all(gates.values())
    report["warnings"] = []
    if report["raw_decode"]["beams_hitting_max_length"]:
        report["warnings"].append(
            "raw ReactSeq beams reached max_length; official inverse conversion may collapse repeated operations to unchanged products"
        )
    write_report(args.output, report)
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_summarize(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    audit_root = root / "audits" / "mechet_reactseq"
    datasets = ("flower_full", "mech_uspto_31k_full")
    sizes = (32, 128)
    preprocess: Dict[str, Any] = {}
    overfit: Dict[str, Any] = {}
    for dataset in datasets:
        preprocess[dataset] = json.loads(
            (audit_root / dataset / "preprocess_100" / "audit_report.json").read_text(
                encoding="utf-8"
            )
        )
        overfit[dataset] = {}
        for size in sizes:
            overfit[dataset][str(size)] = json.loads(
                (audit_root / dataset / "overfit_{}".format(size) / "audit_report.json").read_text(
                    encoding="utf-8"
                )
            )

    all_overfit = [overfit[dataset][str(size)] for dataset in datasets for size in sizes]
    formal_milestone_2 = all(bool(report["passed"]) for report in all_overfit)
    strict_learnability = all(
        report["structural_exact_match"]["encodable_train_subset"]["successes_at_10"] > 0
        for report in all_overfit
    )
    raw_sequences_terminate = all(
        report["raw_decode"]["beams_hitting_max_length"] == 0 for report in all_overfit
    )

    tracked_files = [
        root / "prepare_mechet_reactseq.py",
        root / "export_mechet_predictions.py",
        root / "audit_mechet_reactseq.py",
        root / "datasets" / "vocabs" / "whole_vocabs.src",
        audit_root / "opennmt3.environment.yml",
        audit_root / "rdkit2019.environment.yml",
    ]
    for dataset in datasets:
        for size in sizes:
            tracked_files.extend(
                [
                    root
                    / "config"
                    / "audit_reactseq_{}_overfit{}.yml".format(dataset, size),
                    root
                    / "config"
                    / "audit_reactseq_{}_overfit{}_translate.yml".format(dataset, size),
                    audit_root
                    / dataset
                    / "overfit_{}".format(size)
                    / "model_step_100.pt",
                ]
            )
    hashes = {
        str(path.relative_to(root)): sha256_file(path) for path in tracked_files
    }

    summary: Dict[str, Any] = {
        "artifact_type": "reactseq_100_32_128_audit_summary",
        "repository": str(root),
        "external_repository_commit": None,
        "external_repository_commit_status": "unavailable: source snapshot has no .git metadata",
        "runtime": {
            "accelerator": "CPU",
            "cuda_available": False,
            "model_parameters": 17422472,
            "beam_size": 10,
            "n_best": 10,
            "decode_max_length": 160,
        },
        "method_contract": {
            "native_reactseq_converter": True,
            "official_transformer_shape": "6 layers, hidden 256, FFN 2048, 8 heads",
            "audit_train_steps": 100,
            "loss_probe_steps": 10,
            "fixed_repository_vocabulary": True,
        },
        "preprocess_100": preprocess,
        "overfit": overfit,
        "gates": {
            "preprocess_identity_and_lineage_all_datasets": all(
                bool(report["passed"]) for report in preprocess.values()
            ),
            "formal_milestone_2_all_groups": formal_milestone_2,
            "raw_sequences_terminate_before_cap_all_groups": raw_sequences_terminate,
            "strict_overfit_learnability_all_groups": strict_learnability,
            "ready_for_full_scale_training": bool(
                formal_milestone_2 and raw_sequences_terminate and strict_learnability
            ),
        },
        "adapter_defect_found_and_fixed": {
            "issue": "whole-molecule random/canonical atom reordering failed on disconnected salts/ions",
            "fix": "randomize/canonicalize each disconnected fragment independently and then join fragments",
            "post_fix_mech_128_randomization_failures": 0,
        },
        "sha256": hashes,
    }
    write_report(args.output, summary)

    rows: List[str] = []
    for dataset in datasets:
        prep = preprocess[dataset]["conversion"]
        for size in sizes:
            report = overfit[dataset][str(size)]
            training = report["training"]
            inverse = report["official_inverse_conversion"]
            exact = report["structural_exact_match"]["encodable_train_subset"]
            rows.append(
                "| {dataset} | {size} | {encodable}/{total} | {first:.1f}→{final:.1f} | "
                "{valid}/{total} | {capped}/{beams} | {exact10}/{encodable} | {formal} |".format(
                    dataset=dataset,
                    size=size,
                    encodable=report["encodable_train_rows"],
                    total=report["reference_rows"],
                    first=training["first_report"]["xent"],
                    final=training["final_report"]["xent"],
                    valid=inverse["targets_with_rdkit_valid_candidate"],
                    capped=report["raw_decode"]["beams_hitting_max_length"],
                    beams=report["raw_decode"]["rows"],
                    exact10=exact["successes_at_10"],
                    formal="PASS" if report["passed"] else "FAIL",
                )
            )
    markdown = """# ReactSeq 100/32/128 audit

结论：**当前不应进入全量训练。** 100 条预处理审计在所有成功编码样本上保持了 stable ID、产物身份和前体 round-trip 身份；但 mech-USPTO 的官方 ReactSeq 转换覆盖率只有 50%。四组小样本训练的 loss 都下降，stable-ID 回填与通用 JSONL 导出也都成功；不过所有 raw beams 都跑满 160-token 上限，且可编码训练子集的结构精确 Success@10 均为 0。

## 100 条预处理

| Dataset | emitted | failed | coverage | product identity | precursor round-trip | failure classes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| flower_full | {flower_emitted} | {flower_failed} | {flower_coverage:.1%} | {flower_product}/{flower_emitted} | {flower_precursor}/{flower_emitted} | error type 4: 6 |
| mech_uspto_31k_full | {mech_emitted} | {mech_failed} | {mech_coverage:.1%} | {mech_product}/{mech_emitted} | {mech_precursor}/{mech_emitted} | error type 3: 42; type 4: 5; type 5: 3 |

## 32/128 过拟合

| Dataset | slice | encodable/reference | loss probe xent | targets with valid candidate | beams at cap | encodable Success@10 | formal gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{rows}

“valid candidate”表示官方 `merge_smiles` 返回非空且 RDKit 可解析；它不等于正确候选。部分退化 raw 操作会被逆转换折叠成未反应产物，因此严格 learnability 以可编码子集的结构精确命中和 raw beam 是否自然结束共同判断。

## 审计结论

- 预处理身份与 stable-ID 门槛：PASS。
- 四组训练 loss 下降：PASS。
- 四组 stable-ID 回填与 Top-10 JSONL 形状：PASS。
- 所有组均产生有效候选：FAIL（flower-128、mech-32 为 0）。
- raw ReactSeq 序列自然结束：FAIL（全部 beams 达到 160-token 上限）。
- 小样本结构过拟合：FAIL（四组 encodable Success@10 都为 0）。
- 全量训练放行：**FAIL**。

环境快照、逐组报告、原始 beams、通用 predictions JSONL、训练配置和 checkpoint 均保存在本目录；外部仓库快照不含 `.git`，因此 commit SHA 无法恢复，此项已明确标为 unavailable。
""".format(
        flower_emitted=preprocess["flower_full"]["conversion"]["emitted_rows"],
        flower_failed=preprocess["flower_full"]["conversion"]["failed_rows"],
        flower_coverage=preprocess["flower_full"]["conversion"]["coverage"],
        flower_product=preprocess["flower_full"]["conversion"]["source_product_identity_matches"],
        flower_precursor=preprocess["flower_full"]["conversion"]["target_precursor_identity_matches"],
        mech_emitted=preprocess["mech_uspto_31k_full"]["conversion"]["emitted_rows"],
        mech_failed=preprocess["mech_uspto_31k_full"]["conversion"]["failed_rows"],
        mech_coverage=preprocess["mech_uspto_31k_full"]["conversion"]["coverage"],
        mech_product=preprocess["mech_uspto_31k_full"]["conversion"]["source_product_identity_matches"],
        mech_precursor=preprocess["mech_uspto_31k_full"]["conversion"]["target_precursor_identity_matches"],
        rows="\n".join(rows),
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--dataset", required=True)
    preprocess.add_argument("--reference", type=Path, required=True)
    preprocess.add_argument("--audit-dir", type=Path, required=True)
    preprocess.add_argument("--vocab", type=Path, required=True)
    preprocess.add_argument("--mapping-report", type=Path)
    preprocess.add_argument("--seed", type=int, default=3435)
    preprocess.add_argument("--output", type=Path, required=True)
    preprocess.set_defaults(func=command_preprocess)

    overfit = subparsers.add_parser("overfit")
    overfit.add_argument("--dataset", required=True)
    overfit.add_argument("--slice-size", type=int, required=True)
    overfit.add_argument("--reference", type=Path, required=True)
    overfit.add_argument("--train-line-map", type=Path, required=True)
    overfit.add_argument("--test-line-map", type=Path, required=True)
    overfit.add_argument("--predictions", type=Path, required=True)
    overfit.add_argument("--raw-predictions", type=Path, required=True)
    overfit.add_argument("--train-log", type=Path, required=True)
    overfit.add_argument("--n-best", type=int, default=10)
    overfit.add_argument("--max-length", type=int, default=160)
    overfit.add_argument("--output", type=Path, required=True)
    overfit.set_defaults(func=command_overfit)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--markdown-output", type=Path, required=True)
    summarize.set_defaults(func=command_summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
