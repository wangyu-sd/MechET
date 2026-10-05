#!/usr/bin/env python3
"""PR81 Phase-1a: frozen-encoder EVENT/IMPORT/FINISH route choice.

This deliberately does not enumerate or generate IMPORT fragment SMILES. It
tests whether the successful Phase-0 state encoder can route action families.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.system_one_action_family import (
    ACTION_NAMES, ActionFamilyExample, ActionFamilyHead, parse_action_family_example,
)
from mechet.system_one_decision import append_option_anchors
from scripts.train_system_one_electron_flow import file_sha256, verify_source


@dataclass(frozen=True)
class PreparedRoute:
    example: ActionFamilyExample
    input_ids: array


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def load_split(path: Path) -> tuple[list[ActionFamilyExample], dict]:
    examples = []
    ids = set()
    counts = Counter()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"blank row in {path}")
            example = parse_action_family_example(json.loads(line))
            if example.row_id in ids:
                raise ValueError(f"duplicate decision ID {example.row_id}")
            ids.add(example.row_id)
            counts[ACTION_NAMES[example.label]] += 1
            examples.append(example)
    return examples, {"rows": len(examples), "actions": dict(counts),
                      "reactions": len({example.reaction_id for example in examples})}


def prepare(examples: list[ActionFamilyExample], tokenizer) -> tuple[list[PreparedRoute], dict]:
    prepared = []
    total_tokens = max_length = 0
    for example in examples:
        messages = append_option_anchors(example.messages, example.atom_names)
        prefix = render_qwen_sft_tool_prefix(tokenizer, messages, tools=example.tools)
        ids = array("I", tokenizer(prefix, add_special_tokens=False)["input_ids"])
        if not ids:
            raise ValueError(f"{example.row_id}: empty model input")
        prepared.append(PreparedRoute(example, ids))
        total_tokens += len(ids)
        max_length = max(max_length, len(ids))
    return prepared, {"rows": len(prepared), "input_tokens": total_tokens,
                      "mean_length": total_tokens / len(prepared), "max_length": max_length}


def metrics(labels: list[int], predictions: list[int]) -> dict:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels/predictions are empty or mismatched")
    n = len(labels)
    matrix = [[0] * len(ACTION_NAMES) for _ in ACTION_NAMES]
    for gold, predicted in zip(labels, predictions, strict=True):
        matrix[gold][predicted] += 1
    classes = {}
    f1s = []
    for index, name in enumerate(ACTION_NAMES):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        classes[name] = {"support": support, "predicted": predicted,
                         "precision": precision, "recall": recall, "f1": f1}
    finish = len(ACTION_NAMES) - 1
    false_finish = sum(matrix[gold][finish] for gold in range(finish))
    return {
        "n": n,
        "accuracy": sum(matrix[i][i] for i in range(len(ACTION_NAMES))) / n,
        "macro_f1": sum(f1s) / len(f1s),
        "classes": classes,
        "confusion_gold_rows_predicted_columns": matrix,
        "premature_finish": false_finish,
        "premature_finish_rate_among_nonfinish": false_finish / (n - classes["finish_trace"]["support"]),
    }


def history_majority(train: list[PreparedRoute], selected: list[PreparedRoute]) -> list[int]:
    counts = defaultdict(Counter)
    overall = Counter(item.example.label for item in train)
    default = max(range(len(ACTION_NAMES)), key=lambda label: (overall[label], -label))
    for item in train:
        counts[item.example.history_accepted_actions][item.example.label] += 1
    mapping = {history: max(range(len(ACTION_NAMES)),
                            key=lambda label: (bucket[label], -label))
               for history, bucket in counts.items()}
    return [mapping.get(item.example.history_accepted_actions, default) for item in selected]


def encode_batch(policy, tokenizer, rows: list[PreparedRoute], indices: list[int], device):
    import torch

    width = max(len(rows[index].input_ids) for index in indices)
    input_ids = torch.full((len(indices), width), tokenizer.pad_token_id,
                           dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    positions = []
    for local, index in enumerate(indices):
        ids = rows[index].input_ids
        length = len(ids)
        input_ids[local, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention[local, :length] = 1
        positions.append(length - 1)
    states = policy(input_ids=input_ids, attention_mask=attention,
                    use_cache=False).last_hidden_state
    return states[torch.arange(len(indices), device=device),
                  torch.tensor(positions, device=device)].to(torch.float16).cpu()


def main() -> int:
    args = arguments()
    if args.batch_size < 1 or args.max_length < 1 or args.epochs < 1 or args.lr <= 0:
        raise ValueError("invalid batch/length/epoch/learning-rate configuration")
    if (args.output / "action_family_head.pt").exists():
        raise FileExistsError(args.output / "action_family_head.pt")
    checkpoint_manifest = json.loads((args.checkpoint / "run_manifest_epoch1.json").read_text())
    if checkpoint_manifest.get("artifact_type") != "system_one_phase0_decision_policy":
        raise ValueError("Phase-1a requires a completed PR81 Phase-0 policy")
    adapter_path = args.checkpoint / "adapter_epoch1" / "adapter_model.safetensors"
    head_path = args.checkpoint / "decision_head_epoch1.pt"
    if file_sha256(adapter_path) != checkpoint_manifest["adapter_model_sha256"]:
        raise ValueError("Phase-0 adapter hash mismatch")
    if file_sha256(head_path) != checkpoint_manifest["decision_head_sha256"]:
        raise ValueError("Phase-0 pointer-head hash mismatch")
    source = {}
    examples = {}
    counts = {}
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    status = json.loads((args.data_dir / "ARTIFACT_STATUS.json").read_text())
    allowed_artifacts = {
        "mech_uspto_31k_natural_language_electron_event_history_v2",
        "mech_uspto_31k_natural_language_history_principal_target_v2",
    }
    if (not manifest.get("training_allowed") or not status.get("training_allowed")
            or manifest.get("artifact_type") != status.get("artifact_id")
            or manifest.get("artifact_type") not in allowed_artifacts):
        raise ValueError("Phase-1a requires a validated current-compiler history trace view")
    if manifest["artifact_type"].endswith("principal_target_v2") and (
        manifest.get("target_prompt_contract")
        != "endpoint_proxy_product_target_line_with_unchanged_executor_mixture_v2"
        or status.get("target_prompt_contract") != manifest["target_prompt_contract"]
    ):
        raise ValueError("principal-target Phase-1a source contract mismatch")
    for split in ("train", "valid", "test"):
        path = args.data_dir / f"{split}.jsonl"
        source[split] = verify_source(path)
        examples[split], counts[split] = load_split(path)
        declared = manifest["splits"][split]
        if (counts[split]["rows"] != source[split]["decision_rows"]
                or counts[split]["reactions"] != source[split]["reaction_denominator"]
                or counts[split]["actions"] != {
                    "apply_electron_flow": declared["event_decisions"],
                    "import_fragments": declared["import_decisions"],
                    "finish_trace": declared["finish_decisions"],
                }):
            raise ValueError(f"{split}: decision-class denominator mismatch")
    for split in ("train", "valid"):
        if source[split]["sha256"] != checkpoint_manifest[f"{split}_source"]["sha256"]:
            raise ValueError(f"{split}: checkpoint source differs from action-family data")
    if len({source[split]["artifact_id"] for split in source}) != 1:
        raise ValueError("action-family splits have different artifact lineages")
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if {item.reaction_id for item in examples[a]} & {item.reaction_id for item in examples[b]}:
            raise ValueError(f"reaction overlap between {a} and {b}")

    from transformers import AutoTokenizer
    model = checkpoint_manifest["model"]
    revision = checkpoint_manifest["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision,
                                               trust_remote_code=True)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError("fast tokenizer with a pad token is required")
    prepared = {}
    lengths = {}
    for split in ("train", "valid", "test"):
        prepared[split], lengths[split] = prepare(examples[split], tokenizer)
        if lengths[split]["max_length"] > args.max_length:
            raise ValueError(f"{split}: model input exceeds {args.max_length} tokens")
    audit = {
        "artifact_type": "system_one_phase1a_action_family_preflight",
        "scope": "gold_state_action_routing_only_no_import_smiles_generation",
        "source_status": status,
        "source": source,
        "counts": counts,
        "token_lengths": lengths,
        "checkpoint_manifest": str((args.checkpoint / "run_manifest_epoch1.json").resolve()),
        "adapter_sha256": checkpoint_manifest["adapter_model_sha256"],
        "phase0_pointer_head_sha256": checkpoint_manifest["decision_head_sha256"],
        "model": model,
        "revision": revision,
        "trainer_sha256": file_sha256(Path(__file__)),
        "action_family_sha256": file_sha256(ROOT / "src/mechet/system_one_action_family.py"),
        "input_contract": "full_executor_state_plus_gold_independent_post_state_atom_anchors_v1",
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"phase": "preflight", "rows": counts, "lengths": lengths}), flush=True)
    if args.audit_only:
        return 0

    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from safetensors.torch import save_file
    from transformers import AutoModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Phase-1a encoder extraction requires a BF16 CUDA GPU")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    base = AutoModel.from_pretrained(model, revision=revision, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16,
                                     attn_implementation="sdpa").to(device)
    policy = PeftModel.from_pretrained(base, args.checkpoint / "adapter_epoch1",
                                      is_trainable=False).eval()
    hidden = int(policy.config.hidden_size)
    probe_rows = prepared["train"]
    shortest = min(range(len(probe_rows)), key=lambda index: len(probe_rows[index].input_ids))
    longest = max(range(len(probe_rows)), key=lambda index: len(probe_rows[index].input_ids))
    with torch.inference_mode():
        separate = torch.cat([
            encode_batch(policy, tokenizer, probe_rows, [index], device)
            for index in (shortest, longest)
        ])
        together = encode_batch(policy, tokenizer, probe_rows, [shortest, longest], device)
    similarities = F.cosine_similarity(separate.float(), together.float())
    min_similarity = float(similarities.min())
    print(json.dumps({"phase": "batched_prefill_equivalence",
                      "lengths": [len(probe_rows[index].input_ids)
                                  for index in (shortest, longest)],
                      "minimum_cosine_similarity": min_similarity}), flush=True)
    if min_similarity < 0.999:
        raise ValueError("right-padded batched prefill changes true-token hidden states")
    feature = {}
    encode_times = {}
    for split in ("train", "valid", "test"):
        rows = prepared[split]
        vectors = torch.empty((len(rows), hidden), dtype=torch.float16)
        order = sorted(range(len(rows)), key=lambda index: len(rows[index].input_ids))
        started = time.perf_counter()
        with torch.inference_mode():
            for batch_index in range(0, len(order), args.batch_size):
                indices = order[batch_index:batch_index + args.batch_size]
                vectors[indices] = encode_batch(policy, tokenizer, rows, indices, device)
                processed = min(batch_index + len(indices), len(order))
                if processed % 800 <= len(indices) or processed == len(order):
                    print(json.dumps({"phase": "encode", "split": split,
                                      "rows": processed, "total": len(order),
                                      "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
        feature[split] = vectors
        encode_times[split] = time.perf_counter() - started
        feature_path = args.output / f"{split}_features.safetensors"
        save_file({"features": vectors.contiguous()}, feature_path)
        print(json.dumps({"phase": "encoded_split", "split": split,
                          "rows": len(rows), "seconds": round(encode_times[split], 1),
                          "feature_sha256": file_sha256(feature_path)}), flush=True)
    del policy, base
    torch.cuda.empty_cache()

    train_x = feature["train"].to(device=device, dtype=torch.float32)
    valid_x = feature["valid"].to(device=device, dtype=torch.float32)
    test_x = feature["test"].to(device=device, dtype=torch.float32)
    labels = {split: torch.tensor([item.example.label for item in prepared[split]],
                                  dtype=torch.long, device=device)
              for split in prepared}
    head = ActionFamilyHead(hidden).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    frequencies = torch.bincount(labels["train"], minlength=len(ACTION_NAMES)).float()
    weights = torch.sqrt(frequencies.sum() / (len(ACTION_NAMES) * frequencies))
    weights = weights / weights.mean()
    best_epoch = 0
    best_f1 = -1.0
    best_weights = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        permutation = torch.randperm(len(train_x), device=device)
        loss_sum = 0.0
        for start in range(0, len(train_x), 512):
            idx = permutation[start:start + 512]
            loss = F.cross_entropy(head(train_x[idx]), labels["train"][idx], weight=weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(idx)
        head.eval()
        with torch.inference_mode():
            predictions = head(valid_x).argmax(-1).cpu().tolist()
        valid_result = metrics(labels["valid"].cpu().tolist(), predictions)
        print(json.dumps({"phase": "head_epoch", "epoch": epoch,
                          "train_weighted_loss": loss_sum / len(train_x),
                          "valid_accuracy": valid_result["accuracy"],
                          "valid_macro_f1": valid_result["macro_f1"]}), flush=True)
        if valid_result["macro_f1"] > best_f1 + 1e-6:
            best_f1 = valid_result["macro_f1"]
            best_epoch = epoch
            best_weights = {name: tensor.detach().cpu().clone()
                            for name, tensor in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_weights is None:
        raise RuntimeError("action-family head never produced a checkpoint")
    head.load_state_dict(best_weights)
    head.eval()
    with torch.inference_mode():
        valid_predictions = head(valid_x).argmax(-1).cpu().tolist()
        test_predictions = head(test_x).argmax(-1).cpu().tolist()
    report = {
        "artifact_type": "system_one_phase1a_action_family_result",
        "scope": "gold_state_action_routing_only_no_import_smiles_generation",
        "selected_epoch_by_valid_macro_f1": best_epoch,
        "train_events": counts["train"]["actions"]["apply_electron_flow"],
        "train_decisions": counts["train"]["rows"],
        "valid": metrics(labels["valid"].cpu().tolist(), valid_predictions),
        "test": metrics(labels["test"].cpu().tolist(), test_predictions),
        "history_majority_valid": metrics(labels["valid"].cpu().tolist(),
                                          history_majority(prepared["train"], prepared["valid"])),
        "history_majority_test": metrics(labels["test"].cpu().tolist(),
                                         history_majority(prepared["train"], prepared["test"])),
        "encode_elapsed_s": encode_times,
        "feature_sha256": {split: file_sha256(args.output / f"{split}_features.safetensors")
                           for split in prepared},
        "source_sha256": {split: source[split]["sha256"] for split in source},
        "phase0_adapter_sha256": checkpoint_manifest["adapter_model_sha256"],
        "trainer_sha256": audit["trainer_sha256"],
        "action_family_sha256": audit["action_family_sha256"],
        "import_smiles_predicted": False,
        "product_start_rollout": False,
    }
    head_path = args.output / "action_family_head.pt"
    torch.save(best_weights, head_path)
    report["head_sha256"] = file_sha256(head_path)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": "complete", "output": str(args.output),
                      "best_epoch": best_epoch,
                      "valid_macro_f1": report["valid"]["macro_f1"],
                      "test_macro_f1": report["test"]["macro_f1"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
