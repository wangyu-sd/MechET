"""Batched native Qwen3.5 K1 generation and distributed frozen-VALID reports."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .pr96_evaluation import evaluate_reactions
from .pr96_validation import stage_one_gate
from .qwen35_protocol import tool_prefix
from scripts.build_natural_language_event_sft import TOOLS


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_parts(parts, expected_ids):
    ids = [e["reaction_id"] for p in parts for e in p["episodes"]]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("PR96_EVALUATION_IDS_MISSING_OR_DUPLICATED")
    scalars = ("closed_loop_completed", "closed_loop_expected", "structural_hits", "full_hits",
               "truncated_calls", "expected_tf_flow_decisions")
    result = {k: sum(p["metrics"][k] for p in parts) for k in scalars}
    for key in ("parse", "root_import_tool", "root_import_fragment", "tf_flow_successor", "tf_finish"):
        result[key] = {name: sum(p["metrics"][key][name] for p in parts) for name in ("correct", "total")}
    for key in ("root_predicted_tools", "rejected"):
        counter = Counter()
        for part in parts:
            counter.update(part["metrics"][key])
        result[key] = dict(counter)
    result["gt_future_leak_detected"] = any(p["metrics"]["gt_future_leak_detected"] for p in parts)
    return result


def prepare_output_directory(path, rank, barrier):
    if rank == 0:
        Path(path).mkdir(parents=True, exist_ok=False)
    barrier()
    if not Path(path).is_dir():
        raise ValueError("PR96_VALIDATION_DIRECTORY_NOT_CREATED")


class HFBackend:
    def __init__(self, model, tokenizer, *, max_new_tokens, batch_size, raw_path):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.raw_path = Path(raw_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def generate(self, messages):
        import torch
        from transformers import GenerationConfig
        output = []
        for start in range(0, len(messages), self.batch_size):
            batch = messages[start:start + self.batch_size]
            prompts = [tool_prefix(self.tokenizer, m, TOOLS) for m in batch]
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            width = encoded["input_ids"].shape[1]
            if width + self.max_new_tokens > self.model.config.max_position_embeddings:
                raise ValueError("PR96_GENERATION_CONTEXT_OVERFLOW_NO_TRUNCATION_ALLOWED")
            prompt_ids = [ids[mask.bool()].tolist() for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])]
            encoded = {k:v.to(next(self.model.parameters()).device) for k,v in encoded.items()}
            gen = GenerationConfig.from_model_config(self.model.config)
            gen.do_sample, gen.num_beams = False, 1
            gen.pad_token_id, gen.eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            begin = time.monotonic()
            with torch.inference_mode():
                generated = self.model.generate(**encoded, generation_config=gen,
                                                max_new_tokens=self.max_new_tokens, use_cache=True)
            elapsed = time.monotonic() - begin
            for i, sequence in enumerate(generated):
                ids = sequence[width:].tolist()
                ended = self.tokenizer.eos_token_id in ids
                if ended:
                    ids = ids[:ids.index(self.tokenizer.eos_token_id) + 1]
                value = {"raw": self.tokenizer.decode(ids, skip_special_tokens=False),
                         "generated_token_ids": ids, "prompt_token_ids": prompt_ids[i],
                         "truncated": not ended, "batch_elapsed_seconds": elapsed}
                output.append(value)
                # Persist immediately, before scoring or the next batch. If a
                # later batch fails, its predecessors' raw I/O are still usable.
                with self.raw_path.open("a") as sink:
                    sink.write(json.dumps({"messages": batch[i], "prompt": prompts[i], **value}, ensure_ascii=False) + "\n")
            print(json.dumps({"stage": "fixed-valid-generation", "completed": len(output),
                              "total": len(messages), "percent": 100*len(output)/max(len(messages),1),
                              "batch_seconds": elapsed}), flush=True)
        return output


def run_validation(cfg, model, tokenizer, *, adapter, output_dir, cohort="paired"):
    import torch
    import torch.distributed as dist
    output_dir = Path(output_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    prepare_output_directory(output_dir, rank, dist.barrier if dist.is_initialized() else lambda: None)
    settings = cfg["pr96"]
    cohort_file = Path(settings["cohorts_file"])
    frozen = json.loads(cohort_file.read_text())
    ids = frozen["paired_ids" if cohort == "paired" else "independent_ids"]
    selected = {rid for i,rid in enumerate(ids) if i % world == rank}
    reference_dir = Path(settings["reference_dir"])
    references = [r for l in (reference_dir / "inverse/valid.jsonl").open()
                  if str((r := json.loads(l))["source_id"]) in selected]
    rows = {rid:[] for rid in selected}
    for line in (reference_dir / "state_sft/valid.jsonl").open():
        row = json.loads(line)
        rid = str(row["source_id"])
        if rid in selected:
            rows[rid].append(row)
    if {str(r["source_id"]) for r in references} != selected or any(not v for v in rows.values()):
        raise ValueError("PR96_FIXED_VALID_REFERENCES_MISSING")
    backend = HFBackend(model, tokenizer, max_new_tokens=settings["max_new_tokens"],
                        batch_size=int(settings.get("evaluation_batch_size", 8)),
                        raw_path=output_dir / f"raw_generation.rank{rank:02d}.jsonl")
    was_training = model.training
    model.eval()
    with (output_dir / f"calls.rank{rank:02d}.jsonl").open("x") as journal:
        def write(value):
            journal.write(json.dumps(value, ensure_ascii=False) + "\n")
            journal.flush()
            if value.get("mode") == "episode_complete":
                print(json.dumps({"stage": "fixed-valid-episode", "rank": rank,
                                  "reaction_id": value["reaction_id"], "terminal": value["terminal"],
                                  "structural_exact": value["structural_exact"], "error": value["error"]}), flush=True)
        result = evaluate_reactions(references, rows, backend,
                                    compact_history=settings["stage"] == "history",
                                    max_decisions=64, journal=write)
    model.train(was_training)
    part = {"metrics": result["metrics"], "episodes": result["episodes"]}
    (output_dir / f"result.rank{rank:02d}.json").write_text(json.dumps(part, indent=2) + "\n")
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        parts = [json.loads((output_dir / f"result.rank{i:02d}.json").read_text()) for i in range(world)]
        metrics = aggregate_parts(parts, ids)
        report = {"track": settings["track"], "stage": settings["stage"], "cohort": cohort,
                  "reaction_ids": ids, "cohorts_sha256": sha(cohort_file),
                  "adapter": str(adapter), "adapter_model_sha256": sha(Path(adapter) / "adapter_model.safetensors"),
                  "metrics": metrics, "gate": stage_one_gate(metrics) if cohort == "paired" else None,
                  "base_model": cfg["model_name_or_path"], "base_revision": cfg["training"]["model_revision"],
                  "decoder": {"k": 1, "do_sample": False, "num_beams": 1, "max_decisions": 64,
                              "max_new_tokens": settings["max_new_tokens"]},
                  "timestamp": datetime.now(timezone.utc).isoformat(), "complete_test_evaluation_executed": False}
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"stage": "fixed-valid-complete", "report": str(output_dir / "report.json"),
                          "metrics": metrics, "gate": report["gate"]}), flush=True)
    if dist.is_initialized():
        dist.barrier()
    return output_dir / "report.json"
