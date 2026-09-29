#!/usr/bin/env python3
"""Generate candidate-conditioned, executor-owned R5 verification traces.

The 200 x 5 external source is the only evaluation input. A diagnostic prefix
may exercise the producer, but only an all-products sidecar has the semantics
accepted by the paired R5 scorer. No recorded reference enters the policy or
the executor's expected-precursor field.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from mechet.assistant_masking import render_chat
from scripts.agent_model_init import path_sha256
from scripts.autoresearch.r5_candidate_trace import (
    CandidateTraceSession, candidate_conditioned_prompt,
)
from scripts.autoresearch.stratified_manifest import (
    digest, product_key, verify_evaluation_source,
)
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
from scripts.eval_natural_language_event_local import _trim_completion, prediction_call


class PolicyContextExceeded(ValueError):
    """The frozen prompt cannot fit without silently dropping accepted history."""


def read_public_slots(cohort: Path, *, expected_products: int = 200
                      ) -> tuple[list[dict[str, Any]], str]:
    cohort_sha = verify_evaluation_source(cohort, name="r5_external_predictions")
    status_path = cohort.parent / "ARTIFACT_STATUS.json"
    manifest_path = cohort.parent / "manifest.json"
    if not status_path.is_file() or not manifest_path.is_file():
        raise ValueError("R5 source lacks frozen status or manifest")
    status = json.loads(status_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (status.get("evaluation_allowed") is not True
            or status.get("training_allowed") is not False
            or manifest.get("products") != expected_products
            or manifest.get("ranks_per_product") != 5
            or manifest.get("target_semantics") != "retrosynthetic_precursor_set"):
        raise ValueError("R5 source is not the frozen evaluation cohort")
    products = []
    seen: set[str] = set()
    for line in cohort.read_text().splitlines():
        row = json.loads(line)
        product = row.get("product_smiles")
        if (not isinstance(product, str) or product in seen
                or product_key(product) != product
                or row.get("model_input") != {"product_smiles": product}):
            raise ValueError("R5 source has duplicate or malformed product")
        seen.add(product)
        slots = []
        for rank, candidate in enumerate(row.get("candidates", []), 1):
            if candidate.get("rank") != rank:
                raise ValueError("R5 source has reordered/missing rank slots")
            precursor = candidate.get("canonical_precursors")
            valid = candidate.get("smiles_status") == "valid_smiles"
            if valid != isinstance(precursor, str):
                raise ValueError("R5 source has inconsistent candidate validity")
            if valid and product_key(precursor) != precursor:
                raise ValueError("R5 source candidate is noncanonical")
            slots.append({"rank": rank, "canonical_precursors": precursor,
                          "valid": valid})
        if len(slots) != 5:
            raise ValueError("R5 source must retain exactly five rank slots")
        products.append({"product_smiles": product, "slots": slots})
    if len(products) != expected_products:
        raise ValueError("R5 source product denominator changed")
    return products, cohort_sha


def rollout_candidate(product: str, precursor: str,
                      policy: Callable[[str, int], tuple[str, dict[str, Any], str, str]],
                      *, max_decisions: int, max_tool_calls: int,
                      max_imports: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = CandidateTraceSession(product, precursor,
                                    max_tool_calls=max_tool_calls,
                                    max_imports=max_imports)
    raw_steps = []
    for decision in range(max_decisions):
        prompt = session.prompt()
        try:
            name, arguments, error, completion = policy(prompt, decision)
        except PolicyContextExceeded:
            session.termination_reason = "context_budget"
            raw_steps.append({"decision": decision,
                              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                              "parse_error": "CONTEXT_BUDGET_EXCEEDED"})
            break
        step = {"decision": decision,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "generated_text": completion,
                "parsed_name": name, "parse_error": error}
        if error:
            session.termination_reason = "parse_failed"
            raw_steps.append(step)
            break
        result = session.step(name, arguments)
        step["executor_result"] = result
        raw_steps.append(step)
        if session.termination_reason is not None:
            break
    return session.attempt_record(), raw_steps


def verification_rows(products: list[dict[str, Any]],
                      policy: Callable[[str, int], tuple[str, dict[str, Any], str, str]],
                      *, max_attempts: int, max_decisions: int,
                      max_tool_calls: int, max_imports: int
                      ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 1 <= max_attempts <= 100 or max_decisions < 1:
        raise ValueError("R5 attempt/decision budgets are invalid")
    rows, raw = [], []
    for product_row in products:
        product = product_row["product_smiles"]
        for candidate in product_row["slots"]:
            rank = candidate["rank"]
            precursor = candidate["canonical_precursors"]
            if not candidate["valid"]:
                rows.append({"product_smiles": product, "rank": rank,
                             "model_input": None,
                             "verification_status": "skipped_invalid_or_missing",
                             "attempts": []})
                continue
            model_input = {"product_smiles": product,
                           "proposed_precursors": precursor}
            attempts = []
            for attempt_index in range(max_attempts):
                attempt, steps = rollout_candidate(
                    product, precursor, policy,
                    max_decisions=max_decisions, max_tool_calls=max_tool_calls,
                    max_imports=max_imports)
                attempts.append(attempt)
                raw.append({"product_smiles": product, "rank": rank,
                            "attempt_index": attempt_index,
                            "model_input": model_input, "steps": steps})
            rows.append({"product_smiles": product, "rank": rank,
                         "model_input": model_input,
                         "verification_status": "completed",
                         "attempts": attempts})
    return rows, raw


class QwenPolicy:
    def __init__(self, *, base: str, revision: str, adapter: Path,
                 max_new_tokens: int, temperature: float, top_p: float,
                 seed: int, max_context_tokens: int) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("R5 producer requires exactly one visible CUDA GPU")
        if (max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1
                or max_context_tokens <= max_new_tokens):
            raise ValueError("R5 generation settings are invalid")
        torch.cuda.set_device(0)
        torch.manual_seed(seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            base, revision=revision, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            base, revision=revision, trust_remote_code=True,
            torch_dtype=torch.float16, device_map={"": 0},
            attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_context_tokens = max_context_tokens

    def __call__(self, prompt: str, decision: int
                 ) -> tuple[str, dict[str, Any], str, str]:
        del decision
        rendered = render_chat(self.tokenizer, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ], tools=TOOLS, add_generation_prompt=True)
        encoded = self.tokenizer(rendered, return_tensors="pt",
                                 add_special_tokens=False)
        width = int(encoded["input_ids"].shape[1])
        if width + self.max_new_tokens > self.max_context_tokens:
            raise PolicyContextExceeded(
                f"prompt {width} + generation {self.max_new_tokens} exceeds "
                f"frozen context budget {self.max_context_tokens}")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                **({"temperature": self.temperature, "top_p": self.top_p}
                   if self.temperature > 0 else {}),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        ids = _trim_completion(
            [int(item) for item in generated[0, width:].tolist()], self.tokenizer)
        completion = self.tokenizer.decode(ids, skip_special_tokens=False)
        name, arguments, error = prediction_call(completion, self.tokenizer)
        return name, arguments, error, completion


def run(args: argparse.Namespace) -> dict[str, Any]:
    from rdkit import rdBase

    if rdBase.rdkitVersion != "2026.03.4":
        raise RuntimeError(
            "R5 frozen cohort was canonicalized under RDKit 2026.03.4; "
            f"producer has {rdBase.rdkitVersion}")
    if args.output.exists():
        raise FileExistsError(f"R5 verification output already exists: {args.output}")
    products, cohort_sha = read_public_slots(args.cohort)
    diagnostic = args.diagnostic_products is not None
    if diagnostic:
        if not 1 <= args.diagnostic_products < len(products):
            raise ValueError("R5 diagnostic must be a strict product prefix")
        products = products[:args.diagnostic_products]
    adapter_manifest_path = args.adapter / "adapter_manifest.json"
    if not adapter_manifest_path.is_file():
        raise FileNotFoundError(adapter_manifest_path)
    adapter_manifest = json.loads(adapter_manifest_path.read_text())
    adapter_sha = path_sha256(args.adapter)
    if (adapter_manifest.get("adapter_sha256") != adapter_sha
            or adapter_manifest.get("base_model") not in {
                "Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"}
            or not isinstance(adapter_manifest.get("base_model_revision"), str)):
        raise ValueError("R5 adapter/base-model lineage drifted")
    base = str(args.base_model_path or adapter_manifest["base_model"])
    policy = QwenPolicy(
        base=base, revision=adapter_manifest["base_model_revision"],
        adapter=args.adapter, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        max_context_tokens=args.max_context_tokens)
    args.output.mkdir(parents=True)
    verifications = args.output / "verification.jsonl"
    raw_path = args.output / "raw_generations.jsonl"
    counts = {"products": 0, "slots": 0, "valid_slots": 0,
              "terminal_trace_attempts": 0}
    started = time.monotonic()
    with verifications.open("w", encoding="utf-8") as sink, \
            raw_path.open("w", encoding="utf-8") as raw_sink:
        for product in products:
            rows, raw = verification_rows(
                [product], policy, max_attempts=args.max_attempts,
                max_decisions=args.max_decisions,
                max_tool_calls=args.max_tool_calls,
                max_imports=args.max_imports)
            for row in rows:
                sink.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                counts["slots"] += 1
                counts["valid_slots"] += row["model_input"] is not None
                counts["terminal_trace_attempts"] += sum(
                    item["termination_reason"] == "terminal_tool" and
                    (item.get("final_result") or {}).get("formal_execute") is True
                    for item in row["attempts"])
            for item in raw:
                raw_sink.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
            sink.flush()
            raw_sink.flush()
            counts["products"] += 1
            print(f"[meteor-r5-candidate] {counts['products']}/{len(products)} "
                  f"slots={counts['slots']} valid={counts['valid_slots']} "
                  f"terminal={counts['terminal_trace_attempts']} "
                  f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
    sidecar = {
        "artifact_type": "r5_candidate_conditioned_verification_manifest_v1",
        "verification_sha256": digest(verifications),
        "cohort_sha256": cohort_sha,
        "condition": args.condition,
        "checkpoint_identifier": str(args.adapter.resolve()),
        "checkpoint_sha256": adapter_sha,
        "base_model": adapter_manifest["base_model"],
        "base_model_revision": adapter_manifest["base_model_revision"],
        "rdkit_version": rdBase.rdkitVersion,
        "input_fields": ["product_smiles", "proposed_precursors"],
        "verification_semantics": (
            "candidate_conditioned_diagnostic_partial_v1" if diagnostic
            else "candidate_conditioned_executor_trace_v1"),
        "max_attempts": args.max_attempts,
        "max_decisions": args.max_decisions,
        "max_tool_calls": args.max_tool_calls,
        "max_imports": args.max_imports,
        "max_new_tokens": args.max_new_tokens,
        "max_context_tokens": args.max_context_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "seed": args.seed,
        "prompt_template_sha256": hashlib.sha256(
            inspect.getsource(candidate_conditioned_prompt).encode()).hexdigest(),
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "tools_sha256": hashlib.sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
        "raw_generations_sha256": digest(raw_path),
        "counts": counts, "diagnostic_only": diagnostic,
        "claim_boundary": "Sampled executor-owned support of model-visible external candidate; no proof that unverified proposals are chemically invalid.",
    }
    (args.output / "verification.jsonl.manifest.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--condition", choices=("base", "mech", "existing_diagnostic"),
                        default="existing_diagnostic")
    parser.add_argument("--base-model-path")
    parser.add_argument("--diagnostic-products", type=int)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=32)
    parser.add_argument("--max-tool-calls", type=int, default=48)
    parser.add_argument("--max-imports", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
