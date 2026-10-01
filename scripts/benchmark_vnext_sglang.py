#!/usr/bin/env python3
"""Matched oracle-state SGLang native-server benchmark for PR71.

Start a pinned SGLang server separately. Radix-cache and speculative-decoding
settings belong to its server manifest, not this request client. This program
uses the same fixed states and prompt renderer as the vLLM benchmark, and
reports syntax/handle validity, output tokens, and wall time. It does not
claim autonomous endpoint accuracy or silently alter model-visible input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.vnext_structured_actions import parse_structured_action, structured_action_schema
from scripts.benchmark_vnext_structured_vllm import fixed_examples, structural_handle_ok
from scripts.eval_natural_language_event_local import prediction_call


def payload(prompt: str, example, *, mode: str, max_new_tokens: int, lora_path: str = "") -> dict:
    if mode not in {"unconstrained", "schema", "inventory"}:
        raise ValueError("unknown benchmark mode")
    sampling = {"temperature": 0, "max_new_tokens": max_new_tokens}
    if mode != "unconstrained":
        sampling["json_schema"] = json.dumps(
            structured_action_schema(example if mode == "inventory" else None,
                                     inventory_handles=mode == "inventory"),
            separators=(",", ":"),
        )
    request = {"text": prompt, "sampling_params": sampling}
    if lora_path:
        request["lora_path"] = lora_path
    return request


def post_json(url: str, body: dict, *, timeout: int) -> dict:
    request = Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("SGLang returned a non-object response")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--server-manifest", type=Path, required=True,
                        help="frozen model/revision, LoRA, radix-cache and draft-mode declaration")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--modes", nargs="+", choices=["unconstrained", "schema", "inventory"],
                        default=["unconstrained", "schema", "inventory"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.count < 1 or args.timeout < 1:
        raise ValueError("count and timeout must be positive")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    examples = fixed_examples(args.data, count=args.count, seed=args.seed, rank=0, world=1)
    declared_server = json.loads(args.server_manifest.read_text())
    for key in ("model", "revision", "lora", "radix_cache", "speculative_mode"):
        if key not in declared_server:
            raise ValueError(f"server manifest lacks {key}")
    args.output.mkdir(parents=True)
    manifest = {
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "server_manifest_sha256": hashlib.sha256(args.server_manifest.read_bytes()).hexdigest(),
        "server": declared_server,
        "scope": "oracle_state_request_benchmark_not_product_start_accuracy",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for mode in args.modes:
        start = time.monotonic()
        records = []
        for index, example in enumerate(examples, 1):
            prompt = render_qwen_sft_tool_prefix(tokenizer, example.messages, tools=example.tools)
            request = payload(prompt, example, mode=mode,
                              max_new_tokens=args.max_new_tokens, lora_path=args.lora_path)
            error = ""
            raw = ""
            tokens = 0
            try:
                result = post_json(args.server.rstrip("/") + "/generate", request, timeout=args.timeout)
                raw = str(result.get("text") or "")
                tokens = int((result.get("meta_info") or {}).get("completion_tokens") or 0)
                if mode == "unconstrained":
                    name, arguments, error = prediction_call(raw, tokenizer)
                    if error:
                        raise ValueError(error)
                else:
                    name, arguments = parse_structured_action(raw)
                parsed = True
                handle_ok = structural_handle_ok(name, arguments, example)
                signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
            except Exception as exc:
                parsed, handle_ok, signature = False, False, ""
                error = f"{type(exc).__name__}:{exc}"
            records.append({"id": example.row_id, "parsed": parsed, "handle_ok": handle_ok,
                            "signature": signature, "output_tokens": tokens, "error": error,
                            "raw": raw})
            if index % 8 == 0 or index == len(examples):
                print(json.dumps({"mode": mode, "done": index, "total": len(examples),
                                  "elapsed_s": round(time.monotonic() - start, 1)}), flush=True)
        wall = time.monotonic() - start
        with (args.output / f"{mode}.jsonl").open("w") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        report = {
            "mode": mode, "n_states": len(records), "wall_seconds": wall,
            "states_per_second": len(records) / max(wall, 1e-9),
            "output_tokens_per_second": sum(r["output_tokens"] for r in records) / max(wall, 1e-9),
            "parse_success": sum(r["parsed"] for r in records) / len(records),
            "handle_valid": sum(r["handle_ok"] for r in records) / len(records),
            "scope": manifest["scope"],
        }
        (args.output / f"{mode}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
