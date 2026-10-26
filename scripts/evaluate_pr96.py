#!/usr/bin/env python3
"""Evaluate only frozen PR96 VALID cohorts; no GT-conditioned closed loop."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cohort", choices=["paired", "independent"], default="paired")
    args = p.parse_args()
    import torch
    import torch.distributed as dist
    import yaml
    from peft import PeftModel
    from transformers import AutoTokenizer
    from mechet.pr96_trainer import validate_deployed_hardware
    from mechet.qwen35_protocol import load_language_model, OFFICIAL_REVISION
    from mechet.pr96_eval_runtime import run_validation
    cfg = yaml.safe_load(args.config.read_text())
    validate_deployed_hardware(cfg["pr96"]["track"])
    dist.init_process_group("gloo")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], revision=OFFICIAL_REVISION, local_files_only=True)
    base, _ = load_language_model(cfg["model_name_or_path"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).to(int(os.environ["LOCAL_RANK"]))
    run_validation(cfg, model, tokenizer, adapter=args.adapter, output_dir=args.output, cohort=args.cohort)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
