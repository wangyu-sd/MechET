# -*- coding: utf-8 -*-
import re

def extract_response(text: str) -> str:
    """Extract the response part from the assistant's answer."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text

def get_saved_path(args, config):
    """Generate the path for saving models."""
    tasks_str = '_'.join(args.selected_tasks)
    variants_str = '_'.join(sorted(args.dataset_variants))
    return f"sft/{args.model_name}_{tasks_str}_{variants_str}_{args.prompt_style}_sft_{config.lora_rank}_{config.train_epochs}"