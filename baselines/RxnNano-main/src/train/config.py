# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    max_seq_length: int = 2500
    train_epochs: int = 10
    per_device_train_batch_size: int = 24
    gradient_accumulation_steps: int = 1
    lora_rank: int = 64
    lora_alpha: int = 128
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_ratio: float = 0.02
    eval_ratio: float = 0.05
    save_ratio: float = 0.05
    seed: int = 3407
    max_steps: int = -1
    output_dir: Optional[str] = None
    save_merged: bool = True
    report_to: str = "tensorboard"

# Supported Qwen models
QWEN_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "unsloth/Qwen2.5-Coder-14B-Instruct",
    "unsloth/Qwen2.5-Coder-7B",
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Qwen2.5-32B-Instruct",
    "unsloth/Qwen2.5-72B-Instruct",
]

# Dataset base suffixes for each task
DATASET_BASES = {
    'retrosynthesis': '_50K',
    'retrosynthesis_class': '_typed',
    'forward_prediction': '_480K',
}
