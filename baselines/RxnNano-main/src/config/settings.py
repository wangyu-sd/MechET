# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2500
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    
    # Training parameters
    train_epochs: int = 10
    per_device_train_batch_size: int = 24
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps_ratio: float = 0.02
    eval_steps_ratio: float = 0.05
    save_steps_ratio: float = 0.05
    
    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Task configuration
    tasks: str = "retrosynthesis,retrosynthesis_class,forward_prediction"
    dataset_variants: str = "mapped,unmapped0,unmappedsmile,unmappedraw"
    prompt_style: str = "with_plan"
    
    # Dataset paths
    retro_dataset: str = "_50K"
    forward_dataset: str = "_480K"
    
    def __post_init__(self):
        self.selected_tasks = [task.strip() for task in self.tasks.split(',')]
        self.dataset_variants = [v.strip() for v in self.dataset_variants.split(',')]
        
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class EvaluationConfig:
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2500
    dtype: str = "bfloat16"
    
    # Evaluation parameters
    n: int = 10
    generate_n: int = 50
    top_k_metrics: str = '1,3,5,10'
    batch_size: int = 100000
    temperature: float = 1.3
    top_p: float = 0.98
    top_k: int = 100
    max_tokens: int = 2000
    seed: int = 3407
    
    # Task configuration
    tasks: str = "retrosynthesis"
    prompt_type: str = '1-plan'
    
    # Dataset paths
    retrosynthesis_dataset: str = "test_50K.jsonl"
    retrosynthesis_class_dataset: str = "test_typed.jsonl"
    forward_prediction_dataset: str = "test_480K.jsonl"
    
    # System settings
    cuda_device: str = "1"
    disable_logging_csv: bool = False
    predictions_output: Optional[str] = None
    metrics_output: Optional[str] = None
    
    def __post_init__(self):
        self.selected_tasks = [task.strip() for task in self.tasks.split(',')]
        self.top_k_metrics_list = [int(k) for k in self.top_k_metrics.split(',')]
        
        # Validate generate_n and n
        if self.generate_n < self.n:
            raise ValueError("generate_n must be >= n")
        
        # Validate tasks
        valid_tasks = ['retrosynthesis', 'retrosynthesis_class', 'forward_prediction']
        if not all(task in valid_tasks for task in self.selected_tasks):
            raise ValueError(f"Invalid tasks. Available tasks: {valid_tasks}")
        
        # Validate top_k_metrics
        if not all(k > 0 for k in self.top_k_metrics_list):
            raise ValueError("All top-k values must be positive integers")
        if max(self.top_k_metrics_list) > self.n:
            raise ValueError("All top-k values must be <= n")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
