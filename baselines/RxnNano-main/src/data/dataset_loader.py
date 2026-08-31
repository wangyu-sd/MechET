# -*- coding: utf-8 -*-
from datasets import load_dataset
from typing import Dict, List
from pathlib import Path
from ..config.settings import EvaluationConfig

def get_dataset_files(config: EvaluationConfig) -> Dict[str, str]:
    return {
        'retrosynthesis': config.retrosynthesis_dataset,
        'retrosynthesis_class': config.retrosynthesis_class_dataset,
        'forward_prediction': config.forward_prediction_dataset,
    }

def load_dataset_for_task(task: str, dataset_file: str):
    if not dataset_file.endswith('.jsonl'):
        raise ValueError(f"Dataset file for {task} must end with '.jsonl'")
    
    dataset_path = Path(dataset_file)
    if not dataset_path.is_file():
        dataset_path = Path("data") / dataset_file
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
    return load_dataset("json", data_files=str(dataset_path))["train"]
