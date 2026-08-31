# -*- coding: utf-8 -*-
import os
from ..config.settings import EvaluationConfig

def load_model_and_tokenizer(config: EvaluationConfig):
    """Load the model and tokenizer with the given configuration."""
    # Keep these imports after CUDA_VISIBLE_DEVICES is set by evaluate.py.
    import torch
    from vllm import LLM
    from transformers import AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device
    
    print(f"Loading model: {config.model_name}")
    model = LLM(
        model=config.model_name,
        max_model_len=config.max_seq_length,
        dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    return model, tokenizer
