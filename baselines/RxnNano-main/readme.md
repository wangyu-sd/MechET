# RxnNano: RxnNano:Training Compact LLMs for Chemical Reaction and
Retrosynthesis Prediction via Hierarchical Curriculum Learning

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- Fine-tune LLM for chemical reaction tasks:
  - Retrosynthesis prediction
  - Retrosynthesis with reaction class
  - Forward reaction prediction
- Support for multiple dataset variants:
  - Mapped SMILES
  - Unmapped SMILES (multiple formats)
- LoRA adapter training for efficient fine-tuning
- Comprehensive evaluation metrics:
  - Top-k accuracy

## Installation

1. Clone the repository:
```bash
git clone 
cd chemical-reaction-prediction
```

2. Create and activate a conda environment:
```bash
conda create -n chem_llm python=3.10
conda activate chem_llm
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a model on specific tasks:
```bash
python train.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --tasks retrosynthesis \
    --dataset_variants mapped \
    --cuda_device 0 \
    --train_epochs 3 \
    --lora_rank 64
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py \
    --model_name   [model_path] \
    --tasks retrosynthesis \
    --retrosynthesis_dataset test_50K.jsonl \
    --cuda_device 0 \
    --top_k_metrics 1
```

## Dataset Preparation

Dataset files should be placed in the `data/` directory with the following naming convention:

```
data/
├── train_50K.jsonl                  # Mapped retrosynthesis training
├── train_50K_unmapped0.jsonl        # Unmapped retrosynthesis training
├── train_typed.jsonl                # Typed retrosynthesis training
├── train_480K.jsonl                 # Forward prediction training
├── validation_50K.jsonl             # Mapped retrosynthesis validation
└── test_50K.jsonl                   # Mapped retrosynthesis test
```

Each JSONL file should contain examples in the format:
```json
{
  "product": "CC(=O)Nc1ccccc1",
  "reactants": "c1ccccc1NH2.CC(=O)Cl",
  "rxn_Class": "Amide formation"  # For retrosynthesis_class only
}
```

## Project Structure

```
chemical-reaction-prediction/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── prompts.py          # System and task prompts
│   │   └── settings.py         # Configuration dataclasses
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py   # Dataset loading utilities
│   │   └── dataset_formatters.py # Prompt formatting
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py        # Evaluation logic
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── prompt_formatters.py # Evaluation prompts
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py     # Model loading
│   │   └── lora_utils.py       # LoRA configuration
│   ├── train/
│   │   
│   └── utils/
│       ├── __init__.py
│       ├── logging.py          # Logging setup
│       ├── smiles_utils.py     # SMILES processing
│       └── device.py           # GPU setup
├── data/                       # Dataset files
├── loras/                      # Saved LoRA adapters
├── merged_models/              # Merged model checkpoints
├── logs/                       # Training logs
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Example Workflow

### 1. Prepare Datasets
```bash
# Ensure your dataset files are in the data/ directory
ls data/
# train_50K.jsonl, validation_50K.jsonl, test_50K.jsonl, etc.
```

### 2. Train a Model
```bash
# Train on retrosynthesis with unmapped data
python train.py \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --tasks retrosynthesis \
    --cuda_device 0 \
    --train_epochs 3 \
    --lora_rank 32
```

### 3. Evaluate the Model
```bash
# Evaluate on test set
python evaluate.py \
    --model_path loras/Qwen_Qwen2.5-0.5B-Instruct_retrosynthesis_unmapped0_with_plan_32_5_sft \
    --tasks retrosynthesis \
    --retrosynthesis_dataset test_50K.jsonl \
    --cuda_device 0 \
    --top_k_metrics 1
```


## Support

For questions and support:
- Open an issue on GitHub

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for optimized training
