# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune model for chemical reaction tasks")
    parser.add_argument('--prompt_style', type=str, default='with_plan', choices=['with_plan', 'without_plan'],
                       help="System prompt style: 'with_plan' or 'without_plan'")
    parser.add_argument('--tasks', type=str, default='retrosynthesis,retrosynthesis_class,forward_prediction',
                       help="Comma-separated list of tasks to train on")
    parser.add_argument('--dataset_variants', type=str, default='mapped',
                       help="Comma-separated dataset variants")
    parser.add_argument('--retro_dataset', type=str, default='_50K',
                       help="Retrosynthesis dataset suffix")
    parser.add_argument('--forward_dataset', type=str, default='_480K',
                       help="Forward prediction dataset suffix")
    parser.add_argument('--cuda_device', type=str, default="1",
                       help="CUDA device number(s) to use")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    parser.add_argument('--train_file', type=str,
                       help="Explicit JSONL training file (audit/small-data runs)")
    parser.add_argument('--validation_file', type=str,
                       help="Explicit JSONL validation file (audit/small-data runs)")
    parser.add_argument('--train_epochs', type=int,
                       help="Override the configured number of training epochs")
    parser.add_argument('--max_steps', type=int, default=-1,
                       help="Stop after this many optimizer steps; -1 uses epochs")
    parser.add_argument('--per_device_train_batch_size', type=int,
                       help="Override the per-device training batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int,
                       help="Override gradient accumulation")
    parser.add_argument('--learning_rate', type=float,
                       help="Override the learning rate")
    parser.add_argument('--lora_rank', type=int,
                       help="Override the LoRA rank")
    parser.add_argument('--lora_alpha', type=int,
                       help="Override the LoRA alpha")
    parser.add_argument('--max_seq_length', type=int,
                       help="Override the maximum tokenized sequence length")
    parser.add_argument('--output_dir', type=str,
                       help="Explicit output directory for checkpoints and final model")
    parser.add_argument('--skip_merge', action='store_true',
                       help="Save the LoRA adapter only and skip the merged 16-bit model")
    parser.add_argument('--report_to', choices=['none', 'tensorboard'], default='tensorboard',
                       help="Trainer reporting backend")
    args = parser.parse_args()
    
    # Process selected tasks
    args.selected_tasks = [task.strip() for task in args.tasks.split(',') if
                         task.strip() in ['retrosynthesis', 'retrosynthesis_class', 'forward_prediction']]
    
    # Process dataset variants
    args.dataset_variants = [v.strip() for v in args.dataset_variants.split(',') if 
                           v.strip() in ['mapped', 'unmapped0', 'unmappedsmile', 'unmappedraw']]
    
    if not args.dataset_variants:
        raise ValueError("At least one dataset variant must be specified.")
    
    return args

def main():
    args = parse_args()

    # Set visibility before importing Unsloth or any model code that can
    # initialize CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    from src.train.config import TrainingConfig
    from src.train.dataset import (
        combine_and_shuffle_datasets,
        load_datasets,
        prepare_datasets,
    )
    from src.train.model import (
        apply_lora,
        create_trainer,
        load_model_and_tokenizer,
        save_model,
        setup_chat_template,
    )
    from src.train.utils import extract_response

    config = TrainingConfig()
    for name in (
        "train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "max_seq_length",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    config.max_steps = args.max_steps
    config.output_dir = args.output_dir
    config.save_merged = not args.skip_merge
    config.report_to = args.report_to
    
    # Load datasets
    train_datasets, val_datasets = load_datasets(args)
    
    # Prepare datasets with appropriate formatting
    prepared_train, prepared_val = prepare_datasets(
        train_datasets,
        val_datasets,
        args.selected_tasks,
        prompt_style=args.prompt_style,
    )
    
    # Combine and shuffle datasets for multi-task training
    combined_train, combined_val = combine_and_shuffle_datasets(prepared_train, prepared_val)
    
    # Load model and tokenizer
    model, tokenizer, _ = load_model_and_tokenizer(args.model_name, config.max_seq_length)
    
    # Apply LoRA
    model = apply_lora(model, config)
    
    # Setup chat template
    tokenizer = setup_chat_template(tokenizer)
    
    # Apply chat template to datasets
    def apply_template(example):
        return {
            "text": tokenizer.apply_chat_template(example["conversations"], tokenize=False, add_generation_prompt=False),
            "response": extract_response(example["conversations"][2]["content"])
        }
    
    combined_train = combined_train.map(apply_template, batched=False)
    combined_val = combined_val.map(apply_template, batched=False)
    
    # Create and run trainer
    trainer = create_trainer(model, tokenizer, combined_train, combined_val, args, config)
    trainer_stats = trainer.train()
    
    # Save model
    saved_paths = save_model(model, tokenizer, args, config)

    metrics_path = Path(saved_paths["run_dir"]) / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "trainer_metrics": trainer_stats.metrics,
                "log_history": trainer.state.log_history,
                "train_rows": len(combined_train),
                "validation_rows": len(combined_val),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Training completed. LoRA saved to {saved_paths['adapter']}")
    if saved_paths["merged"]:
        print(f"Merged 16-bit model saved to {saved_paths['merged']}")
    print(f"Training metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
