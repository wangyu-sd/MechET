# -*- coding: utf-8 -*-
from unsloth import FastLanguageModel
import torch
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from src.train.config import TrainingConfig
from pathlib import Path

def load_model_and_tokenizer(model_name, max_seq_length):
    """Load model and tokenizer with appropriate settings."""
    load_in_4bit = '4bit' in model_name
    dtype = torch.bfloat16
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        local_files_only=True,
    )
    
    return model, tokenizer, load_in_4bit

def apply_lora(model, config):
    """Apply LoRA to the model."""
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

def setup_chat_template(tokenizer):
    """Set up chat template for Qwen-2.5."""
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def create_trainer(model, tokenizer, train_dataset, val_dataset, args, config):
    """Create and configure the SFTTrainer."""
    from transformers import TrainingArguments, TrainerCallback
    from trl import SFTTrainer
    
    class TaskBalanceCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                pass
    
    schedule_steps = calculate_max_steps(train_dataset, config)
    
    tasks_str = '_'.join(args.selected_tasks)
    variants_str = '_'.join(sorted(args.dataset_variants))
    
    if config.output_dir:
        trainer_output_dir = str(Path(config.output_dir) / "checkpoints")
    else:
        trainer_output_dir = (
            f"loras/{args.model_name}_{tasks_str}_{variants_str}_"
            f"{args.prompt_style}_{config.lora_rank}_{config.train_epochs}_sft"
        )

    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=max(0, int(config.warmup_ratio * schedule_steps)),
        num_train_epochs=config.train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=max(1, int(config.logging_ratio * schedule_steps)),
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type="linear",
        seed=config.seed,
        output_dir=trainer_output_dir,
        report_to=[] if config.report_to == "none" else [config.report_to],
        eval_strategy="steps",
        eval_steps=max(1, int(config.eval_ratio * schedule_steps)),
        save_strategy="steps",
        save_steps=max(1, int(config.save_ratio * schedule_steps)),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        packing=False,
        args=training_args,
        callbacks=[TaskBalanceCallback()],
    )
    
    return train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

def calculate_max_steps(dataset, config):
    """Calculate max training steps based on dataset size."""
    if config.max_steps > 0:
        return config.max_steps
    steps_per_epoch = max(
        1,
        len(dataset)
        // (
            config.per_device_train_batch_size
            * config.gradient_accumulation_steps
        ),
    )
    return steps_per_epoch * config.train_epochs

def save_model(model, tokenizer, args, config):
    """Save the trained model in different formats."""
    tasks_str = '_'.join(args.selected_tasks)
    variants_str = '_'.join(sorted(args.dataset_variants))
    saved_path = f"sft/{args.model_name}_{tasks_str}_{variants_str}_{args.prompt_style}_sft_{config.lora_rank}_{config.train_epochs}"
    if config.output_dir:
        run_dir = Path(config.output_dir)
        adapter_path = run_dir / "adapter"
        merged_path = run_dir / "merged"
    else:
        run_dir = Path("loras") / saved_path
        adapter_path = run_dir
        merged_path = Path("merged_models") / saved_path
    
    # Save LoRA adapters
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    
    # Merge and save the model in 16-bit format
    merged = None
    if config.save_merged:
        model.save_pretrained_merged(
            str(merged_path),
            tokenizer,
            save_method="merged_16bit",
        )
        merged = str(merged_path)

    return {
        "run_dir": str(run_dir),
        "adapter": str(adapter_path),
        "merged": merged,
    }
