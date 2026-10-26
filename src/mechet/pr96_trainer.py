"""Selective exact-CE Trainer and persistent real training telemetry for PR96."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import Trainer, TrainerCallback

from .qwen35_protocol import supervised_loss

MIN_EPOCHS = {("mech31k", "state"): 10, ("mech31k", "history"): 5,
              ("flower", "state"): 5, ("flower", "history"): 3}


def validate_stage_training(*, track, stage, epochs, max_steps):
    if (track, stage) not in MIN_EPOCHS:
        raise ValueError("UNKNOWN_PR96_TRACK_OR_STAGE")
    if max_steps > 0:
        raise ValueError("PR96_MAX_STEPS_OVERRIDE_FORBIDDEN")
    if epochs < MIN_EPOCHS[(track, stage)]:
        raise ValueError("PR96_FULL_EPOCHS_REQUIRED")


def validate_pr96_config(cfg, *, max_steps):
    from .qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION
    settings, training = cfg["pr96"], cfg["training"]
    validate_stage_training(track=settings["track"], stage=settings["stage"],
                            epochs=float(training["num_train_epochs"]), max_steps=max_steps)
    if cfg.get("model_name_or_path") != OFFICIAL_MODEL or training.get("model_revision") != OFFICIAL_REVISION:
        raise ValueError("PR96_OFFICIAL_MODEL_AND_REVISION_REQUIRED")
    if training.get("encoding_protocol") != "qwen35_tool_span_v1":
        raise ValueError("PR96_MATCHED_ENCODING_REQUIRED")
    if training.get("qlora") or not training.get("bf16") or training.get("fp16", False):
        raise ValueError("PR96_BF16_NONQUANTIZED_LORA_REQUIRED")
    if settings["stage"] == "state" and cfg.get("initial_adapter_path"):
        raise ValueError("PR96_STAGE_ONE_MUST_START_FROM_OFFICIAL_BASE")
    if settings["stage"] == "history" and not cfg.get("initial_adapter_path"):
        raise ValueError("PR96_STAGE_TWO_PARENT_REQUIRED")


def validate_deployed_hardware(track):
    import os
    expected = "A100" if track == "mech31k" else "H20"
    if int(os.environ.get("WORLD_SIZE", "1")) != 8 or torch.cuda.device_count() != 8:
        raise ValueError("PR96_EXACTLY_EIGHT_DDP_GPUS_REQUIRED")
    names = [torch.cuda.get_device_name(i) for i in range(8)]
    if not all(expected in name for name in names):
        raise ValueError(f"PR96_WRONG_GPU_TYPE:{expected}:{names}")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not torch.cuda.is_bf16_supported():
        raise ValueError("PR96_BF16_NOT_SUPPORTED")
    return names


class PR96Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We implement the actual accumulation-wide token denominator. Prevent
        # Trainer from dividing by gradient_accumulation_steps a second time.
        self.model_accepts_loss_kwargs = True
        # Trainer owns microbatch grouping, no_sync and optimizer scheduling.
        # TF5.5 passes GA into Accelerate, whose backward() would otherwise
        # divide our already accumulation-normalized loss again (also on a
        # partial final group). Keep its backward scaling at identity.
        self.accelerator.gradient_accumulation_steps = 1
        self._counts = torch.zeros(3, dtype=torch.long, device=self.args.device)

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self.args.train_sampling_strategy == "group_by_length" and hasattr(dataset, "data"):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            from .pr96_performance import arrow_sequence_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=arrow_sequence_lengths(dataset),
            )
        return super()._get_train_sampler(train_dataset)

    def _get_eval_sampler(self, eval_dataset):
        if self.args.train_sampling_strategy == "group_by_length" and hasattr(eval_dataset, "data"):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            from .pr96_performance import arrow_sequence_lengths
            return LengthGroupedSampler(self.args.eval_batch_size,
                                        lengths=arrow_sequence_lengths(eval_dataset))
        return super()._get_eval_sampler(eval_dataset)

    @property
    def examples_consumed(self):
        return int(self._counts[0].item())

    @property
    def supervised_tokens_consumed(self):
        return int(self._counts[1].item())

    def training_step(self, model, inputs, num_items_in_batch=None):
        ids, labels = inputs["input_ids"], inputs["labels"]
        self._counts[0] += ids.shape[0]
        self._counts[1] += (labels[:, 1:] != -100).sum().to(self._counts.device)
        mask = inputs.get("attention_mask")
        self._counts[2] += (mask.sum() if mask is not None else ids.numel())
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, output = supervised_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            # DDP averages gradients; the denominator above already sums tokens
            # across ranks. This factor cancels that extra gradient average.
            loss = loss * self.accelerator.num_processes
        return (loss, {"loss": loss, "logits": output.logits}) if return_outputs else loss


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PR96Progress(TrainerCallback):
    """All ranks contribute counters; only rank zero writes aggregated logs."""
    def __init__(self, trainer, *, track, stage, lineage, resume_checkpoint=None):
        self.trainer, self.track, self.stage = trainer, track, stage
        self.lineage = dict(lineage)
        self.started = self.last_time = time.monotonic()
        self.last_counts = [0, 0, 0]
        self.last_checkpoint = str(resume_checkpoint) if resume_checkpoint else None
        if resume_checkpoint:
            rank = trainer.args.process_index if hasattr(trainer, "args") else 0
            saved = json.loads((Path(resume_checkpoint) / f"pr96_counters.rank{rank:02d}.json").read_text())
            trainer._counts.copy_(torch.tensor(saved["counts"], device=trainer._counts.device))

    def on_train_begin(self, args, state, control, **kwargs):
        counts = self.trainer._counts.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
        self.last_counts = counts.tolist()
        self.last_time = time.monotonic()
        if self.stage == "history" and self.last_checkpoint and self._at_epoch_boundary(state):
            self._validate_epoch(args, state, Path(self.last_checkpoint))

    @staticmethod
    def _at_epoch_boundary(state):
        epoch = float(state.epoch or 0)
        return epoch > 0 and abs(epoch - round(epoch)) < 1e-6

    def on_step_end(self, args, state, control, **kwargs):
        interval = self.lineage.get("recovery_checkpoint_steps", 0)
        if interval and (type(interval) is not int or interval < 1):
            raise ValueError("PR96_INVALID_RECOVERY_CHECKPOINT_INTERVAL")
        if interval and state.global_step > 0 and state.global_step % interval == 0:
            control.should_save = True
        return control

    def _emit(self, args, state, event):
        if not state.is_world_process_zero:
            return
        event.update(timestamp=datetime.now(timezone.utc).isoformat(), track=self.track,
                     stage=self.stage, elapsed_seconds=round(time.monotonic() - self.started, 3),
                     global_step=state.global_step, total_steps=state.max_steps,
                     epoch=state.epoch, epochs=args.num_train_epochs,
                     percent=100 * state.global_step / max(state.max_steps, 1))
        path = Path(args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        text = json.dumps(event, default=str)
        print(text, flush=True)
        with (path / "progress.jsonl").open("a") as out:
            out.write(text + "\n")
            out.flush()
        temporary = path / "progress.json.building"
        temporary.write_text(text + "\n")
        temporary.replace(path / "progress.json")

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm", "eval_loss"):
            value = (logs or {}).get(key)
            if value is not None and not math.isfinite(float(value)):
                self._emit(args, state, {"event": "nonfinite-training-metric", "metric": key,
                                         "value": str(value), "last_checkpoint": self.last_checkpoint})
                raise FloatingPointError(f"PR96_NONFINITE_METRIC:{key}:{value}")
        counts = self.trainer._counts.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        values = counts.tolist()
        now = time.monotonic()
        seconds = max(now - self.last_time, 1e-9)
        tokens_per_sec = (values[1] - self.last_counts[1]) / seconds
        samples_per_sec = (values[0] - self.last_counts[0]) / seconds
        memory = {}
        if args.device.type == "cuda" and torch.cuda.is_available():
            device = args.device
            memory = {"memory_rank": args.local_rank,
                      "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(device),
                      "gpu_memory_peak_bytes": torch.cuda.max_memory_allocated(device)}
        self._emit(args, state, {"event": "training-log", "metrics": dict(logs or {}),
                   "examples_consumed_global": values[0], "supervised_tokens_consumed_global": values[1],
                   "input_tokens_consumed_global": values[2],
                   "samples_per_second": samples_per_sec, "supervised_tokens_per_second": tokens_per_sec,
                   "latest_checkpoint": self.last_checkpoint, "best_checkpoint": state.best_model_checkpoint,
                   "recent_exceptions": 0, **memory})
        self.last_time, self.last_counts = now, values

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._emit(args, state, {"event": "epoch-validation", "metrics": dict(metrics or {}),
                   "best_checkpoint": state.best_model_checkpoint, "selection_criterion": "eval_loss_min",
                   "next_validation": "next full epoch end"})

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        (checkpoint / f"pr96_counters.rank{args.process_index:02d}.json").write_text(
            json.dumps({"counts": self.trainer._counts.tolist()}) + "\n")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.last_checkpoint = str(checkpoint)
        if state.is_world_process_zero:
            required = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                        "scheduler.pt", "trainer_state.json"]
            for filename in required:
                if not (checkpoint / filename).is_file():
                    raise ValueError(f"INCOMPLETE_RESUMABLE_PR96_CHECKPOINT:{checkpoint / filename}")
            rng = list(checkpoint.glob("rng_state*.pth"))
            if len(rng) != args.world_size:
                raise ValueError("PR96_CHECKPOINT_RNG_RANKS_MISSING")
            manifest = {"track": self.track, "stage": self.stage, "lineage": self.lineage,
                        "step": state.global_step, "epoch": state.epoch,
                        "files": {name: _sha(checkpoint / name) for name in required},
                        "rng_files": {p.name: _sha(p) for p in rng},
                        "counter_files": {p.name: _sha(p) for p in checkpoint.glob("pr96_counters.rank*.json")},
                        "best_checkpoint": state.best_model_checkpoint}
            metadata_path = checkpoint / "pr96_checkpoint_manifest.json"
            temporary = metadata_path.with_suffix(".json.building")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(metadata_path)
            self._emit(args, state, {"event": "checkpoint-saved", "checkpoint": str(checkpoint),
                       "adapter_model_sha256": manifest["files"]["adapter_model.safetensors"],
                       "optimizer_scheduler_rng_saved": True, "best_checkpoint": state.best_model_checkpoint})
        if self.stage == "history" and self._at_epoch_boundary(state):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()  # rank-zero manifest must exist before all ranks audit it
            self._validate_epoch(args, state, checkpoint)

    def _validate_epoch(self, args, state, checkpoint):
        # Recover interrupted callback evaluation before continuing training.
        import yaml
        from .pr96_eval_runtime import run_validation, load_validation_report
        cfg = yaml.safe_load(Path(self.lineage["config_file"]).read_text())
        target = Path(args.output_dir) / "epoch_validation" / f"step-{state.global_step}"
        if load_validation_report(cfg, checkpoint, target, "paired") is None:
            run_validation(cfg, self.trainer.model, self.trainer.processing_class,
                           adapter=checkpoint, output_dir=target, cohort="paired")
        if state.is_world_process_zero:
            metadata_path = checkpoint / "pr96_checkpoint_manifest.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["epoch_validation_report"] = str(target / "report.json")
            temporary = metadata_path.with_suffix(".json.building")
            temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(metadata_path)
