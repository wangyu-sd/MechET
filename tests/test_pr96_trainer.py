from copy import deepcopy

import pytest
import torch

from test_qwen35_protocol import TinyCausalModel


def _ddp_gradient_parity(rank, rendezvous, output_dir, microbatches):
    """Two real Gloo ranks, not simulated DDP gradient averaging."""
    import os
    import torch.distributed as dist
    from accelerate.state import AcceleratorState
    from transformers import TrainingArguments
    from mechet.pr96_trainer import PR96Trainer
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT="29597")
    AcceleratorState._reset_state(reset_partial_state=True)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(31)
        model = TinyCausalModel()
        dense_model = deepcopy(model)
        ids = torch.tensor([[1,2,3,4], [4,3,2,1], [5,6,7,8], [8,7,6,5]])
        labels = torch.tensor([[-100,-100,3,4], [-100,3,2,1], [-100,-100,-100,8], [-100,7,6,5]])
        selected = [i for r in range(2) for i in range(r*2, r*2+microbatches)]
        dense = torch.nn.functional.cross_entropy(
            dense_model(ids[selected], slice(None)).logits[:, :-1].reshape(-1, 17),
            labels[selected, 1:].reshape(-1))
        dense.backward()
        args = TrainingArguments(output_dir=output_dir, use_cpu=True, report_to=[],
                                 gradient_accumulation_steps=2, average_tokens_across_devices=True,
                                 ddp_find_unused_parameters=False)
        trainer = PR96Trainer(model=model, args=args)
        wrapped = trainer.accelerator.prepare_model(model)
        trainer.current_gradient_accumulation_steps = microbatches
        denominator = (labels[selected, 1:] != -100).sum()
        for index in range(microbatches):
            i = rank*2 + index
            from contextlib import nullcontext
            context = trainer.accelerator.no_sync(wrapped) if index+1 < microbatches else nullcontext()
            with context:
                trainer.training_step(wrapped, {"input_ids": ids[i:i+1], "labels": labels[i:i+1]}, denominator)
        for (name, actual), expected in zip(model.named_parameters(), dense_model.parameters()):
            assert torch.allclose(actual.grad, expected.grad, atol=1e-7), name
        # Verify an actual optimizer update, not merely a gradient norm.
        torch.optim.SGD(model.parameters(), lr=.1).step()
        torch.optim.SGD(dense_model.parameters(), lr=.1).step()
        for actual, expected in zip(model.parameters(), dense_model.parameters()):
            assert torch.allclose(actual, expected, atol=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("microbatches", [1, 2])
def test_real_two_rank_token_normalized_accumulation_and_update_match_dense(tmp_path, microbatches):
    import torch.multiprocessing as mp
    mp.start_processes(_ddp_gradient_parity,
                       args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), microbatches),
                       nprocs=2, join=True, start_method="spawn")


@pytest.mark.parametrize("microbatches", [1, 2])
def test_accumulated_gradient_equals_dense_ce_including_partial_final_step(tmp_path, microbatches):
    """Catch Accelerator's extra GA division, including a short last window."""
    from mechet.pr96_trainer import PR96Trainer
    from transformers import TrainingArguments
    torch.manual_seed(23)
    model = TinyCausalModel()
    baseline = deepcopy(model)
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])[:microbatches]
    labels = torch.tensor([[-100, -100, 3, 4], [-100, 3, 2, 1]])[:microbatches]
    dense = torch.nn.functional.cross_entropy(
        baseline(ids, slice(None)).logits[:, :-1].reshape(-1, 17), labels[:, 1:].reshape(-1))
    dense.backward()
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
                             gradient_accumulation_steps=2)
    trainer = PR96Trainer(model=model, args=args)
    trainer.current_gradient_accumulation_steps = microbatches
    total_tokens = (labels[:, 1:] != -100).sum()
    losses = [trainer.training_step(model, {"input_ids": ids[i:i+1], "labels": labels[i:i+1]},
                                    num_items_in_batch=total_tokens) for i in range(microbatches)]
    assert torch.allclose(sum(losses), dense.detach(), atol=1e-7)
    for (name, actual), expected in zip(model.named_parameters(), baseline.parameters()):
        assert torch.allclose(actual.grad, expected.grad, atol=1e-7), name


def test_real_trainer_updates_weights_using_selective_causal_loss(tmp_path):
    from mechet.pr96_trainer import PR96Trainer
    from transformers import TrainingArguments
    torch.manual_seed(17)
    model = TinyCausalModel()
    original = {n:p.detach().clone() for n,p in model.named_parameters()}
    rows = [{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]} for _ in range(4)]
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True,
                             per_device_train_batch_size=1, gradient_accumulation_steps=2,
                             num_train_epochs=1, learning_rate=.01, report_to=[],
                             save_strategy="no", logging_steps=1, remove_unused_columns=False)
    trainer = PR96Trainer(model=model, args=args, train_dataset=rows,
                         data_collator=lambda batch: {k:torch.tensor([r[k] for r in batch]) for k in batch[0]})
    trainer.train()
    assert trainer.state.global_step == 2
    assert trainer.examples_consumed == 4
    assert trainer.supervised_tokens_consumed == 8
    assert any(not torch.equal(p, original[n]) for n,p in model.named_parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_training_configuration_cannot_be_shortened_to_max_steps():
    from mechet.pr96_trainer import validate_stage_training
    validate_stage_training(track="mech31k", stage="state", epochs=10, max_steps=-1)
    validate_stage_training(track="flower", stage="history", epochs=3, max_steps=-1)
    with pytest.raises(ValueError, match="FULL_EPOCHS"):
        validate_stage_training(track="mech31k", stage="state", epochs=8, max_steps=-1)
    with pytest.raises(ValueError, match="MAX_STEPS"):
        validate_stage_training(track="flower", stage="state", epochs=5, max_steps=100)


def test_pr96_configuration_cannot_silently_reuse_old_model_or_encoding():
    from mechet.pr96_trainer import validate_pr96_config
    from mechet.qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION
    cfg = {"model_name_or_path": OFFICIAL_MODEL, "pr96": {"track": "mech31k", "stage": "state"},
           "training": {"model_revision": OFFICIAL_REVISION, "num_train_epochs": 10,
                        "encoding_protocol": "qwen35_tool_span_v1", "qlora": False, "bf16": True}}
    validate_pr96_config(cfg, max_steps=-1)
    for key, value in [("model_name_or_path", "Qwen/Qwen3-0.6B"), ("initial_adapter_path", "old-adapter")]:
        bad = deepcopy(cfg)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_pr96_config(bad, max_steps=-1)
    bad = deepcopy(cfg)
    bad["training"]["encoding_protocol"] = "legacy_chatml_v1"
    with pytest.raises(ValueError, match="ENCODING"):
        validate_pr96_config(bad, max_steps=-1)


def test_nonfinite_metric_is_persisted_and_fails_instead_of_continuing(tmp_path):
    from types import SimpleNamespace
    import json
    from transformers import TrainingArguments
    from transformers.trainer_callback import TrainerState
    from mechet.pr96_trainer import PR96Progress
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    state = TrainerState(global_step=10, max_steps=100, epoch=.1, is_world_process_zero=True)
    callback = PR96Progress(SimpleNamespace(_counts=torch.zeros(3, dtype=torch.long)),
                            track="mech31k", stage="state", lineage={})
    with pytest.raises(FloatingPointError, match="NONFINITE"):
        callback.on_log(args, state, None, logs={"loss": float("nan")})
    events = [json.loads(l) for l in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "nonfinite-training-metric"
    assert events[-1]["global_step"] == 10


def test_checkpoint_counters_resume_instead_of_resetting_and_keep_throughput_session_local(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import json
    from transformers import TrainingArguments
    from transformers.trainer_callback import TrainerState
    from mechet.pr96_trainer import PR96Progress
    # A GPU-equipped host may deliberately run a CPU verifier. Availability
    # must not make the callback query CUDA memory for args.device='cpu'.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda.memory, "is_initialized", lambda: True)
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "pr96_counters.rank00.json").write_text(json.dumps({"counts": [200, 4000, 8000]}))
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    trainer = SimpleNamespace(_counts=torch.zeros(3, dtype=torch.long))
    callback = PR96Progress(trainer, track="mech31k", stage="state", lineage={}, resume_checkpoint=checkpoint)
    state = TrainerState(global_step=10, max_steps=100, epoch=1, is_world_process_zero=True)
    callback.on_train_begin(args, state, None)
    trainer._counts += torch.tensor([5, 100, 200])
    callback.on_log(args, state, None, logs={"loss": 1.0})
    record = json.loads((tmp_path / "progress.json").read_text())
    assert record["examples_consumed_global"] == 205
    assert record["supervised_tokens_consumed_global"] == 4100
    assert record["latest_checkpoint"] == str(checkpoint)
    assert record["supervised_tokens_per_second"] / record["samples_per_second"] == pytest.approx(20)


def test_real_peft_optimizer_rng_and_counters_resume_match_uninterrupted_training(tmp_path):
    """Exercise HF's actual resume loader, not just existence of its files."""
    from peft import get_peft_model, LoraConfig
    from transformers import TrainingArguments, TrainerCallback
    from mechet.pr96_trainer import PR96Trainer, PR96Progress
    torch.manual_seed(41)
    initial = get_peft_model(TinyCausalModel(), LoraConfig(target_modules=["head"], r=2,
                                                         lora_alpha=2, lora_dropout=.1))
    rows = [{"input_ids": [i+1, i+2, i+3, i+4], "labels": [-100, -100, i+3, i+4]}
            for i in range(8)]
    def make(model, output, resume=None):
        args = TrainingArguments(output_dir=str(output), use_cpu=True, report_to=[],
                                 per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                 num_train_epochs=2, learning_rate=.01, seed=17, data_seed=17,
                                 remove_unused_columns=False, save_strategy="epoch", logging_steps=1)
        trainer = PR96Trainer(model=model, args=args, train_dataset=rows,
                             data_collator=lambda batch: {k:torch.tensor([r[k] for r in batch]) for k in batch[0]})
        trainer.add_callback(PR96Progress(trainer, track="mech31k", stage="state", lineage={},
                                          resume_checkpoint=resume))
        return trainer
    full = make(deepcopy(initial), tmp_path / "continuous")
    full.train()
    assert full.optimizer.state_dict()["state"]
    assert any(not torch.equal(value, initial.state_dict()[name])
               for name, value in full.model.state_dict().items() if "lora_" in name)
    class StopAtFirstEpoch(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step == 4:
                control.should_training_stop = True
    interrupted = make(deepcopy(initial), tmp_path / "interrupted")
    interrupted.add_callback(StopAtFirstEpoch())
    interrupted.train()
    assert interrupted.state.global_step == 4
    checkpoint = tmp_path / "interrupted/checkpoint-4"
    resumed = make(deepcopy(initial), tmp_path / "interrupted", checkpoint)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == full.state.global_step == 8
    assert resumed.examples_consumed == full.examples_consumed == 16
    assert resumed.supervised_tokens_consumed == full.supervised_tokens_consumed == 32
    for name, actual in resumed.model.state_dict().items():
        assert torch.allclose(actual, full.model.state_dict()[name], atol=1e-7), name
    for key, state in resumed.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            expected = full.optimizer.state_dict()["state"][key][name]
            assert torch.allclose(value, expected, atol=1e-7), name
    assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
