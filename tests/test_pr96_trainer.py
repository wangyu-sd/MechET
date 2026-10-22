from copy import deepcopy

import pytest
import torch

from test_qwen35_protocol import TinyCausalModel


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
