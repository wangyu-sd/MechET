"""Performance changes must preserve exact rows, sampler order and objective."""
from copy import deepcopy

import pytest
import torch


def test_arrow_lengths_respect_slices_indices_and_never_decode_token_rows(monkeypatch):
    from datasets import Dataset, concatenate_datasets
    from mechet.pr96_performance import arrow_sequence_lengths
    left = Dataset.from_dict({"input_ids": [[1], [2, 3], [4, 5, 6]],
                              "labels": [[1], [2, 3], [4, 5, 6]]})
    data = concatenate_datasets([left, left]).select([5, 0, 4, 5])
    original_columns = data.column_names[:]
    def forbid_decode(*args, **kwargs):
        raise AssertionError("Token rows must not be expanded into Python")
    monkeypatch.setattr(Dataset, "__iter__", forbid_decode)
    monkeypatch.setattr(Dataset, "__getitem__", forbid_decode)
    assert arrow_sequence_lengths(data).tolist() == [3, 1, 2, 3]
    assert data.column_names == original_columns
    sliced = left.select(range(1, 3))
    assert arrow_sequence_lengths(sliced).tolist() == [2, 3]


@pytest.mark.parametrize("rows", [[[1], []], [[1], None]])
def test_arrow_lengths_reject_empty_or_null_sequences(rows):
    from datasets import Dataset
    from mechet.pr96_performance import arrow_sequence_lengths
    with pytest.raises(ValueError, match="LENGTH"):
        arrow_sequence_lengths(Dataset.from_dict({"input_ids": rows}))


def test_memory_probe_uses_actual_longest_arrow_row_from_both_stages(tmp_path):
    import json
    from datasets import Dataset
    from mechet.pr96_performance import largest_cached_example
    caches = []
    for name, sizes in [("state", [3, 5]), ("history", [7, 4])]:
        cache = tmp_path / name
        cache.mkdir()
        data = Dataset.from_dict({"input_ids": [[1] * n for n in sizes],
                                  "labels": [[-100] + [1] * (n - 1) for n in sizes]})
        data.save_to_disk(str(cache / "data"))
        shard = next((cache / "data").glob("*.arrow"))
        (cache / "manifest.json").write_text(json.dumps({"splits": {"train": {
            "arrow_files": [str(shard)], "n_rows": 2, "max_input_tokens": max(sizes)}}}))
        # The on-disk manifest may hold original shared paths; bootstrap copies
        # the shard into the node-local cache root and must use that local copy.
        import shutil
        shutil.copy2(shard, cache / shard.name)
        caches.append(cache)
    row, audit = largest_cached_example(caches)
    assert len(row["input_ids"]) == 7
    assert row["labels"] == [-100, 1, 1, 1, 1, 1, 1]
    assert audit["maximum_input_tokens"] == 7
    assert audit["cache_count"] == 2


def test_pr96_sampler_keeps_native_order_without_python_token_decoding(tmp_path, monkeypatch):
    from datasets import Dataset
    from transformers import TrainingArguments
    from transformers.trainer_pt_utils import LengthGroupedSampler
    from mechet.pr96_trainer import PR96Trainer
    from test_qwen35_protocol import TinyCausalModel
    data = Dataset.from_dict({"input_ids": [[1] * n for n in [4, 1, 5, 2, 3, 6]],
                              "labels": [[1] * n for n in [4, 1, 5, 2, 3, 6]]})
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
                             per_device_train_batch_size=2, gradient_accumulation_steps=2,
                             train_sampling_strategy="group_by_length")
    trainer = PR96Trainer(model=TinyCausalModel(), args=args, train_dataset=data)
    torch.manual_seed(17)
    expected = list(LengthGroupedSampler(4, lengths=[4, 1, 5, 2, 3, 6]))
    def forbid_decode(*args, **kwargs):
        raise AssertionError("Native token-by-token length inference must be bypassed")
    monkeypatch.setattr(Dataset, "__iter__", forbid_decode)
    monkeypatch.setattr(Dataset, "__getitem__", forbid_decode)
    torch.manual_seed(17)
    assert list(trainer._get_train_sampler()) == expected
    assert sorted(expected) == [0, 1, 2, 3, 4, 5]


def test_performance_migration_preserves_global_batch_and_rejects_scientific_changes():
    from mechet.pr96_performance import validate_performance_identity_change
    source = {"model": "same", "training": {"per_device_train_batch_size": 4,
              "gradient_accumulation_steps": 2, "gradient_checkpointing": True,
              "learning_rate": .0001, "num_train_epochs": 5, "seed": 17},
              "lora": {"r": 16}, "contract": {"expected_train_rows": 1908035},
              "bindings": {"track": "flower", "stage": "state", "cohorts_sha256": "same"},
              "code_sha256": {"src/mechet/pr96_trainer.py": "old", "src/mechet/qwen35_protocol.py": "fixed"}}
    target = deepcopy(source)
    target["training"].update(per_device_train_batch_size=8, gradient_accumulation_steps=1,
                              gradient_checkpointing=False)
    target["code_sha256"].update({"src/mechet/pr96_trainer.py": "new",
                                   "src/mechet/pr96_performance.py": "new"})
    validate_performance_identity_change(source, target)
    for section, key, value in [("training", "learning_rate", .1), ("training", "num_train_epochs", 1),
                                ("training", "seed", 42), ("training", "per_device_train_batch_size", 16),
                                ("lora", "r", 32), ("contract", "expected_train_rows", 12),
                                ("bindings", "cohorts_sha256", "other"),
                                ("code_sha256", "src/mechet/qwen35_protocol.py", "changed")]:
        bad = deepcopy(target); bad[section][key] = value
        with pytest.raises(ValueError, match="PERFORMANCE"):
            validate_performance_identity_change(source, bad)


def test_resume_requires_explicit_hashed_migration_and_keeps_optimizer_checks(tmp_path):
    import json
    from mechet.pr96_artifacts import training_identity, verify_resume_checkpoint
    from mechet.pr96_performance import make_performance_resume_authorization
    from mechet.pr96_eval_runtime import sha
    from test_pr96_artifact_gates import checkpoint_fixture
    cfg, checkpoint = checkpoint_fixture(tmp_path)
    cfg["training"].update(per_device_train_batch_size=4, gradient_accumulation_steps=2,
                            gradient_checkpointing=True, num_train_epochs=5)
    metadata_path = checkpoint / "pr96_checkpoint_manifest.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["lineage"]["training_identity"] = training_identity(cfg)
    metadata_path.write_text(json.dumps(metadata))
    target = deepcopy(cfg)
    target["training"].update(per_device_train_batch_size=8, gradient_accumulation_steps=1,
                               gradient_checkpointing=False)
    with pytest.raises(ValueError, match="RESUME"):
        verify_resume_checkpoint(target, checkpoint)
    receipt = make_performance_resume_authorization(target, checkpoint)
    authorization = tmp_path / "migration.json"
    authorization.write_text(json.dumps(receipt))
    target["pr96"]["performance_resume_authorization"] = {"path": str(authorization), "sha256": sha(authorization)}
    value = verify_resume_checkpoint(target, checkpoint)
    assert value["step"] == 12 and value["epoch"] == 1
    (checkpoint / "optimizer.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="RESUME_COMPONENT_HASH"):
        verify_resume_checkpoint(target, checkpoint)


def test_recovery_save_request_does_not_wait_for_an_entire_epoch(tmp_path):
    from types import SimpleNamespace
    from mechet.pr96_trainer import PR96Progress
    trainer = SimpleNamespace(_counts=torch.zeros(3, dtype=torch.long), args=SimpleNamespace(process_index=0))
    callback = PR96Progress(trainer, track="flower", stage="state",
                            lineage={"recovery_checkpoint_steps": 1000})
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_step_end(None, SimpleNamespace(global_step=999), control)
    assert not control.should_save
    callback.on_step_end(None, SimpleNamespace(global_step=1000), control)
    assert control.should_save and not control.should_training_stop


def test_history_recovery_checkpoint_does_not_trigger_extra_mid_epoch_generation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mechet.pr96_trainer import PR96Progress
    from test_pr96_artifact_gates import checkpoint_fixture
    cfg, checkpoint = checkpoint_fixture(tmp_path)
    trainer = SimpleNamespace(_counts=torch.tensor([12, 123, 456]), args=SimpleNamespace(process_index=0))
    callback = PR96Progress(trainer, track="flower", stage="history", lineage={})
    def forbid_mid_epoch_generation(*args, **kwargs):
        raise AssertionError("Recovery-only saves must not launch model evaluation")
    monkeypatch.setattr(callback, "_validate_epoch", forbid_mid_epoch_generation)
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0, world_size=8)
    state = SimpleNamespace(global_step=12, epoch=.25, is_world_process_zero=True,
                            best_model_checkpoint=None, max_steps=48)
    args.num_train_epochs = 1
    callback.on_save(args, state, SimpleNamespace())
    assert callback.last_checkpoint == str(checkpoint)


def test_migration_authorization_cannot_bypass_new_learning_rate_or_changed_receipt(tmp_path):
    import json
    from mechet.pr96_artifacts import training_identity, verify_resume_checkpoint
    from mechet.pr96_performance import make_performance_resume_authorization
    from mechet.pr96_eval_runtime import sha
    from test_pr96_artifact_gates import checkpoint_fixture
    cfg, checkpoint = checkpoint_fixture(tmp_path)
    cfg["training"].update(per_device_train_batch_size=4, gradient_accumulation_steps=2,
                            learning_rate=.0001)
    path = checkpoint / "pr96_checkpoint_manifest.json"
    metadata = json.loads(path.read_text()); metadata["lineage"]["training_identity"] = training_identity(cfg)
    path.write_text(json.dumps(metadata))
    bad = deepcopy(cfg); bad["training"]["learning_rate"] = .5
    with pytest.raises(ValueError, match="PERFORMANCE"):
        make_performance_resume_authorization(bad, checkpoint)
    target = deepcopy(cfg); target["training"].update(per_device_train_batch_size=8, gradient_accumulation_steps=1)
    receipt = make_performance_resume_authorization(target, checkpoint)
    auth = tmp_path / "migration.json"; auth.write_text(json.dumps(receipt))
    target["pr96"]["performance_resume_authorization"] = {"path": str(auth), "sha256": sha(auth)}
    auth.write_text(auth.read_text() + " ")
    with pytest.raises(ValueError, match="RESUME"):
        verify_resume_checkpoint(target, checkpoint)


@pytest.mark.parametrize("stop_step", [2, 6])
def test_actual_resume_with_changed_microbatch_retains_optimizer_schedule_and_data(tmp_path, stop_step):
    """Real HF/PEFT recovery, including mid-epoch and partial last global batch."""
    from datasets import Dataset
    from peft import get_peft_model, LoraConfig
    from transformers import TrainingArguments, TrainerCallback
    from mechet.pr96_trainer import PR96Trainer, PR96Progress
    from test_qwen35_protocol import TinyCausalModel
    torch.manual_seed(41)
    initial = get_peft_model(TinyCausalModel(), LoraConfig(
        target_modules=["head"], r=2, lora_alpha=2, lora_dropout=0))
    rows = [{"input_ids": [i + 1, i + 2, i + 3, i + 4],
             "labels": [-100, -100, i + 3, i + 4]} for i in range(11)]
    dataset = Dataset.from_list(rows)
    def make(output, micro, accum, resume=None):
        args = TrainingArguments(output_dir=str(output), use_cpu=True, report_to=[],
            per_device_train_batch_size=micro, gradient_accumulation_steps=accum,
            train_sampling_strategy="group_by_length", num_train_epochs=2,
            learning_rate=.01, seed=17, data_seed=17, remove_unused_columns=False,
            save_strategy="epoch", logging_steps=1)
        trainer = PR96Trainer(model=deepcopy(initial), args=args, train_dataset=dataset,
            data_collator=lambda batch: {k: torch.tensor([r[k] for r in batch]) for k in batch[0]})
        trainer.add_callback(PR96Progress(trainer, track="flower", stage="state",
            lineage={"recovery_checkpoint_steps": 2}, resume_checkpoint=resume))
        return trainer
    continuous = make(tmp_path / "continuous", 1, 2)
    continuous.train()
    class Interrupt(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step == stop_step:
                control.should_training_stop = True
    interrupted = make(tmp_path / "interrupted", 1, 2)
    interrupted.add_callback(Interrupt())
    interrupted.train()
    checkpoint = tmp_path / "interrupted" / f"checkpoint-{stop_step}"
    recovered = make(tmp_path / "interrupted", 2, 1, checkpoint)
    recovered.train(resume_from_checkpoint=str(checkpoint))
    assert recovered.state.global_step == continuous.state.global_step == 12
    assert recovered.examples_consumed == continuous.examples_consumed == 22
    assert recovered.supervised_tokens_consumed == continuous.supervised_tokens_consumed == 44
    for name, actual in recovered.model.state_dict().items():
        torch.testing.assert_close(actual, continuous.model.state_dict()[name], atol=1e-7, rtol=1e-6)
    for key, values in recovered.optimizer.state_dict()["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, continuous.optimizer.state_dict()["state"][key][name],
                                       atol=1e-7, rtol=1e-6)
    assert recovered.lr_scheduler.state_dict() == continuous.lr_scheduler.state_dict()
