"""Train the independent PR116 DPO comparator from authentic native-token pairs."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

from mechet.external_route_data import file_sha, iter_records
from mechet.route_joint_offline_dpo import FORMAT, TRL_VERSION, export_corpus, native_columns
from mechet.route_training_contract import atomic_json


def native_dpo_trainer():
    """Use the pinned TRL collator and likelihood path with already sampled IDs."""
    if version('trl') != TRL_VERSION:
        raise ValueError(f'PR116 offline DPO requires trl=={TRL_VERSION}')
    from trl import DPOTrainer

    class NativeDPOTrainer(DPOTrainer):
        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            if set(dataset.column_names) != {'prompt_ids', 'chosen_ids', 'rejected_ids'}:
                raise ValueError('offline DPO expected only prevalidated native token columns')
            return dataset

    return NativeDPOTrainer


def train(manifest_path, manifest_sha256, output):
    from datasets import Dataset
    import torch
    from accelerate import PartialState
    from peft import PeftModel
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import DPOConfig
    from mechet.qwen35_protocol import OFFICIAL_REVISION, load_language_model
    from mechet.route_joint_schedule import ACTOR_SHA

    trainer_type = native_dpo_trainer()
    if file_sha(manifest_path) != manifest_sha256:
        raise ValueError('PR116 offline DPO corpus binding changed')
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest['format'] != FORMAT or manifest['trl_version'] != TRL_VERSION
            or manifest['source_train_targets'] != 2048 or manifest['beta'] != .1
            or manifest['parent_sha256'] != ACTOR_SHA or manifest['pair_count'] < 1):
        raise ValueError('PR116 offline DPO corpus has incompatible provenance')
    pairs_path = manifest_path.parent/'pairs.jsonl'
    if file_sha(pairs_path) != manifest['pairs_sha256']:
        raise ValueError('PR116 offline DPO preference payload changed')
    if int(os.environ.get('WORLD_SIZE', '0')) != 4 or 'A100' not in torch.cuda.get_device_name(
            int(os.environ['LOCAL_RANK'])):
        raise ValueError('PR116 offline DPO requires four independent A100 ranks')
    distributed = PartialState()
    folder = output/'trainer'
    train_config = dict(beta=.1, loss_type=['sigmoid'], num_train_epochs=1,
        per_device_train_batch_size=1, bf16=True, learning_rate=manifest['learning_rate'],
        seed=manifest['seed'], data_seed=manifest['seed'], save_steps=8, save_total_limit=2,
        logging_steps=1, max_length=None, report_to=[], gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    binding = dict(manifest_sha256=manifest_sha256, trainer=train_config, world_size=4,
                   trl_version=TRL_VERSION, reference_actor_sha256=ACTOR_SHA)
    with distributed.main_process_first():
        output.mkdir(parents=True, exist_ok=True)
        path = output/'binding.json'
        if path.exists():
            if json.loads(path.read_bytes()) != binding:
                raise ValueError('PR116 offline DPO output binding changed')
        elif distributed.is_main_process:
            if folder.exists():
                raise ValueError('PR116 offline DPO found unbound trainer output')
            atomic_json(path, binding)
        if distributed.is_main_process:
            if not manifest['base_files_sha256'] or any(file_sha(name) != digest
                    for name, digest in manifest['base_files_sha256'].items()):
                raise ValueError('PR116 offline DPO base snapshot changed')
            if file_sha(Path(manifest['parent'])/'adapter_model.safetensors') != ACTOR_SHA:
                raise ValueError('PR116 offline DPO starting adapter changed')

    tokenizer = AutoTokenizer.from_pretrained(manifest['base_path'], revision=OFFICIAL_REVISION,
                                               local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [native_columns(p, eos_id=tokenizer.eos_token_id, vocab_size=len(tokenizer),
                           max_context=manifest['max_context']) for p in iter_records(pairs_path)]
    if len(rows) != manifest['pair_count']:
        raise ValueError('PR116 offline DPO pair denominator changed')
    policy, _ = load_language_model(manifest['base_path'], dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(policy, manifest['parent'], is_trainable=True)
    reference, _ = load_language_model(manifest['base_path'], dtype=torch.bfloat16)
    reference = PeftModel.from_pretrained(reference, manifest['parent'], is_trainable=False)
    reference.requires_grad_(False).eval()
    trainer = trainer_type(model=policy, ref_model=reference,
        args=DPOConfig(output_dir=str(folder), **train_config),
        train_dataset=Dataset.from_list(rows), processing_class=tokenizer)
    done = output/'completion.json'
    if done.exists():
        receipt = json.loads(done.read_bytes())
        if (receipt['binding'] != binding or receipt['optimizer_steps'] < 1
                or file_sha(output/'adapter'/'adapter_model.safetensors') != receipt['actor_sha256']):
            raise ValueError('PR116 offline DPO final model binding changed')
        return
    previous = get_last_checkpoint(str(folder)) if folder.is_dir() else None
    result = trainer.train(resume_from_checkpoint=previous)
    if result.global_step < 1:
        raise RuntimeError('PR116 offline DPO did not update the actor')
    trainer.save_model(str(output/'adapter'))
    distributed.wait_for_everyone()
    if distributed.is_main_process:
        actor_sha = file_sha(output/'adapter'/'adapter_model.safetensors')
        atomic_json(done, dict(binding=binding, optimizer_steps=result.global_step,
            metrics=result.metrics, actor_sha256=actor_sha, distinct_training_pairs=len(rows),
            source_roots_with_pairs=manifest['pair_targets'], validation='pending'))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest='mode', required=True)
    exporter = modes.add_parser('export')
    exporter.add_argument('--source', type=Path, required=True)
    exporter.add_argument('--output', type=Path, required=True)
    runner = modes.add_parser('train')
    runner.add_argument('--manifest', type=Path, required=True)
    runner.add_argument('--manifest-sha256', required=True)
    runner.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.mode == 'export':
        print(export_corpus(args.source, args.output))
    else:
        train(args.manifest, args.manifest_sha256, args.output)
