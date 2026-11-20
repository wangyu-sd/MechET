"""Bind the completed Joint actor and prepare native TRAIN-only distillation targets."""
import json
from collections import Counter
from pathlib import Path

from .external_route_data import file_sha, signature
from .external_route_validation import selection_key


def bind_distillation_source(contract):
    """Select a trained Joint checkpoint from complete 512-root validation receipts."""
    from scripts.train_pr98_route_policy import verify_checkpoint

    contract = Path(contract).resolve()
    cfg = json.loads(contract.read_bytes())
    if cfg.get('step_budget', {}).get('arm') != 'J' or cfg.get('distill_source'):
        raise ValueError('distillation requires a completed Joint comparison arm')
    contract_sha = file_sha(contract)
    root = Path(cfg['output_dir']).resolve()
    completion = root / 'completion.json'
    if not completion.is_file():
        raise ValueError('distillation requires a completed Joint comparison arm')
    completed = json.loads(completion.read_bytes())
    if completed.get('contract_sha256') != contract_sha or completed.get('visited_target_episodes') != 2048:
        raise ValueError('completed Joint run binding mismatch')
    candidates = []
    for step in (32, 64):
        actor_directory = root / 'checkpoints' / f'round-{step:06d}'
        manifest = verify_checkpoint(actor_directory, contract_sha)
        actor_sha = file_sha(actor_directory / 'adapter_model.safetensors')
        if (manifest.get('optimizer_step') != step or manifest.get('next_round') != step
                or manifest['payload_sha256'].get('adapter_model.safetensors') != actor_sha
                or (step == 64 and completed.get('actor_sha256') != actor_sha)):
            raise ValueError('trained Joint actor binding mismatch')
        validation = root / 'validation' / f'checkpoint-{step:06d}' / 'summary.json'
        summary = json.loads(validation.read_bytes())
        if (summary.get('binding', {}).get('contract_sha256') != contract_sha
                or summary.get('actor_sha256') != actor_sha
                or summary.get('checkpoint_round') != step
                or Path(summary.get('checkpoint_path', '')).resolve() != actor_directory
                or summary.get('denominator') != 512
                or summary.get('validation_reference_split') != 'validation'):
            raise ValueError('Joint validation actor binding mismatch')
        candidates.append((selection_key(summary), dict(
            source_contract=str(contract), source_contract_sha256=contract_sha,
            source_completion_sha256=file_sha(completion), actor_directory=str(actor_directory),
            actor_sha256=actor_sha, checkpoint_round=step,
            checkpoint_receipt_sha256=file_sha(actor_directory / 'complete.json'),
            validation_sha256=file_sha(validation))))
    return min(candidates, key=lambda item: item[0])[1]


def prepare_distill_examples(groups, cfg, train_rows, behavior_version, eos_token_id, oracle):
    from .route_self_distill import mix_examples, prepare_state
    from .route_self_distill_runtime import export_search_states

    actor_sha = cfg['distill_source']['actor_sha256']
    train_ids = {r['task_id']: r['target'] for r in train_rows}
    stock_path = str(Path(cfg['dataset']) / 'stock.txt')
    exported = export_search_states(
        groups, actual_actor_sha=actor_sha, training_task_ids=train_ids,
        behavior_version=behavior_version,
        tokenizer_sha=file_sha(Path(cfg['base_path']) / 'tokenizer.json'),
        tool_schema_sha=signature([cfg['public_observation_contract'], cfg['code_files_sha256']]),
        executor_sha=signature(cfg['code_files_sha256']),
        stock_contract_sha=signature([cfg['files_sha256'][stock_path], cfg.get('stock_material_contract')]),
        budget_sha=signature({k: cfg[k] for k in ('step_budget', 'refit', 'candidates', 'max_context',
                                                'max_action_tokens')}),
        reward_contract_sha=signature([cfg['reward_contract'], cfg['code_files_sha256'],
                                       cfg['dataset_manifest_sha256']]), reward_oracle=oracle)
    targets = []
    skipped = Counter()
    for state in exported['states']:
        target = prepare_state(state, exported['receipt_manifest']['receipts'],
                               expected_actor_sha=actor_sha, training_target_ids=train_ids,
                               eos_token_id=eos_token_id)
        if target['status'] == 'prepared':
            targets.append(target)
        else:
            skipped[target['reason']] += 1
    return mix_examples(targets, []), dict(
        exported['statistics'], eligible_states=len(targets), skipped_states=dict(skipped))
