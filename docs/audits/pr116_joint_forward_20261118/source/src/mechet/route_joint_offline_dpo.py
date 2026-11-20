"""Export the reward-only arm's actual TRAIN preferences for independent DPO."""
import json
import os
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from .external_route_data import file_sha, iter_records, signature
from .route_joint_cache import content_key
from .route_joint_forward import group_preferences
from .route_joint_schedule import validate
from .route_training_contract import atomic_json

FORMAT = 'pr116_reward_only_native_dpo_v1'
TRL_VERSION = '1.9.2'


def batch_preferences(groups, source_rows, candidates, behavior_version, teacher, margin):
    if len(groups) != len(source_rows):
        raise ValueError('offline DPO TRAIN rank denominator mismatch')
    pairs = []
    for group, (task_id, target) in zip(groups, source_rows, strict=True):
        if len(group) != candidates:
            raise ValueError('offline DPO TRAIN root candidate mismatch')
        reports = [report for root in group for report in [root, *root['refit']['suffix_reports']]]
        records = {}
        for report in reports:
            if report['task_id'] != task_id or report['root'] != target:
                raise ValueError('offline DPO TRAIN root binding mismatch')
            decisions = report['reward_evidence'].get('forward_joint', {}).get('decisions', [])
            if any(decision['protocol'] != teacher for decision in decisions):
                raise ValueError('offline DPO teacher protocol mismatch')
            for record in report['records']:
                request = record['request_id']
                if record['behavior_version'] != behavior_version or records.setdefault(request, record) != record:
                    raise ValueError('offline DPO behavior receipt mismatch')
        for chosen, rejected in group_preferences(group, margin):
            yes, no = records[chosen], records[rejected]
            if (yes['eos_token_id'] != no['eos_token_id']
                    or yes['prompt_token_ids'] != no['prompt_token_ids']):
                raise ValueError('offline DPO native conditioning mismatch')
            pairs.append(dict(task_id=task_id, behavior_version=behavior_version,
                chosen_request=chosen, rejected_request=rejected, eos_token_id=yes['eos_token_id'],
                prompt_ids=yes['prompt_token_ids'], chosen_ids=yes['generated_token_ids'],
                rejected_ids=no['generated_token_ids']))
    return pairs


def native_columns(pair, *, eos_id, vocab_size, max_context):
    cols = {k: pair[k] for k in ('prompt_ids', 'chosen_ids', 'rejected_ids')}
    if any(not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < vocab_size for i in ids)
           for ids in cols.values()):
        raise ValueError('offline DPO invalid native token IDs')
    prompt, chosen, rejected = (cols[k] for k in ('prompt_ids', 'chosen_ids', 'rejected_ids'))
    if pair['eos_token_id'] != eos_id or chosen[-1] != eos_id or rejected[-1] != eos_id or chosen == rejected:
        raise ValueError('offline DPO native preference or EOS mismatch')
    if len(prompt) + max(len(chosen), len(rejected)) > max_context:
        raise ValueError('offline DPO context would require truncation')
    return cols


def export_corpus(contract, output):
    """Stream complete JR receipts with the protocol and original actor bound to each round."""
    from scripts.train_pr98_route_policy import verify_checkpoint

    contract, output = Path(contract).resolve(), Path(output).resolve()
    cfg = json.loads(contract.read_bytes())
    spec = cfg['step_budget']
    if spec['arm'] != 'JR':
        raise ValueError('offline DPO requires the completed teacher-reward-only arm')
    validate(cfg)
    contract_sha = file_sha(contract)
    root = Path(cfg['output_dir'])
    completion = root/'completion.json'
    done = json.loads(completion.read_bytes())
    if done['contract_sha256'] != contract_sha or done['visited_target_episodes'] != 2048:
        raise ValueError('offline DPO reward-only run completion mismatch')
    terminal = root/'checkpoints'/'round-000064'
    final = verify_checkpoint(terminal, contract_sha)
    if final['optimizer_step'] != 64 or done['actor_sha256'] != final['payload_sha256']['adapter_model.safetensors']:
        raise ValueError('offline DPO reward-only final actor mismatch')
    targets = [(r['task_id'], r['target']) for r in iter_records(spec['roots'])][:2048]
    order_sha = signature([(0, [task for task, _ in targets[i:i+32]]) for i in range(0, 2048, 32)])
    if final['training_order_sha256'] != order_sha or len(set(targets)) != 2048:
        raise ValueError('offline DPO frozen TRAIN stream mismatch')
    opts = cfg['joint_forward']
    teacher = content_key(opts['teacher_protocol'])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix='offline-dpo-corpus-') as temp:
        work = Path(temp)/'corpus'
        work.mkdir()
        starts = cfg['parent_sha256']
        seen = set()
        pair_targets = set()
        receipt_files = {}
        pair_counts = []
        rows = work/'pairs.jsonl'
        with rows.open('w') as stream:
            for i in range(64):
                n = 0
                chunk = dict(targets[32*i:32*(i+1)])
                observed = set()
                version = f'round{i:06d}:{starts}'
                for rank in range(cfg['world_size']):
                    relative = Path(f'round-{i:06d}')/f'rank{rank:02d}'/'joint_rewarded_episodes.json'
                    raw = (root/relative).read_bytes()
                    receipt_files[str(relative)] = sha256(raw).hexdigest()
                    groups = json.loads(raw)
                    assignments = []
                    for group in groups:
                        task = group[0]['task_id']
                        if task not in chunk or task in observed:
                            raise ValueError('offline DPO duplicate or foreign TRAIN root')
                        observed.add(task)
                        assignments.append((task, chunk[task]))
                    rank_pairs = batch_preferences(groups, assignments,
                                                   cfg['candidates'], version, teacher, opts['margin'])
                    for pair in rank_pairs:
                        identity = pair['chosen_request'], pair['rejected_request']
                        if identity in seen:
                            raise ValueError('offline DPO duplicate source preference')
                        seen.add(identity)
                        pair_targets.add(pair['task_id'])
                        stream.write(json.dumps(pair, separators=(',', ':'))+'\n')
                    n += len(rank_pairs)
                if observed != chunk.keys():
                    raise ValueError('offline DPO rank journals did not cover the full TRAIN batch')
                pair_counts.append(n)
                saved = root/'checkpoints'/f'round-{i+1:06d}'
                state = json.loads((saved/'complete.json').read_bytes())
                starts = file_sha(saved/'adapter_model.safetensors')
                if (state['contract_sha256'] != contract_sha or state['next_round'] != i+1
                        or state['optimizer_step'] != i+1 or state['training_order_sha256'] != order_sha
                        or state['payload_sha256']['adapter_model.safetensors'] != starts):
                    raise ValueError('offline DPO source checkpoint binding mismatch')
        if not seen:
            raise ValueError('offline DPO source contains no teacher-scored pairs')
        atomic_json(work/'manifest.json', dict(format=FORMAT, trl_version=TRL_VERSION,
            source_contract=str(contract), source_contract_sha256=contract_sha,
            source_completion_sha256=file_sha(completion), source_receipts_sha256=receipt_files,
            pairs_sha256=file_sha(rows), pair_count=len(seen), pair_targets=len(pair_targets),
            pairs_per_round=pair_counts, source_train_targets=2048, source_teacher_sha256=teacher,
            teacher_protocol=opts['teacher_protocol'], beta=opts['beta_d'], margin=opts['margin'],
            parent=cfg['parent'], parent_sha256=cfg['parent_sha256'], base_path=cfg['base_path'],
            base_files_sha256={name: digest for name, digest in cfg['files_sha256'].items()
                               if Path(name).is_relative_to(cfg['base_path'])},
            max_context=cfg['max_context'], seed=cfg['seed'], learning_rate=cfg['learning_rate']))
        os.rename(work, output)
    return output/'manifest.json'
