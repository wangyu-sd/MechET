"""Read-only audit of saved route-v8 receipts; no model, rollout or rescoring.

Run with Python's -S flag. Source shared_host_storage.env first. Output contains
aggregates only; raw task identities, molecule structures and prompts stay private.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median


KEYS = ('targets', 'sampled_route_episodes', 'model_decision_requests',
        'completed_reaction_programs', 'native_stock_solved',
        'verified_stock_solved', 'effective_groups', 'censored_groups')
SHARED = Path('/aaa/fionafyang/buddy1').resolve()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def shared(path):
    path = Path(path).resolve()
    if not path.is_relative_to(SHARED):
        raise ValueError('SHARED_STORAGE_REQUIRED')
    return path


def audit(submission, up_to_round, diagnostic_round):
    cfg = load(submission / 'contract.json')
    expected_contract = '07f202555705fcba65ff2b60b8e37a0c9c85aa24fcb5c299340e5fba7ab21262'
    if sha(submission / 'contract.json') != expected_contract:
        raise ValueError('AUDITED_CONTRACT_CHANGED')
    output = shared(cfg['output_dir'])
    expected = dict(train_targets=23730, validation_targets=2986, world_size=4,
                    candidates=4, public_observation_contract='route_action_feedback_v8')
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError('AUDITED_CONDITION_CHANGED')
    manifest = load(Path(__file__).parent / 'source_manifest.json')
    for name, wanted in manifest['files_sha256'].items():
        if sha(Path(__file__).parent / 'source' / name) != wanted:
            raise ValueError('PUBLISHED_SOURCE_CHANGED:' + name)
        if cfg['code_files_sha256'].get(name) != wanted:
            raise ValueError('PUBLISHED_SOURCE_NOT_DEPLOYED:' + name)
    totals = dict.fromkeys(KEYS, 0)
    rounds = []
    for index in range(1, up_to_round + 1):
        checkpoint = output / 'checkpoints' / f'round-{index:06d}'
        complete = load(checkpoint / 'complete.json')
        if complete['contract_sha256'] != expected_contract or complete['next_round'] != index:
            raise ValueError('CHECKPOINT_CONTRACT_OR_CURSOR_MISMATCH')
        metrics = complete['round_metrics']
        if len(metrics) != cfg['world_size']:
            raise ValueError('INCOMPLETE_RANK_METRICS')
        values = {k: sum(row.get(k, 0) for row in metrics) for k in KEYS}
        values.update(round=index, optimizer_step=complete['optimizer_step'],
                      collection_seconds=max(row['sampling_seconds'] for row in metrics))
        rounds.append(values)
        for key in KEYS:
            totals[key] += values[key]
    # Check an immutable chosen checkpoint, even if the live latest pointer advances.
    checkpoint = output / 'checkpoints' / f'round-{up_to_round:06d}'
    complete = load(checkpoint / 'complete.json')
    weight_sha = sha(checkpoint / 'adapter_model.safetensors')
    if weight_sha != complete['payload_sha256']['adapter_model.safetensors']:
        raise ValueError('CHECKPOINT_WEIGHT_CHANGED')
    report = dict(captured_at=datetime.now().astimezone().isoformat(),
                  scope='saved TRAIN evidence; no new model evaluation',
                  contract_sha256=expected_contract, up_to_round=up_to_round,
                  optimizer_step=complete['optimizer_step'], scheduled_rounds=371,
                  full_train_denominator=cfg['train_targets'],
                  full_validation_denominator=cfg['validation_targets'],
                  train_coverage_percent=100 * totals['targets'] / cfg['train_targets'],
                  cumulative=totals, per_round=rounds,
                  median_collection_seconds=median(x['collection_seconds'] for x in rounds),
                  selected_checkpoint_weight_sha256=weight_sha,
                  selected_checkpoint_differs_from_parent=weight_sha != cfg['parent_sha256'],
                  checkpoint_manifest_and_weight_verified=True,
                  validation_summaries_observed=len(list(output.glob('validation/checkpoint-*/summary.json'))),
                  held_out_gain_established=False)
    censored = Counter()
    output_caps = Counter()
    rewards = Counter()
    censored_groups = episodes = groups = 0
    example = None
    paths = sorted((output / f'round-{diagnostic_round - 1:06d}').glob('rank*/rewarded_episodes.json'))
    if len(paths) != cfg['world_size']:
        raise ValueError('INCOMPLETE_DIAGNOSTIC_ROUND')
    for path in paths:
        for group in load(path):
            if len(group) != cfg['candidates']:
                raise ValueError('INCOMPLETE_G4')
            groups += 1
            any_censored = False
            for episode in group:
                episodes += 1
                rewards[str(episode.get('reward'))] += 1
                if episode.get('status') in ('resource_censored', 'protected_censored'):
                    any_censored = True
                    censored[episode.get('error')] += 1
                last = episode['records'][-1]
                if episode.get('error') == 'ACTION_OUTPUT_CONTEXT_LIMIT':
                    prompt = len(last.get('prompt_token_ids', []))
                    completion = len(last.get('generated_token_ids', []))
                    if completion == cfg['max_action_tokens'] and cfg['max_context'] - prompt >= completion:
                        output_caps['single_action_cap'] += 1
                    elif prompt + completion >= cfg['max_context']:
                        output_caps['remaining_physical_context'] += 1
                    else:
                        output_caps['other_truncation'] += 1
                if example is None and episode.get('error') == 'PHYSICAL_CONTEXT_LIMIT':
                    messages = last['messages']
                    system = messages[0]['content']
                    history = system.split('\nCOMPLETE ROUTE EVENT HISTORY (PAST EVENTS)\n', 1)[1]
                    context = json.loads(history.split('\nCURRENT REACTION (AUTHORITATIVE FOR NEXT ACTION)\n', 1)[0])
                    compact = lambda value: len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
                    example = dict(completed_reactions=len(episode['reactions']),
                                   decisions_before_censor=len(episode['records']) - 1,
                                   prompt_token_ids=len(last['prompt_token_ids']),
                                   system_characters=len(system), current_user_characters=len(messages[-1]['content']),
                                   history_component_characters={k: compact(v) for k, v in context.items()},
                                   ledger_component_characters={k: compact(v) for k, v in context['ledger'].items()})
            censored_groups += any_censored
    report['diagnostic_round'] = dict(round=diagnostic_round, groups=groups, episodes=episodes,
                                     censored_groups=censored_groups, censor_reasons=dict(censored),
                                     action_output_censor_breakdown=dict(output_caps), rewards=dict(rewards),
                                     prompt_growth_example=example)
    profile_path = sorted((output / 'runtime_profiles/rank00').glob(f'round-{diagnostic_round:06d}-*.json'))[-1]
    profile = load(profile_path)
    report['diagnostic_rank00_phase_seconds'] = {k: profile['phases'][k]['union_seconds'] for k in
        ('collection.total', 'sampling.generate', 'learner.total', 'checkpoint.publish', 'io.journal_flush')}
    report['profile_scope'] = 'rank00 host-observed phase intervals; nested spans overlap; no device prefill/decode attribution'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission-dir', type=Path, required=True)
    parser.add_argument('--up-to-round', type=int, required=True)
    parser.add_argument('--diagnostic-round', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.diagnostic_round <= args.up_to_round:
        raise ValueError('INVALID_ROUND_RANGE')
    result = audit(shared(args.submission_dir), args.up_to_round, args.diagnostic_round)
    destination = shared(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'aggregate.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(path), optimizer_step=result['optimizer_step'],
                          cumulative=result['cumulative']), ensure_ascii=False))
