#!/usr/bin/env python3
"""Eight-shard, product-only K=1 evaluation of the frozen 31k causal-import policy.

This reuses the already-audited natural-language closed-loop executor and
generation code. Gold precursors are read only after generation for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


SHARED = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
SOURCE = SHARED / 'data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/test.jsonl'
HISTORY = SHARED / 'data/mech_uspto_31k_causal_import_v1/history/test.jsonl'
DATASET = HISTORY.parents[1]
ADAPTER = SHARED / 'outputs/agent/mech_uspto31k_causal_import_stage2_continuation_h20_seed17_20260930'
EXPECTED_ADAPTER_SHA = 'f5589166bbc1dd6dc884e14e521dccff4e4af39d5342fb63cc6fa57d460e4456'
EXPECTED_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
DEFAULT_OUTPUT = SHARED / 'outputs/eval/mech_uspto31k_causal_stage2_test1253_product_k1_20261007'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def smoke_module():
    path = SHARED / 'scripts/smoke_nl_history_stratified.py'
    spec = importlib.util.spec_from_file_location('mechet_frozen_nl_smoke', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selection(smoke):
    source_rows, history_rows, chosen, _ = smoke.select(SOURCE, HISTORY, 1_000_000, 17)
    if len(chosen) != 1253 or len(source_rows) != 1253 or len(history_rows) != 1253:
        raise ValueError('31k executable trace-view test must have exactly 1,253 IDs')
    return source_rows, history_rows, chosen


def fingerprint(max_new_tokens: int, max_decisions: int, world_size: int) -> dict:
    manifest = json.loads((DATASET / 'manifest.json').read_text())
    status = json.loads((DATASET / 'ARTIFACT_STATUS.json').read_text())
    adapter_manifest = json.loads((ADAPTER / 'adapter_manifest.json').read_text())
    if manifest['reaction_denominator'] != {'train': 10152, 'valid': 1319, 'test': 1253}:
        raise ValueError('wrong executable trace-view denominator')
    if manifest['full_reaction_denominator'] != {'train': 24959, 'valid': 3120, 'test': 3120}:
        raise ValueError('wrong full reaction denominator')
    if not status['training_allowed'] or status['status'] != 'validated_trace_view':
        raise ValueError('causal-import artifact is not validated')
    if digest(SOURCE) != manifest['splits']['test']['source_sha256']:
        raise ValueError('source test bytes changed')
    if digest(HISTORY) != manifest['splits']['test']['history']['output_sha256']:
        raise ValueError('history test bytes changed')
    if digest(ADAPTER / 'adapter_model.safetensors') != EXPECTED_ADAPTER_SHA:
        raise ValueError('final adapter weights changed')
    if adapter_manifest['condition_name'] != 'mech_uspto31k_causal_import_stage2_continuation_v1':
        raise ValueError('wrong trained condition')
    if adapter_manifest['base_model_revision'] != EXPECTED_REVISION:
        raise ValueError('wrong Qwen3-8B revision')
    return {
        'dataset': 'mech_uspto_31k_current_compiler_executable_trace_view',
        'denominator': 1253,
        'full_endpoint_denominator_not_evaluated': 3120,
        'source_sha256': digest(SOURCE),
        'history_sha256': digest(HISTORY),
        'adapter_weights_sha256': EXPECTED_ADAPTER_SHA,
        'base_revision': EXPECTED_REVISION,
        'input': 'product_only',
        'decode': 'greedy_k1',
        'max_new_tokens_per_decision': max_new_tokens,
        'max_decisions': max_decisions,
        'max_imports': 32,
        'world_size': world_size,
        'seed': 17,
    }


def ensure_manifest(output: Path, expected: dict) -> None:
    path = output / 'run_manifest.json'
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('existing output belongs to a different evaluation')
    else:
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + '\n')


def shard(identifier: str, world_size: int) -> int:
    return int(hashlib.sha256(f'17:{identifier}'.encode()).hexdigest(), 16) % world_size


def preflight(smoke, source_rows, history_rows, chosen):
    from mechet.in_place_grounded_flow import deterministic_unmapped_state
    from scripts.build_natural_language_event_sft import SYSTEM, TOOLS

    for identifier in chosen:
        row = source_rows[identifier]
        first = history_rows[identifier][0]
        target = deterministic_unmapped_state(row['target_smiles']).text
        prompt = smoke.policy_prompt(target, row['target_smiles'],
                                     include_inventory=True, actions=[], compact_history=True)
        if first['messages'][0]['content'] != SYSTEM or first['tools'] != TOOLS:
            raise ValueError(f'system/tool schema mismatch: {identifier}')
        if first['messages'][1]['content'] != prompt:
            raise ValueError(f'first-decision prompt mismatch: {identifier}')
    print(f'[meteor-31k] prompt contract exact for {len(chosen)}/{len(chosen)} first decisions', flush=True)


def aggregate(output: Path, chosen: list[str], world_size: int):
    rows = []
    for rank in range(world_size):
        path = output / f'rank{rank:02d}' / 'closed_loop_reactions.jsonl'
        rows.extend(read_jsonl(path))
    ids = [str(row['reaction_id']) for row in rows]
    if len(ids) != len(chosen) or set(ids) != set(chosen):
        raise ValueError(f'incomplete or duplicated outputs: {len(ids)}/{len(chosen)}')
    summary = {
        'artifact_type': 'product_only_closed_loop_k1',
        'denominator': len(chosen),
        'terminal': sum(bool(row['terminal']) for row in rows),
        'formally_executed_electron_event': sum(bool(row['transformed']) for row in rows),
        'terminal_with_event': sum(bool(row['terminal'] and row['transformed']) for row in rows),
        'full_exact': sum(bool(row['full_exact']) for row in rows),
        'structural_exact': sum(bool(row['structural_exact']) for row in rows),
        'strict_structural_exact': sum(bool(row['structural_exact'] and row['transformed']) for row in rows),
        'failure_counts': {},
        'raw_shards': [str(output / f'rank{rank:02d}' / 'closed_loop_reactions.jsonl')
                       for rank in range(world_size)],
    }
    for row in rows:
        reason = str(row['failure'])
        summary['failure_counts'][reason] = summary['failure_counts'].get(reason, 0) + 1
    (output / 'evaluation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(f'[meteor-31k] final={json.dumps(summary, ensure_ascii=False)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'worker', 'aggregate'], required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=384)
    parser.add_argument('--max-decisions', type=int, default=40)
    args = parser.parse_args()
    if args.world_size != 8 or not 0 <= args.rank < args.world_size:
        raise ValueError('requires exactly eight valid shard ranks')
    smoke = smoke_module()
    source_rows, history_rows, chosen = selection(smoke)
    ensure_manifest(args.output, fingerprint(args.max_new_tokens, args.max_decisions, args.world_size))
    if args.mode == 'preflight':
        preflight(smoke, source_rows, history_rows, chosen)
    elif args.mode == 'worker':
        mine = [identifier for identifier in chosen if shard(identifier, args.world_size) == args.rank]
        print(f'[meteor-31k] rank={args.rank} assigned={len(mine)}', flush=True)
        output = args.output / f'rank{args.rank:02d}'
        output.mkdir(parents=True, exist_ok=True)
        tokenizer, model = smoke.load_model(ADAPTER, 'Qwen/Qwen3-8B', args.cache_dir)
        smoke.run_closed_loop(tokenizer, model, source_rows, mine, output,
                              args.max_new_tokens, args.max_decisions)
        completed = list(read_jsonl(output / 'closed_loop_reactions.jsonl'))
        if {str(row['reaction_id']) for row in completed} != set(mine) or len(completed) != len(mine):
            raise ValueError(f'rank {args.rank} output incomplete')
        print(f'[meteor-31k] rank={args.rank} completed={len(completed)}', flush=True)
    else:
        aggregate(args.output, chosen, args.world_size)


if __name__ == '__main__':
    main()
