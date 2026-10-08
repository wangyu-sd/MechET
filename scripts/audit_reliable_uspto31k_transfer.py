#!/usr/bin/env python3
"""Audit product-only FlowER State-SFT transfer to the 31k trace-view test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_reliable_strict_test_eval import audit


def audit_beam_tree_row(row: dict) -> None:
    """Check that every executed edge and terminal survives serialization."""
    if row.get('search_config', {}).get('mode') != 'deterministic_beam':
        raise ValueError('beam result lacks deterministic_beam search config')
    nodes = row.get('search_tree')
    attempts = row.get('attempts')
    if not isinstance(nodes, list) or not nodes or not isinstance(attempts, list):
        raise ValueError('beam result lacks its search tree or attempted edges')
    if [node['node_id'] for node in nodes] != list(range(len(nodes))):
        raise ValueError('search tree node IDs are not contiguous')
    if nodes[0]['parent_node_id'] is not None:
        raise ValueError('search tree root has a parent')
    if [attempt['attempt_id'] for attempt in attempts] != list(range(len(attempts))):
        raise ValueError('search tree attempt IDs are not contiguous')
    for node in nodes[1:]:
        parent = node['parent_node_id']
        edge_id = node['via_attempt_id']
        if not (0 <= parent < node['node_id'] and 0 <= edge_id < len(attempts)):
            raise ValueError('search tree node has an invalid parent or edge')
        edge = attempts[edge_id]
        if (edge['parent_node_id'] != parent or
                edge['child_node_id'] != node['node_id'] or
                not edge['accepted']):
            raise ValueError('search tree parent/child edge disagrees with attempt')
    for edge in attempts:
        child = edge['child_node_id']
        if child is None and edge['accepted']:
            raise ValueError('accepted attempt has no child node')
        if child is not None and (not edge['accepted'] or not 0 < child < len(nodes)):
            raise ValueError('rejected or invalid attempt has a child node')
    terminal_ids = {node['node_id'] for node in nodes if node['terminal']}
    if terminal_ids != set(row['terminal_node_ids']) or len(terminal_ids) != row['n_terminals']:
        raise ValueError('terminal branches were dropped from the search tree')
    top_id = row['top_node_id']
    if not 0 <= top_id < len(nodes) or bool(nodes[top_id]['terminal']) != bool(row['top_terminal']):
        raise ValueError('top candidate is not represented correctly in the search tree')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--expected-adapter-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--beam-search', action='store_true')
    args = parser.parse_args()

    result = audit(
        source=args.source, manifest=args.manifest,
        results_dir=args.results_dir, expected_rows=1253,
        adapter=args.adapter,
        expected_adapter_sha256=args.expected_adapter_sha256,
        stage='state',
    )
    if result['observed_predictions'] != 1253 or result['missing_predictions'] != 0:
        raise ValueError('the 1,253-reaction test evaluation is incomplete')
    if args.beam_search:
        beam_counts = {'pass_at_beam': 0, 'tree_nodes': 0, 'attempted_edges': 0,
                       'generated_candidates': 0}
        for shard in sorted(args.results_dir.glob('results.shard-*.jsonl')):
            with shard.open(encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    audit_beam_tree_row(row)
                    beam_counts['pass_at_beam'] += int(bool(row['pass_at_beam']))
                    beam_counts['tree_nodes'] += len(row['search_tree'])
                    beam_counts['attempted_edges'] += len(row['attempts'])
                    beam_counts['generated_candidates'] += len(row.get('generations', []))
        result.update(beam_counts)
        result['pass_at_beam_accuracy'] = beam_counts['pass_at_beam'] / result['test_denominator']
    result.update({
        'artifact_type': (
            'mechet_flowER_state_sft_zero_shot_mech_uspto31k_trace_view_beam'
            if args.beam_search else
            'mechet_flowER_state_sft_zero_shot_mech_uspto31k_trace_view_k1'
        ),
        'train_dataset': 'FlowER strict-executable 257167 train reactions',
        'test_dataset': 'mech-USPTO-31k current-compiler executable trace view',
        'full_endpoint_test_denominator_not_evaluated': 3120,
        'input_contract': 'product_only_no_reference_feedback',
        'policy': 'Qwen3-0.6B FlowER Stage-I State-SFT, no 31k finetuning',
        'candidate_policy': (
            'deterministic_token_beam_plus_executor_state_beam'
            if args.beam_search else 'greedy_k1_single_executor_trajectory'
        ),
    })
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
