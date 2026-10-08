#!/usr/bin/env python3
"""Audit product-only FlowER State-SFT transfer to the 31k trace-view test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_reliable_strict_test_eval import audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--expected-adapter-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
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
    result.update({
        'artifact_type': 'mechet_flowER_state_sft_zero_shot_mech_uspto31k_trace_view_k1',
        'train_dataset': 'FlowER strict-executable 257167 train reactions',
        'test_dataset': 'mech-USPTO-31k current-compiler executable trace view',
        'full_endpoint_test_denominator_not_evaluated': 3120,
        'input_contract': 'product_only_no_reference_feedback',
        'policy': 'Qwen3-0.6B FlowER Stage-I State-SFT, no 31k finetuning',
        'candidate_policy': 'greedy_k1_single_executor_trajectory',
    })
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
