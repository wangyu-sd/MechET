"""User-authorized32-reaction greedy product-only diagnostic; no training.

Reuses the existing tested MechET policy/runtime/executor unchanged. The gold
source is consulted by rollout() only after search_unlabeled() has returned.
The generate hook records actual tensors without changing generation arguments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from argparse import Namespace
from collections import Counter

CODE = Path('/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy')
sys.path[:0] = [str(CODE), str(CODE / 'src')]

from scripts.run_natural_language_value_search import (
    Action, Node, Runtime, execute, policy_prompt, product_only_private_state,
    rollout, validate_matched_v2_args, validate_v2_adapter_manifest, visible,
)
from mechet.in_place_grounded_flow import mapped_atom_numbers
from mechet.assistant_masking import render_qwen_sft_tool_prefix
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS

SHARED = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
OUTPUT = Path(__file__).resolve().parent
TF = SHARED / 'outputs/eval/mech31k_state8ep_teacher_forced_pr93_20261022_01'
ADAPTER = SHARED / 'outputs/agent/reliable_mech_uspto31k_state_qwen3_0_6b_v100_seed17/checkpoint-4056'
WEIGHT_SHA = 'bf5cacc93d2c5e64b11df585e0dd0febf790773e21d13ec0838fa8dd538cdf51'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    import transformers
    import peft
    import rdkit

    tasks_file = TF / 'prepared/tasks.jsonl'
    selection = json.loads((TF / 'prepared/selection.json').read_text())
    assert sha(tasks_file) == selection['tasks_sha256']
    assert sha(ADAPTER / 'adapter_model.safetensors') == WEIGHT_SHA
    state = json.loads((ADAPTER / 'trainer_state.json').read_text())
    assert state['epoch'] == 8 and state['global_step'] == 4056
    contract = json.loads((ADAPTER.parent / 'data_contract.json').read_text())
    assert contract['environment_revision'] == 'natural_language_electron_event_v2'
    assert contract['training_runtime']['fp16'] is True
    validate_v2_adapter_manifest(ADAPTER.parent, compact_history=False, vnext=True,
                                 expected_model='Qwen/Qwen3-0.6B', expected_revision=REVISION)
    source = Path(selection['source'])
    assert sha(source) == selection['source_sha256']
    tasks = [json.loads(l) for l in tasks_file.open() if l.strip()]
    groups = {}
    for task in tasks:
        groups.setdefault(task['reaction_id'], []).append(task)
    sources = {str(r['source_id']): r for r in map(json.loads, source.open())}
    rows = [sources[rid] for rid in groups]
    assert len(rows) == 32 and len(tasks) == 157
    record_dir = OUTPUT / 'records'
    record_dir.mkdir(exist_ok=False)
    with (OUTPUT / 'selected_source_rows.jsonl').open('w') as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + '\n')

    # Software preflight only: prove that independently rebuilt public-product
    # roots and imports can reproduce saved SFT prefixes; never called online.
    matched = 0
    for rid, gold_rows in groups.items():
        target = gold_rows[0]['messages'][1]['content'].split('TARGET PRODUCT SMILES: ', 1)[1].split('\n', 1)[0]
        initial = product_only_private_state(target)
        node = Node(target=target, state=initial, next_map=max(mapped_atom_numbers(initial)) + 1,
                    visited={visible(initial)})
        for task in gold_rows:
            assert policy_prompt(target, node.state, include_inventory=True, actions=node.actions,
                                 compact_history=False) == task['messages'][1]['content']
            child, error = execute(node, Action(task['gold_name'], task['gold_arguments'], '', 0, 0),
                                   max_imports=32, reject_target_retained_finish=False)
            assert child is not None, (rid, task['key'], error)
            node = child
            matched += 1
    args = Namespace(
        model='Qwen/Qwen3-0.6B', model_revision=REVISION, policy_adapter=str(ADAPTER),
        value_adapter='', value_kind='successor_pn', value_weight=0.,
        pointer_head='', pointer_weight=0., no_4bit=True, dtype='float16',
        matched_v2=True, legacy_dual_prompt=False, vnext_v2_prefix=False,
        compact_history=False, branching=1, early_beam=1, late_beam=1, early_depth=2,
        max_decisions=40, max_imports=32, max_new_tokens=512, max_context=4096,
        product_only_remap=True, reject_target_retained_finish=False,
        planning_sample=False, raw_model_nll=False,
    )
    validate_matched_v2_args(args)
    provenance = {
        'schema': 'mechet_state8ep_product_only_greedy_smoke_v1',
        'model': args.model, 'revision': REVISION, 'adapter_path': str(ADAPTER),
        'adapter_sha256': WEIGHT_SHA, 'epoch': 8, 'checkpoint_step': 4056,
        'reaction_ids': list(groups), 'reaction_ids_sha256': selection['reaction_ids_sha256'],
        'denominator': 32, 'source': str(source), 'source_sha256': selection['source_sha256'],
        'teacher_forced_task_manifest': str(TF / 'prepared/selection.json'),
        'gold_replay_prefix_audit': {'products': 32, 'decisions_matched': matched, 'errors': 0},
        'model_receives_gt_prefix': False, 'gold_used_by_search': False,
        'gt_scheduled_imports': False, 'product_only_remap': True,
        'state_only_prompt': True, 'greedy_k1': True, 'value_model': False,
        'bounds': {k: getattr(args, k) for k in ('max_decisions', 'max_imports', 'max_new_tokens', 'max_context')},
        'dtype': args.dtype, 'quantization': None, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'transformers': transformers.__version__,
        'peft': peft.__version__, 'rdkit': rdkit.__version__, 'gpu': torch.cuda.get_device_name(0),
        'runtime_reference': '6874ea7b60f803816447f8f1420e9c687eaa851c',
        'code_sha256': {str(p.relative_to(CODE)): sha(p) for p in (
            CODE / 'scripts/run_natural_language_value_search.py',
            CODE / 'scripts/build_natural_language_event_sft.py',
            CODE / 'src/mechet/forward_expert.py', CODE / 'src/mechet/endpoints.py',
        )}, 'driver_sha256': sha(__file__),
    }
    (OUTPUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'stage': 'verified_preflight', 'products': 32, 'gold_calls': matched,
                      'adapter_sha256': WEIGHT_SHA, 'gt_used_online': False}), flush=True)
    torch.manual_seed(17)
    runtime = Runtime(args, 0)
    current_case = {'id': ''}
    generated_tokens = 0
    raw_writer = (record_dir / 'raw_generations.jsonl').open('w')
    original_generate = runtime.model.generate

    def record_generate(*positional, **kwargs):
        nonlocal generated_tokens
        output = original_generate(*positional, **kwargs)
        prompt_width = kwargs['input_ids'].shape[1]
        for seq in output.sequences:
            completion = seq[prompt_width:].tolist()
            generated_tokens += len(completion)
            record = {
                'reaction_id': current_case['id'], 'adapter_sha256': WEIGHT_SHA,
                'input_token_ids': kwargs['input_ids'][0].tolist(),
                'attention_mask': kwargs['attention_mask'][0].tolist(),
                'prompt': runtime.tokenizer.decode(kwargs['input_ids'][0], skip_special_tokens=False),
                'generated_token_ids': completion,
                'generated_text': runtime.tokenizer.decode(completion, skip_special_tokens=False),
                'greedy': not kwargs.get('do_sample', False),
            }
            raw_writer.write(json.dumps(record, ensure_ascii=False) + '\n')
            raw_writer.flush()
        return output

    runtime.model.generate = record_generate
    results = []
    started = time.monotonic()
    with (record_dir / 'episodes.jsonl').open('w') as writer:
        for i, row in enumerate(rows, 1):
            current_case['id'] = str(row['source_id'])
            # All GT endpoint access is inside rollout AFTER search_unlabeled.
            result = rollout(runtime, row, args)
            result['elapsed_seconds_total'] = time.monotonic() - started
            results.append(result)
            writer.write(json.dumps(result, ensure_ascii=False) + '\n')
            writer.flush()
            print(json.dumps({'stage': 'closed_loop', 'done': i, 'total': 32,
                              'percent': round(i * 100 / 32, 2), 'id': row['source_id'],
                              'terminal': result['top_terminal'],
                              'structural_hit': result['top1_structural_exact'],
                              'full_hit': result['top1_full_exact'],
                              'actions': result['n_actions'], 'errors': result['rejected']}), flush=True)
    raw_writer.close()
    errors = Counter()
    for result in results:
        errors.update(result['rejected'])
    summary = {
        'complete': len(results) == 32, 'denominator': 32, 'completed_reactions': len(results),
        'greedy_k1': True, 'gold_state_conditioned': False, 'model_receives_gt_prefix': False,
        'structural_endpoint_hits': sum(r['top1_structural_exact'] for r in results),
        'full_mixture_endpoint_hits': sum(r['top1_full_exact'] for r in results),
        'terminal_reactions': sum(r['top_terminal'] for r in results),
        'accepted_actions': sum(r['n_actions'] for r in results),
        'generated_tokens': generated_tokens, 'errors': dict(errors),
        'elapsed_seconds': time.monotonic() - started,
        'claim_boundary': '32-reaction validation smoke; not full3120 test or multistep synthesis planning',
        'provenance': provenance,
    }
    for stem in ('structural_endpoint', 'full_mixture_endpoint'):
        summary[stem + '_rate'] = summary[stem + '_hits'] / 32
    summary['terminal_rate'] = summary['terminal_reactions'] / 32
    summary['raw_artifacts'] = {str(f.relative_to(OUTPUT)): {'sha256': sha(f), 'bytes': f.stat().st_size}
                                for f in record_dir.glob('*.jsonl')}
    (OUTPUT / 'evaluation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
