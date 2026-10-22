"""Paired user-requested main-product-only ablation of the 32-case smoke.

No production runtime changes. Dropped product-side components never enter
search, model observations, imports or hidden root state.
"""
from __future__ import annotations

from argparse import Namespace
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

CODE = Path('/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy')
SHARED = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
PREVIOUS = SHARED / 'outputs/eval/mech31k_state8ep_closed_loop_pr93_20261022_01'
OUTPUT = Path(__file__).resolve().parent
ADAPTER = SHARED / 'outputs/agent/reliable_mech_uspto31k_state_qwen3_0_6b_v100_seed17/checkpoint-4056'
WEIGHT_SHA = 'bf5cacc93d2c5e64b11df585e0dd0febf790773e21d13ec0838fa8dd538cdf51'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
sys.path[:0] = [str(CODE), str(CODE / 'src')]

from rdkit import Chem
from scripts.run_natural_language_value_search import (
    Runtime, product_only_private_state, rollout, validate_matched_v2_args,
    validate_v2_adapter_manifest, visible,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main_product(smiles):
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    mol = Chem.MolFromSmiles(smiles, parser)
    assert mol is not None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    fragments = list(Chem.GetMolFrags(mol, asMols=True))
    organic = [f for f in fragments if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    assert organic, 'No carbon-containing product component'
    ordered = sorted(organic, key=lambda f: (-f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)))
    assert len(ordered) == 1 or ordered[0].GetNumHeavyAtoms() > ordered[1].GetNumHeavyAtoms(), 'Ambiguous largest carbon-containing product'
    selected = Chem.MolToSmiles(ordered[0], canonical=True, isomericSmiles=True)
    assert '.' not in selected and ':' not in selected
    others = sorted(Chem.MolToSmiles(f, canonical=True, isomericSmiles=True)
                    for f in fragments if f is not ordered[0])
    assert len(others) == len(fragments) - 1
    assert visible(product_only_private_state(selected)) == visible(selected)
    return selected, others


def main():
    import torch
    import transformers
    import peft
    import rdkit

    old = json.loads((PREVIOUS / 'evaluation.json').read_text())
    assert old['complete'] and old['completed_reactions'] == 32
    assert sha(ADAPTER / 'adapter_model.safetensors') == WEIGHT_SHA == old['provenance']['adapter_sha256']
    trainer_state = json.loads((ADAPTER / 'trainer_state.json').read_text())
    assert trainer_state['epoch'] == 8 and trainer_state['global_step'] == 4056
    source = Path(old['provenance']['source'])
    assert sha(source) == old['provenance']['source_sha256']
    validate_v2_adapter_manifest(ADAPTER.parent, compact_history=False, vnext=True,
                                 expected_model='Qwen/Qwen3-0.6B', expected_revision=REVISION)
    source_rows = {str(r['source_id']): r for r in map(json.loads, source.open())}
    reaction_ids = old['provenance']['reaction_ids']
    assert len(set(reaction_ids)) == len(reaction_ids) == 32
    queries = []
    inference_rows = []
    for rid in reaction_ids:
        original = source_rows[rid]
        selected, dropped = main_product(original['target_smiles'])
        assert len(dropped) == 1
        queries.append({'reaction_id': rid, 'main_product_smiles': selected,
                        'dropped_components': dropped, 'original_product_side': visible(original['target_smiles'])})
        # No original trace plan or other future-state metadata is passed even
        # to rollout(). References below are accessed only after its search.
        inference_rows.append({k: original[k] for k in (
            'id', 'source_id', 'expected_precursor', 'full_precursor_state',
            'structural_precursor')})
        inference_rows[-1]['target_smiles'] = selected
    records = OUTPUT / 'records'
    records.mkdir(exist_ok=False)
    with (OUTPUT / 'selected_inputs.jsonl').open('w') as writer:
        for query in queries:
            writer.write(json.dumps(query, ensure_ascii=False) + '\n')
    args = Namespace(
        model='Qwen/Qwen3-0.6B', model_revision=REVISION, policy_adapter=str(ADAPTER),
        value_adapter='', value_kind='successor_pn', value_weight=0., pointer_head='', pointer_weight=0.,
        no_4bit=True, dtype='float16', matched_v2=True, legacy_dual_prompt=False,
        vnext_v2_prefix=False, compact_history=False, branching=1, early_beam=1,
        late_beam=1, early_depth=2, max_decisions=40, max_imports=32,
        max_new_tokens=512, max_context=4096, product_only_remap=True,
        reject_target_retained_finish=False, planning_sample=False, raw_model_nll=False,
    )
    validate_matched_v2_args(args)
    provenance = dict(old['provenance'])
    provenance.pop('gold_replay_prefix_audit', None)
    provenance.update({
        'schema': 'mechet_state8ep_main_product_only_smoke_v1',
        'input_condition': 'single_largest_carbon_containing_component_of_trace_view_product_side',
        'single_major_product_only': True, 'root_contains_dropped_components': False,
        'initial_gt_imports': False, 'gold_state_conditioned': False,
        'paired_evaluation': str(PREVIOUS / 'evaluation.json'),
        'paired_evaluation_sha256': sha(PREVIOUS / 'evaluation.json'),
        'selected_inputs_sha256': sha(OUTPUT / 'selected_inputs.jsonl'),
        'driver_sha256': sha(__file__),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'transformers': transformers.__version__, 'peft': peft.__version__,
        'rdkit': rdkit.__version__, 'gpu': torch.cuda.get_device_name(0),
        'input_alignment_note': 'Same tool grammar and state-only renderer; unlike training roots, omitted side components must be inferred/imported by the policy. No GT action-prefix parity claim for modified roots.',
    })
    for relative, expected in provenance['code_sha256'].items():
        assert sha(CODE / relative) == expected
    (OUTPUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'stage': 'preflight', 'reactions': 32, 'single_component_roots': 32,
                      'side_components_removed': 32, 'gt_imports_injected': False,
                      'checkpoint': '4056', 'epoch': 8}), flush=True)
    torch.manual_seed(17)
    runtime = Runtime(args, 0)
    current_case = {'id': ''}
    generated_tokens = 0
    raw_writer = (records / 'raw_generations.jsonl').open('w')
    original_generate = runtime.model.generate

    def record_generate(*positional, **kwargs):
        nonlocal generated_tokens
        output = original_generate(*positional, **kwargs)
        width = kwargs['input_ids'].shape[1]
        for seq in output.sequences:
            completion = seq[width:].tolist()
            generated_tokens += len(completion)
            raw_writer.write(json.dumps({
                'reaction_id': current_case['id'], 'adapter_sha256': WEIGHT_SHA,
                'input_token_ids': kwargs['input_ids'][0].tolist(),
                'attention_mask': kwargs['attention_mask'][0].tolist(),
                'prompt': runtime.tokenizer.decode(kwargs['input_ids'][0], skip_special_tokens=False),
                'generated_token_ids': completion,
                'generated_text': runtime.tokenizer.decode(completion, skip_special_tokens=False),
                'greedy': not kwargs.get('do_sample', False),
            }, ensure_ascii=False) + '\n')
            raw_writer.flush()
        return output

    runtime.model.generate = record_generate
    results = []
    started = time.monotonic()
    with (records / 'episodes.jsonl').open('w') as writer:
        for i, row in enumerate(inference_rows, 1):
            current_case['id'] = str(row['source_id'])
            result = rollout(runtime, row, args)
            result['elapsed_seconds_total'] = time.monotonic() - started
            results.append(result)
            writer.write(json.dumps(result, ensure_ascii=False) + '\n')
            writer.flush()
            print(json.dumps({'stage': 'main_product_only', 'done': i, 'total': 32,
                'percent': round(i * 100 / 32, 2), 'id': row['source_id'],
                'terminal': result['top_terminal'], 'structural_hit': result['top1_structural_exact'],
                'full_hit': result['top1_full_exact'], 'actions': result['n_actions'],
                'errors': result['rejected']}), flush=True)
    raw_writer.close()
    errors = Counter()
    for result in results:
        errors.update(result['rejected'])
    summary = {
        'complete': len(results) == 32, 'denominator': 32, 'completed_reactions': len(results),
        'single_major_product_only': True, 'greedy_k1': True,
        'gold_state_conditioned': False, 'model_receives_gt_prefix': False,
        'structural_endpoint_hits': sum(r['top1_structural_exact'] for r in results),
        'full_mixture_endpoint_hits': sum(r['top1_full_exact'] for r in results),
        'terminal_reactions': sum(r['top_terminal'] for r in results),
        'accepted_actions': sum(r['n_actions'] for r in results),
        'generated_tokens': generated_tokens, 'errors': dict(errors),
        'elapsed_seconds': time.monotonic() - started, 'provenance': provenance,
        'claim_boundary': 'Paired 32-validation-reaction main-product-only diagnostic; not full3120 test.',
        'baseline_with_side_components': {'structural_hits': old['structural_endpoint_hits'],
            'full_hits': old['full_mixture_endpoint_hits'], 'terminal': old['terminal_reactions']},
    }
    for stem in ('structural_endpoint', 'full_mixture_endpoint'):
        summary[stem + '_rate'] = summary[stem + '_hits'] / 32
    summary['terminal_rate'] = summary['terminal_reactions'] / 32
    summary['raw_artifacts'] = {str(f.relative_to(OUTPUT)): {'sha256': sha(f), 'bytes': f.stat().st_size}
                                for f in records.glob('*.jsonl')}
    (OUTPUT / 'evaluation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
