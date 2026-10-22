"""Read-only model/executor audit; write derived endpoint analysis artifacts.

No model sampling. Core precursor references are consistently defined using
MAIN-product atoms for both conditions, not all atoms of the original mixture.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import run_main_product_only as run
from rdkit import Chem
from transformers import AutoTokenizer
from scripts.run_natural_language_value_search import (
    Action, Node, execute, mapped_atom_numbers, policy_prompt, prediction_call,
    product_only_private_state, visible,
)
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.endpoints import split_precursor_endpoints, structural_exact


def read_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def main_mapped(smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    assert mol is not None
    fragments = [f for f in Chem.GetMolFrags(mol, asMols=True)
                 if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    fragments.sort(key=lambda f: f.GetNumHeavyAtoms(), reverse=True)
    assert fragments and (len(fragments) == 1 or fragments[0].GetNumHeavyAtoms() > fragments[1].GetNumHeavyAtoms())
    return Chem.MolToSmiles(fragments[0], canonical=True, isomericSmiles=True)


def main():
    root = run.OUTPUT
    summary = json.loads((root / 'evaluation.json').read_text())
    assert summary['complete'] and summary['completed_reactions'] == 32
    assert run.sha(root / 'run_main_product_only.py') == summary['provenance']['driver_sha256']
    for relative, meta in summary['raw_artifacts'].items():
        assert run.sha(root / relative) == meta['sha256']
    queries = read_rows(root / 'selected_inputs.jsonl')
    assert run.sha(root / 'selected_inputs.jsonl') == summary['provenance']['selected_inputs_sha256']
    assert all('.' not in q['main_product_smiles'] and len(q['dropped_components']) == 1 for q in queries)
    source = {str(r['source_id']): r for r in read_rows(summary['provenance']['source'])}
    assert run.sha(summary['provenance']['source']) == summary['provenance']['source_sha256']
    episodes = read_rows(root / 'records/episodes.jsonl')
    baseline = {r['source_id']: r for r in read_rows(run.PREVIOUS / 'records/episodes.jsonl')}
    generations = defaultdict(list)
    for generation in read_rows(root / 'records/raw_generations.jsonl'):
        generations[generation['reaction_id']].append(generation)
    assert [q['reaction_id'] for q in queries] == [e['source_id'] for e in episodes] == summary['provenance']['reaction_ids']
    assert len(episodes) == len(generations) == len({e['source_id'] for e in episodes}) == 32
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', revision=run.REVISION, local_files_only=True)
    counts = Counter()
    first_tools = Counter()
    reference_changes = []
    per_case = []
    for query, episode in zip(queries, episodes, strict=True):
        rid = episode['source_id']
        selected, dropped = run.main_product(source[rid]['target_smiles'])
        assert selected == query['main_product_smiles'] and dropped == query['dropped_components']
        initial = product_only_private_state(selected)
        target = visible(initial)
        node = Node(target=target, state=initial, next_map=max(mapped_atom_numbers(initial)) + 1,
                    visited={target})
        assert len(generations[rid]) == len(episode['attempts'])
        for generation, attempt in zip(generations[rid], episode['attempts'], strict=True):
            assert generation['greedy'] and generation['adapter_sha256'] == run.WEIGHT_SHA
            prompt = render_qwen_sft_tool_prefix(tok, [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': policy_prompt(target, node.state, include_inventory=True,
                                                        actions=node.actions, compact_history=False)},
            ], tools=TOOLS)
            assert tok(prompt, add_special_tokens=False)['input_ids'] == generation['input_token_ids'], (rid, 'prompt')
            assert generation['prompt'] == tok.decode(generation['input_token_ids'], skip_special_tokens=False)
            name, arguments, error = prediction_call(generation['generated_text'], tok)
            assert not error and name == attempt['name'] and arguments == attempt['arguments']
            assert attempt['state_before'] == visible(node.state)
            child, rejection = execute(node, Action(name, arguments, '', 0, 0), max_imports=32,
                                       reject_target_retained_finish=False)
            assert (child is not None) == attempt['accepted'] and rejection == attempt['error']
            if child is not None:
                result = child.actions[-1]['result']
                assert attempt['state_after'] == str(result.get('current_state') or result.get('derived_precursor') or '')
                node = child
            counts['verified_prompts_and_attempts'] += 1
        assert node.actions == episode['top_actions'] and node.terminal == episode['top_terminal']
        assert visible(node.state) == episode['top_precursor'] and len(node.actions) == episode['n_actions']
        full_reference = source[rid]['full_precursor_state']
        core_reference = split_precursor_endpoints(full_reference, main_mapped(source[rid]['target_smiles'])).structural
        if not structural_exact(core_reference, source[rid]['structural_precursor']):
            reference_changes.append(rid)
        full_hit = bool(node.terminal and visible(node.state) == visible(full_reference))
        core_prediction = split_precursor_endpoints(node.state, initial).structural
        core_hit = bool(node.terminal and structural_exact(core_prediction, core_reference))
        assert full_hit == episode['top1_full_exact']
        old = baseline[rid]
        old_core_prediction = ''
        if old['top_terminal']:
            assert old['top_actions'][-1]['name'] == 'finish_trace'
            old_root_main = main_mapped(product_only_private_state(source[rid]['target_smiles']))
            old_core_prediction = split_precursor_endpoints(old['top_actions'][-1]['state_before'], old_root_main).structural
        old_core_hit = bool(old['top_terminal'] and structural_exact(old_core_prediction, core_reference))
        first_tools[episode['attempts'][0]['name']] += 1
        counts.update({'reactions': 1, 'main_core_hits': core_hit, 'main_full_hits': full_hit,
                       'main_terminal': node.terminal, 'baseline_core_hits': old_core_hit,
                       'baseline_full_hits': old['top1_full_exact'], 'baseline_terminal': old['top_terminal']})
        per_case.append({'reaction_id': rid, 'main_product': selected, 'dropped_components': dropped,
                         'expected_main_derived_precursors': visible(core_reference),
                         'predicted_main_derived_precursors': visible(core_prediction),
                         'core_hit': core_hit, 'full_hit': full_hit, 'terminal': node.terminal,
                         'baseline_core_hit': old_core_hit, 'baseline_full_hit': old['top1_full_exact'],
                         'errors': episode['rejected'], 'accepted_actions': len(node.actions)})
    max_output = max(len(g['generated_token_ids']) for gs in generations.values() for g in gs)
    max_prompt = max(len(g['input_token_ids']) for gs in generations.values() for g in gs)
    analysis = {
        'audit': 'PASS', 'complete': True, 'denominator': 32, 'counts': counts,
        'input_condition': 'largest_carbon_containing_component_only; no injected side components',
        'single_component_roots_verified': 32, 'same_checkpoint_verified': run.WEIGHT_SHA,
        'first_model_tool_counts': first_tools,
        'max_generated_tokens': max_output, 'max_prompt_tokens': max_prompt,
        'generation_token_cap_reached': max_output >= 512,
        'core_reference_scope': 'only precursor fragments containing atoms of the main product, applied to both conditions',
        'original_two_component_core_reference_changes': reference_changes,
        'legacy_row_structural_scores': 'Original evaluation.json uses the old all-product-side structural reference; cite this unified rescore for main-derived structure. Full-mixture scores are unchanged.',
        'driver_sha256_verified': True, 'raw_artifact_sha256_verified': True,
        'actual_prompt_token_ids_match_model_executed_state': True,
        'all_accepted_and_rejected_actions_replayed': True,
        'no_gt_prefix_or_injected_imports': True,
    }
    with (root / 'per_reaction_endpoint_comparison.jsonl').open('w') as writer:
        for row in per_case:
            writer.write(json.dumps(row, ensure_ascii=False) + '\n')
    (root / 'endpoint_comparison.json').write_text(json.dumps(analysis, indent=2) + '\n')
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
