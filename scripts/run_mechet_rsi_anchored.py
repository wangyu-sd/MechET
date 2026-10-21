#!/usr/bin/env python3
"""PR92 actual anchored controls + frozen, common-seed chemical evaluation."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from mechet.rsi_contract import R0_SHA, fingerprint, sha256, write_json, stable_seed
from mechet.rsi_preference_v1 import strong_pairs, paired_candidates, chemical_key, per_target_budget
from scripts.audit_rsi_v1 import audit, reference_classes, label_episodes
from scripts.run_mechet_rsi_smoke import Experiment, freeze_adapter, stage_file, summarize_evaluation, require_fresh_round


def productive_summary(episodes, target_ids):
    from mechet.endpoints import split_precursor_endpoints
    result = summarize_evaluation(episodes, target_ids)
    productive = [e for e in episodes if e.get('outcome', {}).get('success') and
                  e.get('productivity', {}).get('task_compatible_net_progress') is True]
    result.update(productive_roundtrip_at_1_targets=len({e['id'] for e in productive if e.get('candidate', 0) == 0}),
        productive_roundtrip_at_k_targets=len({e['id'] for e in productive}),
        distinct_productive_disconnections=len({(e['id'], chemical_key(split_precursor_endpoints(
            e['final_state'],e['initial_state']).structural)) for e in productive}),
        task_compatibility_unknown_episodes=sum(e.get('productivity', {}).get('task_compatible_net_progress') is None for e in episodes),
        reference_structural_targets=len({e['id'] for e in episodes if e.get('terminal') and e.get('reference_structural_exact')}))
    for metric in ('productive_roundtrip_at_1', 'productive_roundtrip_at_k'):
        result[metric+'_rate'] = result[metric+'_targets']/max(len(set(target_ids)), 1)
    result['label_scope'] = 'private_task_class_compatibility_and_frozen_model_confirmation_not_experimental_truth'
    return result


def matched_views(models, identifiers):
    costs = {name: {identifier: sum(len(s['completion_ids']) for e in es if e['id'] == identifier for s in e['steps'])
                    for identifier in identifiers} for name, es in models.items()}
    budgets = {identifier: min(v[identifier] for v in costs.values()) for identifier in identifiers}
    result = {'per_target_reverse_budget': budgets, 'models': {},
              'boundary': 'whole_episode_censoring; full frozen target denominator; no global prefix break'}
    for name, es in models.items():
        view = per_target_budget(es, identifiers, budgets)
        result['models'][name] = {**{k:v for k,v in view.items() if k != 'episodes'},
                                 **productive_summary(view['episodes'], identifiers)}
    return result


class AnchoredExperiment(Experiment):
    def prompt_gate(self):
        from mechet.assistant_masking import render_qwen_sft_tool_prefix
        from mechet.in_place_grounded_flow import mapped_atom_numbers
        from scripts.run_natural_language_value_search import Node
        from scripts.rsi_runtime import reverse_prompt
        sources=json.loads((self.output/'replay/selected_source_rows.json').read_text())
        # Held-out prompts are audited, never retained or used as training loss.
        import re,hashlib
        wanted={r['id'] for r in self.valid_rows};digest=hashlib.sha256()
        with Path(self.cfg['valid_history_source']).open('rb') as f:
            for line in f:
                digest.update(line)
                match=re.search(rb'"source_id"\s*:\s*"([^"]+)"',line[:500])
                if not match or match[1].decode() not in wanted:continue
                row=json.loads(line)
                if row['metadata']['decision_index']==0:sources[row['source_id']]=[row]
        if digest.hexdigest()!=self.cfg['valid_history_sha']:raise ValueError('held-out history prompt source SHA changed')
        records=[]
        for row in self.profile_rows+self.valid_rows:
            state=row['initial_state'];node=Node(target=row['target'],state=state,next_map=max(mapped_atom_numbers(state))+1)
            expected=sources[row['id']][0]
            text=reverse_prompt(node,self.tokenizer)
            if text!=render_qwen_sft_tool_prefix(self.tokenizer,expected['messages'][:2],tools=expected['tools']):
                raise ValueError('full cohort root Stage-II prompt mismatch: '+row['id'])
            records.append({'source_id':row['id'],'prompt_sha':fingerprint(text),'exact_prefix':True})
        write_json(self.output/'prompt_parity.json',{'records':records,'targets':len(records),
            'held_out_observations_only_never_supervised':True,'authentic_product_representation':True,
            'private_maps_not_model_visible':True})

    def collect(self, rows, actor, phase, indices, direction='reverse', temperature=1., guard=False):
        from scripts.rsi_runtime import collect_episodes
        indices = list(indices)
        contract = {'rows': rows, 'actor_sha': self.actor_sha[actor], 'phase': phase, 'indices': indices,
            'direction': direction, 'temperature': temperature, 'guard': guard, 'seed_contract': 'v1',
            'attempts': 24, 'tokens': 384, 'runtime_sha': sha256(ROOT/'scripts/rsi_runtime.py')}
        def produce():
            start = time.monotonic()
            episodes = collect_episodes(rows, owner=self.owner, tokenizer=self.tokenizer, lora=self.loras[actor],
                actor_sha=self.actor_sha[actor], round_id=phase, sample_indices=indices, direction=direction,
                journal_path=self.output/(phase+'.jsonl'), max_attempts=24, max_tokens=384,
                seed_contract='v1', temperature=temperature, rejection_guard=guard)
            return {'episodes': episodes, 'seconds': time.monotonic()-start,
                    'generated_tokens_including_recovered': sum(len(s['completion_ids']) for e in episodes for s in e['steps'])}
        value = stage_file(self.output/'stages'/(phase+'.json'), contract, produce)
        self.log('collection-complete', actor=actor, collection=phase, seconds=value['seconds'],
                 episodes=len(value['episodes']), tokens=value['generated_tokens_including_recovered'])
        return copy.deepcopy(value['episodes'])

    def feedback_v1(self, episodes, phase, references, forward_limit=None):
        from mechet.roundtrip_bridge import bridge_reverse_episode
        from mechet.roundtrip_reward import forward_outcome, component_match
        # Old forward caches have a different decoding-seed/protocol identity;
        # never silently reuse them as matched v1 independent confirmation.
        for e in episodes:
            if e.get('forward_judge_contract') != 'v1':
                e['outcome'] = {'status':'forward_unchecked','success':False}
        label_episodes(episodes, references)
        groups = defaultdict(list); unique = {}; bridge_audit = []
        for e in episodes:
            if not e['terminal'] or not e['flow_count']:
                e['outcome'] = {'status': 'inverse_incomplete' if not e['terminal'] else 'nonproductive', 'success': False}
                continue
            bridge = bridge_reverse_episode(e)
            key = fingerprint([self.actor_sha['F0'], bridge['initial_state'], e['target'],
                'autonomous_F0_v1_seed17_K1_temperature1_24x384', sha256(ROOT/'src/mechet/roundtrip_bridge.py')])
            e['bridge_key'] = key; groups[key].append(e)
            bridge_audit.append({'episode_id': e['episode_id'], 'bridge': bridge, 'key': key})
            if bridge['initial_state'] and not component_match(bridge['initial_state'], e['target']):
                unique.setdefault(key, {'episode': e, 'bridge': bridge})
                e['forward_check_eligible'] = True
            else:e['forward_check_eligible'] = False
        if forward_limit is None:
            selected = [v['episode'] for _, v in sorted(unique.items())]
        else:
            # Include nonterminal executor-invalid comparators, but only
            # terminal bridged actions can consume a forward-generation slot.
            selected = paired_candidates(episodes, forward_limit, known_keys=self.roundtrips,
                resolved_sources={p['source_id'] for p in strong_pairs(episodes)})
            selected = [e for e in selected if e.get('bridge_key') in unique]
        selected_keys = list(dict.fromkeys(e['bridge_key'] for e in selected))
        check_rows = [{'id': key, 'initial_state': unique[key]['bridge']['initial_state']}
                      for key in selected_keys if key not in self.roundtrips]
        if check_rows:
            forward = self.collect(check_rows, 'F0', phase+'_forward', [0], direction='forward')
            for f in forward:
                key = f['id']; target = unique[key]['episode']['target']
                self.roundtrips[key] = {'outcome': forward_outcome(f, target), 'episode_id': f['episode_id'],
                    'f0_sha': self.actor_sha['F0'], 'protocol': 'v1', 'generated_tokens': sum(len(s['completion_ids']) for s in f['steps'])}
        for key, es in groups.items():
            for e in es:
                e['outcome'] = copy.deepcopy(self.roundtrips[key]['outcome']) if key in self.roundtrips else {
                    'status': 'nonproductive' if component_match(e['final_state'], e['target']) else 'forward_unchecked', 'success': False}
                e['forward_judge_contract'] = 'v1'
        label_episodes(episodes, references)
        write_json(self.output/(phase+'.bridge_audit.json'), bridge_audit)
        write_json(self.output/'f0_v1_cache.json', self.roundtrips)
        record = {'episodes': episodes, 'unique_candidate_keys': len(unique), 'fresh_forward_checks': len(check_rows),
            'selected_forward_checks': len(selected_keys), 'cache_reused': len(selected_keys)-len(check_rows),
            'fresh_forward_tokens_including_recovered': sum(self.roundtrips[r['id']]['generated_tokens'] for r in check_rows),
            'logical_selected_forward_tokens_including_cache': sum(self.roundtrips[k]['generated_tokens'] for k in selected_keys if k in self.roundtrips),
            'allocation': 'within_target_complementary_pairs' if forward_limit is not None else 'all_validation_candidates',
            'old_v0_cache_not_reused_reason': 'different_decoder_seed_contract'}
        write_json(self.output/(phase+'.feedback.json'), record)
        self.log('v1-feedback', collection=phase, **{k:v for k,v in record.items() if k != 'episodes'})
        return episodes, len(check_rows)

    def pool_v1(self, rows, actor, phase, references):
        episodes = []; selected_rows = rows; used = 0
        for lower, upper, cap in ((0, 2, 32), (2, 4, 16), (4, 8, 16)):
            if not selected_rows:break
            batch = self.collect(selected_rows, actor, phase+f'_K{upper}', range(lower, upper))
            for e in batch:e['collection_phase'] = e['round']; e['round'] = phase
            episodes += batch
            episodes, count = self.feedback_v1(episodes, phase+f'_K{upper}', references, min(cap, 64-used))
            used += count
            pairs = strong_pairs(episodes); resolved = {p['source_id'] for p in pairs}
            # Expand only plausible unresolved alternatives, not all failures.
            plausible = {e['id'] for e in paired_candidates(episodes, len(rows)*2)}
            selected_rows = [r for r in rows if r['id'] in plausible-resolved]
        pairs = strong_pairs(episodes)
        write_json(self.output/(phase+'.allocation.json'), {'forward_checks_used': used, 'limit': 64,
            'strong_pairs': len(pairs), 'independent_pair_sources': len({p['source_id'] for p in pairs}),
            'tool_pairs': dict(Counter(p['tool'] for p in pairs)), 'source_cap': 2})
        write_json(self.output/(phase+'.preferences.json'), pairs)
        return episodes, pairs

    def evaluate_v1(self, rows, actor, references, *, guard=False):
        from mechet.endpoints import split_precursor_endpoints, structural_exact
        result = {}
        for mode, indices, temperature in (('greedyK1', [0], 0.), ('stochasticK2', [0, 1], 1.)):
            phase = 'valid_'+actor+'_'+mode+('_guard' if guard else '')
            es = self.collect(rows, actor, phase, indices, temperature=temperature, guard=guard)
            es, _ = self.feedback_v1(es, phase, references)
            lookup = {r['id']:r for r in rows}
            for e in es:
                e['reference_structural_exact'] = structural_exact(
                    split_precursor_endpoints(e['final_state'], e['initial_state']).structural,
                    lookup[e['id']]['reference_structural'])
            summary = productive_summary(es, lookup)
            result[mode] = es
            write_json(self.output/'evaluation'/(actor+'_'+mode+('_guard' if guard else '')+'.json'),
                {'summary': summary, 'episodes': es, 'guard': guard, 'temperature': temperature,
                 'actor_sha': self.actor_sha[actor], 'seed_contract': 'v1_actor_independent'})
            self.log('validation-complete', actor=actor, mode=mode, guard=guard, **summary)
        return result

    def retention(self, replay, adapter, name):
        import torch
        groups = defaultdict(list)
        for r in replay:groups[(r['tool'],r['reference_class'])].append(r)
        selected = [min(rs, key=lambda r: stable_seed(17, r['id'])) for _, rs in sorted(groups.items())]
        selected = selected[:16]; records = []
        for r in selected:
            with torch.no_grad():lp = float(self.scorer.logp(r['prompt_ids'], r['completion_ids'], adapter))
            records.append({'source_id':r['source_id'], 'id':r['id'], 'tool':r['tool'],
                'tokens':len(r['completion_ids']), 'action_mean_nll':-lp/len(r['completion_ids'])})
        result = {'actions': len(records), 'action_mean_nll': sum(r['action_mean_nll'] for r in records)/max(len(records),1),
            'records': records, 'scope':'TRAIN-only authentic replay diagnostic, not held-out chemistry'}
        write_json(self.output/'retention'/(name+'.json'), result)
        return result


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    cfg = yaml.safe_load(p.parse_args().config.read_text()); out = Path(cfg['output']); old = Path(cfg['legacy_source'])
    out.mkdir(parents=True, exist_ok=True)
    config_path = out/'resolved_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:raise ValueError('immutable resolved config changed')
    write_json(config_path, cfg)
    train = json.loads((old/'selected_train_ids.json').read_text()); valid = json.loads((old/'selected_valid_ids.json').read_text())
    if len(train) != 128 or len(valid) != 32:raise ValueError('frozen TRAIN128 / VALID32 changed')
    write_json(out/'selected_train_ids.json', train); write_json(out/'selected_valid_ids.json', valid)
    r0 = freeze_adapter(cfg['reverse_source'], out/'frozen/R0', R0_SHA)
    f0 = freeze_adapter(cfg['forward_source'], out/'frozen/F0', cfg['forward_sha'])
    if not (out/'S0/audit.json').exists():audit(old, cfg['inputs']['train']['path'], cfg['inputs']['train']['sha256'], out/'S0')
    from mechet.rsi_replay import replay_source_ids, build_replay_pool, mapped_replay_targets
    from transformers import AutoTokenizer
    ids = replay_source_ids(cfg['inputs']['train']['path'], [r['id'] for r in train], 512)
    references = reference_classes(cfg['inputs']['train']['path'], ids, cfg['inputs']['train']['sha256'])
    mapped_targets = mapped_replay_targets(cfg['mapped_replay_parent'], ids, cfg['mapped_replay_parent_sha'], out/'replay/original_mapped_targets.json')
    for identifier in ids:references[identifier]['replay_mapped_target'] = mapped_targets[identifier]
    valid_refs = reference_classes(cfg['inputs']['valid']['path'], [r['id'] for r in valid], cfg['inputs']['valid']['sha256'])
    valid_targets=mapped_replay_targets(cfg['mapped_valid_parent'],[r['id'] for r in valid],cfg['mapped_valid_parent_sha'],out/'valid_product_initial_states.json')
    from scripts.run_natural_language_value_search import visible
    root_changes=[]
    for split,rows,targets in (('train',train,mapped_targets),('valid',valid,valid_targets)):
        for row in rows:
            public=visible(targets[row['id']])
            if public!=row['target']:root_changes.append({'source_id':row['id'],'split':split,'v0_target':row['target'],'v1_target':public})
            row.update(target=public,initial_state=targets[row['id']])
    write_json(out/'aligned_product_inputs.json',{'train':train,'valid':valid,'representation_changes':root_changes,
        'no_future_GT_fields_in_prompt':True,'same_source_ids_as_v0':True,
        'boundary':'input_representation_fix_and_common_seeds; legacy_v0_is_not_exact_paired_weight_ablation'})
    tokenizer = AutoTokenizer.from_pretrained(out/'frozen/R0', local_files_only=True)
    replay, replay_manifest = build_replay_pool(cfg['replay_source'], ids, tokenizer, cfg['replay_sha'], out/'replay', references)
    write_json(out/'provenance.json', {'R0':r0, 'F0':f0, 'runtime_commit':os.environ.get('MECHET_RSI_CODE_SHA'),
        'legacy_source':str(old), 'legacy_train_selection_sha':sha256(old/'selected_train_ids.json'),
        'legacy_valid_selection_sha':sha256(old/'selected_valid_ids.json'), 'replay':replay_manifest,
        'test_used':False, 'primary_actor_selected_before_validation':'M3',
        'historical_source_episodes':'read-only S0 audit only; D0v1 newly generated from aligned source products',
        'control_budget_boundary':'equal scored examples/updates; active-loss tokens separately reported'})
    ex = AnchoredExperiment(cfg, out); ex.profile_rows = train;ex.valid_rows=valid
    try:
        ex.setup()
        if (out/'f0_v1_cache.json').exists():ex.roundtrips = json.loads((out/'f0_v1_cache.json').read_text())
        ex.register('R1v0', old/'models/R1/policy')
        if ex.actor_sha['R1v0'] != cfg['legacy_r1_sha']:raise ValueError('legacy R1 checkpoint changed')
        # Original v0 product normalization lost some explicit-H / Kekule state
        # parity. Keep those logs as S0 evidence; regenerate the actual v1 D0
        # under authenticated, same-for-every-actor product representations.
        d0,pairs = ex.pool_v1(train,'R0','D0v1',references)
        forward_paths = sorted(out.glob('D0v1_K*_forward.jsonl'))
        forward = [json.loads(line) for path in forward_paths for line in path.open()]
        ex.likelihood_gate(d0+forward, require_update=bool(pairs))
        retention0 = ex.retention(replay, 'reference', 'R0')
        training = {}; evaluations = {}
        from mechet.rsi_objective import anchored_plan,matched_round_plans
        plan = anchored_plan(pairs, replay, max_steps=cfg['max_optimizer_steps'], max_exposures=cfg['max_pair_exposures'])
        for name, mode in (('M1','replay_only'), ('M2','dpo_only'), ('M3','anchored')):
            info = ex.scorer.train_anchored(pairs, replay, out/'frozen/R0', R0_SHA, out/'models'/name,
                mode=mode, max_steps=20, max_exposures=4, lr=2e-6, beta=.1, replay_weight=.25,
                plan_override=plan if pairs else None)
            training[name] = info
            if info['status'] == 'updated':
                ex.register(name, info['path']); ex.retention(replay, 'policy', name)
            write_json(out/'training_summary.json', training)
        for name in ('R0','R1v0','M1','M2','M3'):
            if name not in ex.loras:continue
            evaluations[name] = ex.evaluate_v1(valid, name, valid_refs)
            publish_evaluation(out, evaluations, valid, training, cfg)
        # Guard is a controlled runtime ablation: BOTH R0 and M3, never mixed
        # into the primary all-guard-off policy comparison.
        if 'M3' in ex.loras and cfg['guard_ablation']:
            for name in ('R0','M3'):ex.evaluate_v1(valid, name, valid_refs, guard=True)
        recursive = {'status':'no_informative_updated_parent'}
        if 'M3' in ex.loras:
            # No VALID-based parent selection. Stop if authentic TRAIN replay
            # retention collapsed by >20%, before spending a fresh-data round.
            retention3 = json.loads((out/'retention/M3.json').read_text())
            if retention3['action_mean_nll'] <= retention0['action_mean_nll']*1.2:
                d1, p1 = ex.pool_v1(train, 'M3', 'D1v1', references)
                require_fresh_round(d1, ex.actor_sha['M3'], 'D1v1', {e['episode_id'] for e in d0})
                if p1:
                    round_plans, budget = matched_round_plans({'R2v1':p1,'R2staticD0':pairs},replay)
                    common_steps = budget['optimizer_steps']
                    if common_steps:
                        recursive = {'status':'fresh_data_updated','fresh_actor_sha':ex.actor_sha['M3'],
                            'fresh_episodes':len(d1),'fresh_pairs':len(p1),'matched_steps':common_steps,
                            'budget':budget}
                        for name, ps in (('R2v1',p1),('R2staticD0',pairs)):
                            info = ex.scorer.train_anchored(ps,replay,training['M3']['path'],R0_SHA,out/'models'/name,
                                max_steps=common_steps,mode='anchored',plan_override=round_plans[name])
                            training[name] = info
                            if info['status']=='updated':
                                ex.register(name,info['path']);evaluations[name]=ex.evaluate_v1(valid,name,valid_refs)
                            publish_evaluation(out,evaluations,valid,training,cfg)
                    else:recursive={'status':'fresh_pairs_but_no_matched_static_budget','fresh_pairs':len(p1)}
                else:recursive={'status':'fresh_collection_zero_strong_pairs','fresh_episodes':len(d1)}
            else:recursive={'status':'TRAIN_retention_gate_failed','baseline':retention0,'M3':retention3}
        write_json(out/'recursive_summary.json',recursive)
        for source,identity in ((cfg['reverse_source'],r0),(cfg['forward_source'],f0)):
            if sha256(Path(source)/'adapter_model.safetensors')!=identity['weight_sha256']:raise ValueError('source model mutated')
        ex.log('completed',training=training,recursive=recursive)
        write_json(out/'completion.json',{'status':'finished','training':training,'recursive':recursive,'test_used':False})
    except BaseException as error:
        write_json(out/'failure.json',{'error':f'{type(error).__name__}:{error}','timestamp':time.time()});raise
    finally:
        if hasattr(ex,'owner'):ex.owner.close()
        ex.raw.close()


def publish_evaluation(out,evaluations,valid,training,cfg):
    result={'fixed_f0_sha':cfg['forward_sha'],'seed_contract':'v1_actor_independent_seed17',
        'primary_guard':False,'training':training,'equal_budget_boundary':
        'same episode counts; reverse-token-matched censoring reported separately from uncapped F0 costs',
        'decoder':{'greedyK1':{'K':1,'temperature':0},'stochasticK2':{'K':2,'temperature':1},
                   'attempts':24,'tokens_per_attempt':384},'models':{},'token_matched':{}}
    for name,modes in evaluations.items():
        result['models'][name]={mode:productive_summary(es,[r['id'] for r in valid]) for mode,es in modes.items()}
    for mode in ('greedyK1','stochasticK2'):
        result['token_matched'][mode]=matched_views({name:modes[mode] for name,modes in evaluations.items()},[r['id'] for r in valid])
    write_json(out/'evaluation/paired_validation.json',result)


if __name__=='__main__':main()
