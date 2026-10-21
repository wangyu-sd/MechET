#!/usr/bin/env python3
"""PR91 two-round cycle-grounded policy iteration on frozen TRAIN/VALID IDs."""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from mechet.rsi_contract import (BASE_REVISION,R0_SHA,authenticate_adapter,fingerprint,
                                select_stratified,sha256,stable_seed,write_json,JsonJournal)


def require_fresh_round(episodes,actor_sha,round_id,old_ids):
    if not episodes:raise ValueError('empty fresh rollout pool')
    for e in episodes:
        if e['actor_sha']!=actor_sha or e['round']!=round_id or e['episode_id'] in old_ids:
            raise ValueError('D1 must be fresh R1-generated data')


def summarize_evaluation(episodes,target_ids):
    from mechet.roundtrip_reward import components
    from mechet.endpoints import split_precursor_endpoints
    ids=set(target_ids);hits={e['id'] for e in episodes if e.get('outcome',{}).get('success')}
    first_hits={e['id'] for e in episodes if e.get('candidate',0)==0 and e.get('outcome',{}).get('success')}
    observed={e['id'] for e in episodes}
    unique={fingerprint([e['id'],sorted(components(split_precursor_endpoints(e['final_state'],e['initial_state']).structural).items())]) for e in episodes if e.get('outcome',{}).get('success')}
    full_unique={fingerprint([e['id'],sorted(components(e['final_state']).items())]) for e in episodes if e.get('outcome',{}).get('success')}
    return {'targets':len(ids),'observed_targets':len(ids&observed),'missing_target_ids':sorted(ids-observed),
            'forward_verified_targets':len(ids&hits),'forward_verified_target_rate':len(ids&hits)/max(len(ids),1),
            'roundtrip_at_1_targets':len(ids&first_hits),'roundtrip_at_1_rate':len(ids&first_hits)/max(len(ids),1),
            'roundtrip_at_k_targets':len(ids&hits),'roundtrip_at_k_rate':len(ids&hits)/max(len(ids),1),
            'distinct_forward_verified_reactions':len(unique),'distinct_forward_verified_full_mixtures':len(full_unique),'episodes':len(episodes),
            'terminal_episodes':sum(bool(e.get('terminal')) for e in episodes),
            'reverse_generated_tokens':sum(len(s['completion_ids']) for e in episodes for s in e['steps']),
            'outcomes':dict(collections.Counter(e.get('outcome',{}).get('status','unknown') for e in episodes)),
            'errors_by_tool':dict(collections.Counter(s['name']+':'+str(s.get('error')) for e in episodes for s in e['steps'] if not s.get('accepted')))}


def stage_file(path,contract,producer):
    path=Path(path);identity=fingerprint(contract)
    if path.exists():
        data=json.loads(path.read_text())
        if data['contract_sha']!=identity:raise ValueError(f'immutable stage contract changed: {path}')
        return data['value']
    value=producer();write_json(path,{'contract_sha':identity,'contract':contract,'value':value});return value


def targets_needing_comparison(episodes,ids):
    from mechet.roundtrip_reward import preference_pairs
    groups=collections.defaultdict(list)
    for e in episodes:groups[e['id']].append(e)
    return [identifier for identifier in ids if not preference_pairs(groups[identifier])]


def check_likelihood_parity(records):
    errors=[abs(r['sampled_sum_logp']-r['scored_sum_logp'])/r['tokens'] for r in records
            if r.get('sampled_sum_logp') is not None and r['tokens']>0]
    return {'pass':bool(errors) and len(errors)==len(records) and max(errors)<=.03 and sum(errors)/len(errors)<=.01,
            'actions':len(records),'max_mean_logp_error':max(errors) if errors else None,
            'mean_mean_logp_error':sum(errors)/len(errors) if errors else None,
            'max_action_mean_logp_tolerance':.03,'mean_action_mean_logp_tolerance':.01}


def load_reaction_rows(path,expected,expected_sha=None,audit=None):
    """Source-only census; no model outputs or test labels select training IDs."""
    from scripts.run_natural_language_value_search import normal_smiles
    import hashlib
    rows=[];digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for line in f:
            digest.update(line)
            r=json.loads(line);m=r['metadata']
            proof=r['messages'][-1]['content']
            rows.append({'id':r['id'],'target':normal_smiles(r['target_smiles']),
                 'event_count':int(m.get('n_edges',proof.count('\nEDGE '))),
                 'import_count':proof.count('\n  IMPORT '),'recorded_reactants':m['initial_reactants'],
                 'reference_structural':r['structural_precursor'],'source_split':m['source_split']})
    if len(rows)!=expected or len({r['id'] for r in rows})!=expected:raise ValueError('strict reaction universe changed')
    if expected_sha and digest.hexdigest()!=expected_sha:raise ValueError('proof input SHA mismatch')
    if audit is not None:audit.update(path=str(Path(path).resolve()),rows=len(rows),sha256=digest.hexdigest(),
        view='strict_executable_proof',eligibility_filter='named4train_0valid_4test_corrupt_upstream_rows_only',
        overlap_filtering=False,token_length_filtering=False,smoke_selection='stratified128train_32valid_seed17')
    return rows


def freeze_adapter(source,destination,expected=None):
    destination=Path(destination)
    if (destination/'frozen_identity.json').exists():
        previous=json.loads((destination/'frozen_identity.json').read_text())
        if sha256(destination/'adapter_model.safetensors')!=previous['weight_sha256'] or (expected and previous['weight_sha256']!=expected):
            raise ValueError('frozen snapshot corrupt or identity changed')
        return previous
    before=authenticate_adapter(source,expected)
    destination.mkdir(parents=True,exist_ok=True)
    for p in Path(source).iterdir():
        if p.is_file() and (p.suffix in {'.json','.safetensors','.jinja','.txt'} or p.name=='README.md'):
            if p.name=='trainer_state.json':continue
            shutil.copy2(p,Path(destination)/p.name)
    after=authenticate_adapter(source,expected)
    if before['weight_sha256']!=after['weight_sha256']:raise ValueError('source changed during freeze')
    if sha256(Path(destination)/'adapter_model.safetensors')!=before['weight_sha256']:raise ValueError('freeze copy corrupt')
    write_json(destination/'frozen_identity.json',before)
    return before


class Experiment:
    def __init__(self,cfg,output):
        self.cfg=cfg;self.output=Path(output);self.stats=[];self.roundtrips={};self.loras={};self.forward_train_attempts={}
        self.raw=JsonJournal(self.output/'phase_events.jsonl')

    def log(self,phase,**values):
        r={'phase':phase,'timestamp':time.time(),**values};self.raw.append(r);print(r,flush=True)

    def setup(self):
        from mechet.forward_qwen_events import forward_tools
        from mechet.rsi_objective import PolicyScorer
        from scripts.rsi_runtime import GPUOwner
        import torch,vllm,rdkit
        from vllm import LLMEngine,EngineArgs,SamplingParams
        from vllm.lora.request import LoRARequest
        from scripts.vllm_safe_lora_expand import install_safe_lora_expand
        if torch.cuda.device_count()!=2 or rdkit.__version__!='2026.03.4' or vllm.__version__.split('+')[0]!='0.8.5':
            raise ValueError('PR91 requires verified two-GPU/RDKit2026/vLLM0.8.5 runtime')
        self.lora_type=LoRARequest
        runtime=install_safe_lora_expand(vllm.__version__)
        args=EngineArgs(model='Qwen/Qwen3-0.6B',revision=BASE_REVISION,dtype='bfloat16',
           max_model_len=16384,max_num_seqs=64,max_num_batched_tokens=16384,gpu_memory_utilization=.75,
           enable_lora=True,max_loras=1,max_cpu_loras=8,max_lora_rank=16,enforce_eager=True,disable_log_stats=False)
        self.engine=LLMEngine.from_engine_args(args);self.tokenizer=self.engine.get_tokenizer()
        self.actor_sha={'R0':R0_SHA,'F0':self.cfg['forward_sha']}
        for name in ['R0','F0']:self.register(name,self.output/'frozen'/name)
        self.prompt_gate()
        self.profile_decode()
        eos=self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        def params(r):return SamplingParams(n=1,temperature=r.get('temperature',1.),top_p=1.,top_k=-1,max_tokens=r['max_tokens'],
                     seed=r['seed'],stop_token_ids=[eos],ignore_eos=False,logprobs=1)
        self.owner=GPUOwner(self.engine,params,max_inflight=64)
        self.scorer=PolicyScorer('Qwen/Qwen3-0.6B',self.output/'frozen/R0',self.output/'frozen/F0',device='cuda:1')
        self.log('runtime-ready',torch=torch.__version__,rdkit=rdkit.__version__,vllm=vllm.__version__,lora_runtime=runtime)

    def register(self,name,path):
        self.loras[name]=self.lora_type(name,len(self.loras)+1,str(Path(path).resolve()))
        if name not in self.actor_sha:self.actor_sha[name]=sha256(Path(path)/'adapter_model.safetensors')

    def prompt_gate(self):
        from scripts.rsi_runtime import reverse_prompt
        from scripts.run_natural_language_value_search import Node,private_product_state,visible
        from mechet.in_place_grounded_flow import mapped_atom_numbers
        from mechet.assistant_masking import render_qwen_sft_tool_prefix
        source=Path(self.cfg['shared'])/'data/flower_natural_language_event_history_v2/train.jsonl'
        with source.open() as f:r=json.loads(next(f))
        state=private_product_state(r['target_smiles']);node=Node(target=visible(state),state=state,next_map=max(mapped_atom_numbers(state))+1)
        actual=reverse_prompt(node,self.tokenizer)
        expected=render_qwen_sft_tool_prefix(self.tokenizer,r['messages'][:2],tools=r['tools'])
        if actual!=expected:raise ValueError('R0 SFT/current history prompt parity FAILED')
        write_json(self.output/'prompt_parity.json',{'source_id':r['source_id'],'exact_prefix':True,
            'prompt_sha':fingerprint(actual),'im_end_id':self.tokenizer.convert_tokens_to_ids('<|im_end|>'),
            'source_manifest':json.loads((Path(self.cfg['reverse_source'])/'adapter_manifest.json').read_text()),
            'no_manifest_overwrite':True})

    def profile_decode(self):
        """Paired ABBA decode-only measurements; never chemical accuracy."""
        from scripts.rsi_runtime import reverse_prompt
        from scripts.run_natural_language_value_search import Node,private_product_state,visible
        from mechet.in_place_grounded_flow import mapped_atom_numbers
        from vllm import SamplingParams
        requests=[]
        for row in self.profile_rows[:16]:
            state=private_product_state(row.get('initial_state') or row['target']);node=Node(target=visible(state),state=state,next_map=max(mapped_atom_numbers(state))+1)
            prompt=reverse_prompt(node,self.tokenizer);ids=self.tokenizer.encode(prompt,add_special_tokens=False)
            requests.append((ids,stable_seed(17,'decode_profile',row['id'])))
        results=[];raw=JsonJournal(self.output/'decode_profile_raw.jsonl')
        for run,mode in enumerate(['warmup','wave','rolling','rolling','wave']):
            pending=collections.deque(enumerate(requests[:4] if mode=='warmup' else requests));active=set();outputs={};start=time.monotonic()
            while pending or active:
                if mode!='wave' or not active:
                    while pending and len(active)<8:
                        index,(ids,seed)=pending.popleft();uid=f'profile-{run}-{index}'
                        self.engine.add_request(uid,{'prompt_token_ids':ids},SamplingParams(n=1,temperature=1.,top_p=1.,top_k=-1,
                            max_tokens=32,seed=seed,stop_token_ids=[self.tokenizer.convert_tokens_to_ids('<|im_end|>')]),lora_request=self.loras['R0'])
                        active.add(uid)
                for o in self.engine.step():
                    if o.finished:
                        active.remove(o.request_id);outputs[int(o.request_id.rsplit('-',1)[1])]=list(o.outputs[0].token_ids)
            seconds=time.monotonic()-start;tokens=sum(map(len,outputs.values()))
            entry={'mode':mode,'run':run,'seconds':seconds,'tokens':tokens,'tokens_per_second':tokens/max(seconds,1e-9),'output_ids':outputs}
            raw.append(entry)
            if mode!='warmup':results.append(entry)
            self.log('decode-profile',mode=mode,seconds=seconds,tokens=tokens)
        raw.close();same=all(r['output_ids']==results[0]['output_ids'] for r in results)
        wave=sum(r['seconds'] for r in results if r['mode']=='wave');rolling=sum(r['seconds'] for r in results if r['mode']=='rolling')
        self.decode_profile={'measurements':[{k:v for k,v in r.items() if k!='output_ids'} for r in results],
            'identical_output_tokens':same,'wave_over_rolling_wall_ratio':wave/rolling if same else None,
            'boundary':'same16TRAINprompts,32tokens,ABBA,wave8versusrolling8; decode-only, NOT whole-agent speedup'}
        write_json(self.output/'evaluation/decode_speed_comparison.json',self.decode_profile)

    def collect(self,rows,actor,phase,indices,direction='reverse'):
        from scripts.rsi_runtime import collect_episodes
        indices=list(indices)
        def produce():
            start=time.monotonic()
            actions_path=Path(str(self.output/(phase+'.jsonl'))+'.actions.jsonl')
            offset=actions_path.stat().st_size if actions_path.exists() else 0
            episodes=collect_episodes(rows,owner=self.owner,tokenizer=self.tokenizer,lora=self.loras[actor],
                 actor_sha=self.actor_sha[actor],round_id=phase,sample_indices=indices,direction=direction,
                 journal_path=self.output/(phase+'.jsonl'),max_attempts=24,max_tokens=384)
            seconds=time.monotonic()-start;total_tokens=sum(len(s['completion_ids']) for e in episodes for s in e['steps']);tokens=0
            if actions_path.exists():
                with actions_path.open('rb') as f:
                    f.seek(offset)
                    for line in f:tokens+=len(json.loads(line)['completion_ids'])
            return {'episodes':episodes,'measurement':{'phase':phase,'seconds':seconds,'generated_tokens':tokens,
                'total_episode_tokens_including_recovered':total_tokens,'recovered_token_count':total_tokens-tokens,
                'tokens_per_second':tokens/max(seconds,1e-9),'episodes':len(episodes),'actor_sha':self.actor_sha[actor],
                'cpu_executor_wall_sum':sum(s.get('cpu_executor_wall_seconds',0) for e in episodes for s in e['steps'])}}
        cached=stage_file(self.output/'stages'/(phase+'.json'),{'rows':rows,'actor_sha':self.actor_sha[actor],
             'phase':phase,'indices':indices,'direction':direction,'attempts':24,'tokens':384,
             'runtime_source_sha':sha256(ROOT/'scripts/rsi_runtime.py')},produce)
        episodes=sorted(cached['episodes'],key=lambda e:e['episode_id']);self.stats.append(cached['measurement'])
        self.log('collection-complete',measurement=self.stats[-1]);return episodes

    def likelihood_gate(self,episodes,require_update=True):
        import torch
        selected={}
        for label,adapter in [('R0','reference'),('F0','forward_judge')]:
            candidates=[]
            for e in sorted(episodes,key=lambda e:e['episode_id']):
                if e['actor_sha']!=self.actor_sha[label]:continue
                candidates.extend(s for s in e['steps'] if s.get('complete') and s.get('generated_sum_logp') is not None)
            records=[]
            for s in candidates[:8]:
                with torch.no_grad():scored=float(self.scorer.logp(s['prompt_ids'],s['completion_ids'],adapter))
                records.append({'tokens':len(s['completion_ids']),'sampled_sum_logp':s['generated_sum_logp'],'scored_sum_logp':scored})
            selected[label]={'records':records,**check_likelihood_parity(records)}
        write_json(self.output/'sampled_vs_teacher_forced_likelihood_gate.json',selected)
        if require_update and not all(v['pass'] for v in selected.values()):raise ValueError('sampled/HF exact-action likelihood gate FAILED; no DPO allowed')

    def feedback(self,episodes,phase,forward_limit=None):
        from mechet.roundtrip_bridge import bridge_reverse_episode
        from mechet.roundtrip_reward import forward_outcome,component_match,components
        from mechet.endpoints import split_precursor_endpoints
        audit=JsonJournal(self.output/(phase+'.bridge_audit.jsonl'));scores=JsonJournal(self.output/(phase+'.forward_witness_scores.jsonl'))
        episodes=sorted(episodes,key=lambda e:e['episode_id']);unique={};groups={}
        for e in episodes:
            if not e['terminal'] or not e['flow_count']:
                e['outcome']={'status':'inverse_incomplete' if not e['terminal'] else 'nonproductive','success':False};continue
            bridge=bridge_reverse_episode(e);key=fingerprint([self.actor_sha['F0'],e['target'],sorted(components(e['final_state']).items())])
            e['bridge_key']=key;groups.setdefault(key,[]).append(e)
            audit.append({'episode_id':e['episode_id'],'bridge':bridge,
                'endpoint_views':split_precursor_endpoints(e['final_state'],e['initial_state']).to_dict()})
            if key in unique:continue
            entry={'bridge':bridge,'target':e['target'],'key':key,'witness_mean_logp':None}
            if bridge['status']=='verified':
                try:
                    score=self.scorer.witness_score(bridge['witness'],self.actor_sha['F0']);entry['witness_mean_logp']=score['mean_logp']
                    scores.append({'key':key,'episode_id':e['episode_id'],**score})
                except Exception as exc:scores.append({'key':key,'error':f'{type(exc).__name__}:{exc}'})
            unique[key]=entry
        eligible=[x for x in unique.values() if x['bridge']['initial_state'] and not component_match(x['bridge']['initial_state'],x['target'])]
        # Deliberately sample both high/low path support, never let the score
        # itself manufacture a positive label. Validation has no forward cap.
        eligible.sort(key=lambda x:(x['witness_mean_logp'] is None,x['witness_mean_logp'] or -1e9,x['key']))
        ordered=[]
        while eligible:
            ordered.append(eligible.pop(0))
            if eligible:ordered.append(eligible.pop())
        chosen=ordered if forward_limit is None else ordered[:forward_limit]
        check_rows=[{'id':x['key'],'initial_state':x['bridge']['initial_state']} for x in chosen if x['key'] not in self.roundtrips]
        if check_rows:
            forward=self.collect(check_rows,'F0',phase+'_forward',[0],direction='forward')
            for f in forward:
                target=unique[f['id']]['target'];outcome=forward_outcome(f,target)
                self.roundtrips[f['id']]={'outcome':outcome,'episode_id':f['episode_id'],'generated_tokens':sum(len(s['completion_ids']) for s in f['steps'])}
        for key,values in groups.items():
            if key in self.roundtrips:out=self.roundtrips[key]['outcome']
            elif component_match(values[0]['final_state'],values[0]['target']):out={'status':'nonproductive','success':False}
            else:out={'status':'forward_unchecked','success':False}
            for e in values:e['outcome']=out
        audit.close();scores.close()
        write_json(self.output/(phase+'.feedback.json'),{'episodes':episodes,'unique_candidates':len(unique),
          'witness_verified':sum(x['bridge']['status']=='verified' for x in unique.values()),
          'forward_attempts':len(check_rows),'proxy_used_for_preferences':False,
          'proxy_calibration':self.calibration(unique)})
        return episodes

    def calibration(self,entries):
        measured=[(v['witness_mean_logp'],bool(self.roundtrips[k]['outcome']['success'])) for k,v in entries.items()
                  if k in self.roundtrips and v['witness_mean_logp'] is not None]
        good=[s for s,y in measured if y];bad=[s for s,y in measured if not y]
        auc=(sum((a>b)+.5*(a==b) for a in good for b in bad)/(len(good)*len(bad))) if good and bad else None
        return {'measured_candidates':len(measured),'positive':len(good),'negative_or_censored':len(bad),
                'path_support_pairwise_auc':auc,'used_as_DPO_positive':False,
                'boundary':'censored forward runs are not chemical negatives; score never substitutes for autonomous success'}

    def train_pool(self,rows,actor,phase):
        from mechet.roundtrip_reward import preference_pairs
        episodes=[];pairs=[];sampling_rows=rows
        for lower,upper in [(0,2),(2,4),(4,8)]:
            self.log('adaptive-stage',actor=actor,round_id=phase,lower=lower,upper=upper,targets=len(rows))
            batch=self.collect(sampling_rows,actor,phase+f'_K{upper}',range(lower,upper))
            # Round identity is explicit; episode IDs retain generation phase.
            for e in batch:e['collection_phase']=e['round'];e['round']=phase
            budget_key=phase;used=self.forward_train_attempts.get(budget_key,0)
            before=len(self.roundtrips);stage_limit=min(32 if upper==2 else 16,max(64-used,0))
            episodes+=self.feedback(batch,phase+f'_K{upper}',forward_limit=stage_limit)
            self.forward_train_attempts[budget_key]=used+len(self.roundtrips)-before
            by_id=collections.defaultdict(list)
            for e in episodes:by_id[e['id']].append(e)
            pairs=sorted([p for identifier in sorted(by_id) for p in preference_pairs(by_id[identifier])],key=lambda p:p['id'])
            unresolved=set(targets_needing_comparison(episodes,[r['id'] for r in rows]))
            sampling_rows=[r for r in rows if r['id'] in unresolved]
            if not sampling_rows:break
        write_json(self.output/(phase+'.allocation.json'),{'forward_checks_used':self.forward_train_attempts.get(phase,0),
            'max_forward_checks_for_this128target_cohort':64,'informative_target_ids':sorted({e['id'] for e in episodes}-{r['id'] for r in sampling_rows})})
        return episodes,pairs

    def evaluate(self,rows,actor):
        episodes=self.collect(rows,actor,'valid_'+actor,[0,1])
        episodes=self.feedback(episodes,'valid_'+actor,forward_limit=None)
        from mechet.endpoints import split_precursor_endpoints,structural_exact
        refs={r['id']:r for r in rows}
        for e in episodes:
            try:e['reference_structural_exact']=structural_exact(split_precursor_endpoints(e['final_state'],e['initial_state']).structural,refs[e['id']]['reference_structural'])
            except Exception:e['reference_structural_exact']=False
        summary=summarize_evaluation(episodes,[r['id'] for r in rows]);summary['reference_structural_targets']=len({e['id'] for e in episodes if e['reference_structural_exact'] and e['terminal']})
        write_json(self.output/'evaluation'/f'{actor}.json',{'summary':summary,'episodes':episodes});return episodes,summary

    def forward_update(self,train,valid):
        """Roles-swapped bounded extension; inverse mismatch remains unknown."""
        from rdkit import Chem
        from mechet.forward_qwen_events import initialize_forward_state
        from mechet.endpoints import split_precursor_endpoints,structural_exact
        from mechet.roundtrip_reward import preference_pairs,component_match,reconstruction_success
        from mechet.roundtrip_bridge import mol,text
        source=[{'id':r['id'],'initial_state':initialize_forward_state(r['recorded_reactants'])} for r in train[:32]]
        forward=self.collect(source,'F0','F1_train_F0',[0,1],direction='forward')
        reverse_rows=[];wanted={};by_episode={}
        for e in forward:
            e['outcome']={'status':'nonproductive' if not e['flow_count'] else 'unknown','success':False}
            if not e['terminal'] or not e['flow_count']:continue
            fragments=[m for m in Chem.GetMolFrags(mol(e['final_state']),asMols=True) if any(a.GetAtomicNum()==6 for a in m.GetAtoms())]
            if not fragments:continue
            product=max(fragments,key=lambda m:(m.GetNumHeavyAtoms(),text(m)));mapped_product=text(product)
            if component_match(e['initial_state'],mapped_product):
                e['outcome']={'status':'nonproductive','success':False};continue
            for a in product.GetAtoms():a.SetAtomMapNum(0)
            key=e['episode_id'];reverse_rows.append({'id':key,'target':text(product)})
            wanted[key]=split_precursor_endpoints(e['initial_state'],mapped_product).structural;by_episode[key]=e
        if reverse_rows:
            back=self.collect(reverse_rows,'R2','F1_R2_reconstruction',[0])
            for e in back:
                if not e['terminal']:continue
                if reconstruction_success(e,wanted[e['id']]):
                    by_episode[e['id']]['outcome']={'status':'roundtrip_hit','success':True}
        grouped=collections.defaultdict(list)
        for e in forward:grouped[e['id']].append(e)
        pairs=[p for group in grouped.values() for p in preference_pairs(group)]
        write_json(self.output/'F1_preferences.json',pairs)
        result=self.scorer.train_dpo(pairs,self.output/'frozen/F0',self.cfg['forward_sha'],self.output/'models/F1',
                                      steps=50,reference_adapter='forward_judge')
        if result['status']=='updated':
            self.register('F1',result['path'])
            rows=[{'id':r['id'],'initial_state':initialize_forward_state(r['recorded_reactants'])} for r in valid]
            sampled=self.collect(rows,'F1','F1_recorded_forward_valid',[0],direction='forward')
            from mechet.roundtrip_reward import forward_outcome
            targets={r['id']:r['target'] for r in valid}
            result['validation']=[{'id':e['id'],'outcome':forward_outcome(e,targets[e['id']])} for e in sampled]
        result['forward_episode_count']=len(forward);result['reverse_check_count']=len(reverse_rows)
        write_json(self.output/'evaluation/F1.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(args.config.read_text());output=Path(cfg['output'])
    if (output/'artifact_manifest.json').exists():
        finished=json.loads((output/'artifact_manifest.json').read_text())
        for row in finished['files']:
            if sha256(output/row['path'])!=row['sha256']:raise ValueError('completed run artifact mismatch')
        print({'phase':'already-complete-verified','manifest':str(output/'artifact_manifest.json')},flush=True);return
    output.mkdir(parents=True,exist_ok=True)
    if (output/'smoke_config_resolved.yaml').exists():
        if yaml.safe_load((output/'smoke_config_resolved.yaml').read_text())!=cfg:raise ValueError('resume config mismatch')
    else:shutil.copy2(args.config,output/'smoke_config_resolved.yaml')
    r0=freeze_adapter(cfg['reverse_source'],output/'frozen/R0',R0_SHA)
    f0=freeze_adapter(cfg['forward_source'],output/'frozen/F0',cfg['forward_sha'])
    input_audit={}
    train_all=load_reaction_rows(cfg['inputs']['train']['path'],257167,cfg['inputs']['train']['sha256'],input_audit.setdefault('train',{}))
    valid_all=load_reaction_rows(cfg['inputs']['valid']['path'],2890,cfg['inputs']['valid']['sha256'],input_audit.setdefault('valid',{}))
    reserved=select_stratified(train_all,256,17);train=reserved[:128];valid=select_stratified(valid_all,32,17)
    if {r['id'] for r in reserved}&{r['id'] for r in valid}:raise ValueError('TRAIN/VALID overlap')
    write_json(output/'selected_train_ids.json',train);write_json(output/'selected_valid_ids.json',valid)
    write_json(output/'provenance.json',{'R0':r0,'F0':f0,'runtime_commit':os.environ.get('MECHET_RSI_CODE_SHA'),
        'reverse_source_data_contract':json.loads((Path(cfg['reverse_source'])/'data_contract.json').read_text()),
        'input_contracts':input_audit,
        'test_used':False,'selection':'event_length_import_count_stratified_seed17','no_source_weights_overwritten':True})
    experiment=Experiment(cfg,output);experiment.profile_rows=train
    try:
        experiment.setup()
        from mechet.forward_qwen_events import initialize_forward_state
        from mechet.roundtrip_reward import forward_outcome
        # Independent recorded-reactant forward baseline, not reversed witness.
        source_forward=[{'id':r['id'],'initial_state':initialize_forward_state(r['recorded_reactants'])} for r in train[:4]+valid]
        baseline=experiment.collect(source_forward,'F0','F0_recorded_reactants',[0],direction='forward')
        targets={r['id']:r['target'] for r in train[:4]+valid}
        write_json(output/'evaluation/F0_recorded_forward.json',[{'id':e['id'],'outcome':forward_outcome(e,targets[e['id']])} for e in baseline])
        evaluations={};valid_episodes={};f1={'status':'no_update','reason':'R2 not available for comparable forward feedback'}
        valid_episodes['R0'],evaluations['R0']=experiment.evaluate(valid,'R0')
        d0,p0=experiment.train_pool(train,'R0','D0')
        if not p0:
            train=reserved;write_json(output/'selected_train_ids_expanded256.json',train)
            extra,extra_pairs=experiment.train_pool(train[128:],'R0','D0_expansion256')
            d0+=extra;p0+=extra_pairs
        write_json(output/'d0_preferences.json',p0)
        # Actual F0 and R0 sampled likelihoods must agree with training scoring
        # on identical generated action IDs before any policy update.
        experiment.likelihood_gate(baseline+valid_episodes['R0']+d0,require_update=bool(p0))
        r1=experiment.scorer.train_dpo(p0,output/'frozen/R0',R0_SHA,output/'models/R1',steps=50)
        training={'R1':r1};d1=[];p1=[]
        if r1['status']=='updated':
            experiment.register('R1',r1['path']);valid_episodes['R1'],evaluations['R1']=experiment.evaluate(valid,'R1')
            d1,p1=experiment.train_pool(train,'R1','D1')
            require_fresh_round(d1,r1['weight_sha256'],'D1',{e['episode_id'] for e in d0})
            write_json(output/'d1_preferences.json',p1)
            r2=experiment.scorer.train_dpo(p1,r1['path'],R0_SHA,output/'models/R2',steps=50);training['R2']=r2
            if r2['status']=='updated':
                experiment.register('R2',r2['path']);valid_episodes['R2'],evaluations['R2']=experiment.evaluate(valid,'R2')
                static=experiment.scorer.train_dpo(p0,r1['path'],R0_SHA,output/'models/R2_static_D0',steps=r2['optimizer_steps'])
                training['R2_static_D0']=static;experiment.register('R2_static_D0',static['path'])
                valid_episodes['R2_static_D0'],evaluations['R2_static_D0']=experiment.evaluate(valid,'R2_static_D0')
                f1=experiment.forward_update(train,valid)
        # Token-matched conservative prefix analysis retains the FULL32 target
        # denominator, including episodes not reached by the common budget.
        budget=min(v['reverse_generated_tokens'] for v in evaluations.values());matched={}
        for name,episodes in valid_episodes.items():
            kept=[];used=0
            for e in sorted(episodes,key=lambda e:(e['id'],e['candidate'])):
                cost=sum(len(s['completion_ids']) for s in e['steps'])
                if used+cost>budget:break
                kept.append(e);used+=cost
            matched[name]={'spent_tokens':used,**summarize_evaluation(kept,[r['id'] for r in valid])}
        write_json(output/'evaluation/paired_validation.json',{'fixed_f0_sha':cfg['forward_sha'],'decoder':{'temperature':1.,'top_p':1.,'top_k':-1,'K':2,'attempts':24,'tokens_per_attempt':384},
            'models':evaluations,'common_reverse_token_budget':budget,'token_matched_prefix':matched,'training':training,
            'F1':f1})
        write_json(output/'evaluation/speed_comparison.json',{'measured_phases':experiment.stats,'decode_only_ABBA':experiment.decode_profile,
            'whole_agent_speedup_claim':None,'boundary':'decode-only ABBA separated from actual full-tool-loop costs'})
        for source,identity in [(cfg['reverse_source'],r0),(cfg['forward_source'],f0)]:
            if sha256(Path(source)/'adapter_model.safetensors')!=identity['weight_sha256']:raise ValueError('source mutated')
        experiment.log('completed',training=training,evaluation=evaluations)
        experiment.raw.close()
        (output/'README.md').write_text(
            '# PR91 executable round-trip policy iteration\n\n'
            'Measured run; original TRAIN/VALID selections and all failed/censored actions retained. No TEST used.\n\n'
            'F0 is the provisional checkpoint-54000, not a completed two-epoch model. '
            'Forward witness log likelihood is not autonomous reconstruction or chemical truth. '
            'This run checks full terminal mixtures; a separate structural-only forward-input experiment is not claimed.\n\n'
            'Command: `python -u scripts/run_mechet_rsi_smoke.py --config configs/experiments/mechet_rsi_resolved_20261021.yaml`\n\n'
            'Exact runtime commit and source identities: `provenance.json`. '
            'Complete arguments: `smoke_config_resolved.yaml`. '
            'Sampled prompts, token IDs, calls, states and failures: `*.actions.jsonl`.\n\n'
            '## Paired validation and actual updates\n\n```json\n'+json.dumps({'training':training,'validation':evaluations,'F1':f1},indent=2)+'\n```\n',
            encoding='utf-8')
        files=[p for p in output.rglob('*') if p.is_file()]
        write_json(output/'artifact_manifest.json',{'files':[{ 'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files],
            'status':'reverse_two_round_run_finished' if training.get('R2',{}).get('status')=='updated' else 'no_complete_recursive_update',
            'no_test_used':True,'F1_updated':f1['status']=='updated'})
    except BaseException as exc:
        write_json(output/'failure.json',{'error':f'{type(exc).__name__}:{exc}','time':time.time()});raise
    finally:
        if hasattr(experiment,'owner'):experiment.owner.close()
        experiment.raw.close()


if __name__=='__main__':main()
