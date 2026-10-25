#!/usr/bin/env python3
"""Full accepted-TRAIN PR96 Stage III: refreshed-policy, adaptive-horizon EARHO.

Uses the immutable PR96 parent source as its imports, not a rewritten executor.
No TEST labels, graph policy, external critic or separate precursor decoder.
"""
import argparse
from collections import defaultdict
import copy
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

ENTRY = Path(__file__).resolve().parent
REFERENCE = Path(os.environ.get('MECHET_PR96_REFERENCE_CODE_ROOT', ENTRY.parent))
sys.path[:0] = [str(REFERENCE/'src'), str(REFERENCE)]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


contract_api = import_file('pr96_earho_contract', ENTRY/'pr96_earho_contract.py')
storage = import_file('pr96_earho_storage', ENTRY/'pr96_earho_storage.py')
evaluation = import_file('pr96_earho_evaluation', ENTRY/'evaluate_pr96_test.py')


def rows(path):
    with Path(path).open() as reader:
        for line in reader:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.building')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def append(path, value):
    with Path(path).open('a') as sink:
        sink.write(json.dumps(value, ensure_ascii=False)+'\n')
        sink.flush()


def emit(root, rank, event, **values):
    record = dict(event=event, timestamp=time.time(), rank=rank, **values)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    append(root/f'progress.rank{rank:02d}.jsonl', record)
    write(root/f'progress.rank{rank:02d}.json', record)


def replay(source, history):
    from mechet.pr96_evaluation import public_root, messages_for
    from scripts.run_natural_language_value_search import Action, execute
    from scripts.build_mech31k_main_product_only_sft import same_chemical_state
    node = public_root(source)
    nodes = [node]
    for index, row in enumerate(history):
        if (int(row['metadata']['decision_index']) != index
                or messages_for(node, True) != row['messages'][:2]):
            raise ValueError('EARHO_HISTORY_PREFIX_MISMATCH:'+str(source['source_id']))
        fn = row['messages'][2]['tool_calls'][0]['function']
        child, error = execute(node, Action(fn['name'], fn['arguments'], '', 0, 0), max_imports=32)
        if child is None:
            raise ValueError('EARHO_REFERENCE_EXECUTION_FAILED:'+error)
        nodes.append(child)
        node = child
    if not node.terminal or not same_chemical_state(node.state, source['full_precursor_state']):
        raise ValueError('EARHO_REFERENCE_ENDPOINT_MISMATCH')
    return nodes


def model_step(node, output):
    from mechet.pr96_evaluation import parsed
    from scripts.run_natural_language_value_search import Action, execute
    name, arguments, error = parsed(output)
    if error:
        return None, error
    return execute(node, Action(name, arguments, output['raw'], 0,
                               len(output['generated_token_ids'])), max_imports=32)


class SampleBackend:
    """Unwarped on-policy K2 first proposals; all continuations are greedy."""
    def __init__(self, model, tokenizer, cfg, raw_path):
        self.model, self.tokenizer, self.cfg, self.raw_path = model, tokenizer, cfg, raw_path

    def generate(self, messages):
        import torch
        from transformers import GenerationConfig
        from mechet.qwen35_protocol import tool_prefix
        from scripts.build_natural_language_event_sft import TOOLS
        results = []
        for start in range(0, len(messages), self.cfg['generation_batch_size']):
            batch = messages[start:start+self.cfg['generation_batch_size']]
            prompts = [tool_prefix(self.tokenizer, m, TOOLS) for m in batch]
            encoded = self.tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False)
            width = encoded['input_ids'].shape[1]
            if width+self.cfg['max_new_tokens'] > self.model.config.max_position_embeddings:
                raise ValueError('EARHO_CONTEXT_OVERFLOW_NO_TRUNCATION')
            prompt_ids = [v[m.bool()].tolist() for v,m in zip(encoded['input_ids'], encoded['attention_mask'])]
            encoded = {k:v.to(next(self.model.parameters()).device) for k,v in encoded.items()}
            generation = GenerationConfig.from_model_config(self.model.config)
            generation.do_sample, generation.num_beams = True, 1
            generation.temperature, generation.top_p, generation.top_k = 1., 1., 0
            generation.pad_token_id = self.tokenizer.pad_token_id
            generation.eos_token_id = self.tokenizer.eos_token_id
            with torch.inference_mode():
                generated = self.model.generate(**encoded, generation_config=generation,
                                               max_new_tokens=self.cfg['max_new_tokens'], use_cache=True)
            for i, sequence in enumerate(generated):
                ids = sequence[width:].tolist()
                ended = self.tokenizer.eos_token_id in ids
                if ended:
                    ids = ids[:ids.index(self.tokenizer.eos_token_id)+1]
                result = dict(raw=self.tokenizer.decode(ids, skip_special_tokens=False),
                              generated_token_ids=ids, prompt_token_ids=prompt_ids[i], truncated=not ended)
                append(self.raw_path, dict(messages=batch[i], prompt=prompts[i], **result))
                results.append(result)
            print(json.dumps(dict(event='earho-first-action-generation', completed=len(results), total=len(messages))), flush=True)
        return results


def public_rows(sources):
    from mechet.pr96_evaluation import public_root
    result = []
    for source in sources:
        node = public_root(source)
        result.append(dict(source_id=str(source['source_id']), product_smiles=node.target,
                           initial_mapped_product_state=node.state))
    return result


def run_episodes(source, model, tokenizer, cfg, directory):
    from mechet.pr96_eval_runtime import HFBackend
    directory.mkdir(parents=True, exist_ok=True)
    backend = HFBackend(model, tokenizer, max_new_tokens=cfg['max_new_tokens'],
                        batch_size=cfg['generation_batch_size'], raw_path=directory/'raw_generation.jsonl')
    calls = []
    def journal(value):
        append(directory/'calls.jsonl', value)
        if value['mode'] == 'closed_loop':
            calls.append(value)
    model.eval()
    predictions = evaluation.rollout(public_rows(source), backend, compact_history=True,
                                    max_decisions=cfg['max_decisions'], journal=journal)
    indexed = {str(s['source_id']):s for s in source}
    scored = [evaluation.score_endpoint(v, indexed[v['reaction_id']]) for v in predictions]
    write(directory/'episodes.json', scored)
    return scored, calls


def collect(ids, source, history, nodes, model, reference, tokenizer, cfg, directory, horizon, rehearsal):
    import torch
    from mechet.pr96_eval_runtime import HFBackend
    from mechet.pr96_evaluation import messages_for
    from mechet.qwen35_protocol import encode_tool_decision
    from scripts.build_mech31k_main_product_only_sft import same_chemical_state
    baseline, calls = run_episodes([source[r] for r in ids], model, tokenizer, cfg, directory/'product_start')
    indexed = {v['reaction_id']:v for v in baseline}
    byid = defaultdict(list)
    for call in calls:
        byid[call['reaction_id']].append(call)
    groups, branches = [], []
    for rid in ids:
        index = contract_api.first_divergence(indexed[rid], byid[rid], [n.state for n in nodes[rid]],
                    same_chemical_state, reference_terminals=[n.terminal for n in nodes[rid]])
        full = rid in rehearsal
        if index is None and not full:
            continue
        index = 0 if full else index
        group = dict(reaction_id=rid, index=index, rehearsal=full, candidates=[])
        groups.append(group)
        for candidate in range(cfg['candidates']):
            branches.append(dict(group=group, candidate=candidate, node=copy.deepcopy(nodes[rid][index]),
                                 error='', first=None, first_output=None))
    directory.mkdir(parents=True, exist_ok=True)
    sampler = SampleBackend(model, tokenizer, cfg, directory/'sampled_first_actions.jsonl')
    outputs = sampler.generate([messages_for(b['node'], True) for b in branches])
    def journal(branch, previous, child, error, output, depth):
        append(directory/'branch_calls.jsonl', dict(reaction_id=branch['group']['reaction_id'],
            candidate=branch['candidate'], anchor_index=branch['group']['index'], depth=depth,
            input_messages=messages_for(previous, True), output=output,
            state_before=previous.state, state_after=child.state if child else previous.state,
            error=error, gt_used_for_generation_or_stopping=False))
    for branch, output in zip(branches, outputs):
        child, error = model_step(branch['node'], output)
        journal(branch, branch['node'], child, error, output, 0)
        branch.update(first=copy.deepcopy(child), first_output=output, error=error)
        if child is not None:
            branch['node'] = child
    backend = HFBackend(model, tokenizer, max_new_tokens=cfg['max_new_tokens'],
                        batch_size=cfg['generation_batch_size'], raw_path=directory/'continuation_raw.jsonl')
    for full in (False, True):
        subset = [b for b in branches if b['group']['rehearsal'] == full]
        contract_api.continue_branches(subset, cfg['max_decisions'] if full else horizon,
            lambda ns: backend.generate([messages_for(n, True) for n in ns]), model_step, journal)
    for branch in branches:
        group = branch['group']
        rid, index = group['reaction_id'], group['index']
        first, final = branch['first'], branch['node']
        finished = final.terminal and not branch['error']
        episode = dict(terminal=finished, predicted_state=final.state, initial_state=nodes[rid][0].state,
                       target=nodes[rid][0].target)
        exact = evaluation.score_endpoint(episode, source[rid])['structural_exact']
        successor = (not group['rehearsal'] and first is not None
            and first.terminal == nodes[rid][index+1].terminal
            and same_chemical_state(first.state, nodes[rid][index+1].state))
        value = dict(candidate=branch['candidate'], first_output=branch['first_output'],
            first_state=first.state if first else None, first_terminal=first.terminal if first else None,
            endpoint=final.state, endpoint_exact=exact, reference_successor=successor,
            censored=branch['censored'], error=branch['error'],
            reward=1. if exact else (.25 if successor else (-.1 if first is None else 0.)),
            positive=bool(exact or successor))
        group['candidates'].append(value)
    for group in groups:
        group.update(contract_api.assign_advantages(group['candidates'], same_chemical_state))
        # Fallback only when this sampled frontier supplies no productive support.
        group['fallback_replay'] = not group['productive'] and not group['rehearsal']
        if group['fallback_replay']:
            encoded, _ = encode_tool_decision(tokenizer, history[group['reaction_id']][group['index']], max_length=32768)
            positions = [i for i, label in enumerate(encoded['labels']) if label != -100]
            group['replay_prompt'] = encoded['input_ids'][:positions[0]]
            group['replay_response'] = encoded['input_ids'][positions[0]:positions[-1]+1]
    supported = [c for g in groups for c in g['candidates'] if c['advantage']]
    with torch.no_grad():
        for start in range(0,len(supported),cfg.get('loss_batch_size',4)):
            part = supported[start:start+cfg.get('loss_batch_size',4)]
            sequences = [(c['first_output']['prompt_token_ids'],c['first_output']['generated_token_ids']) for c in part]
            behavior = contract_api.token_logps_batch(model,sequences,tokenizer.pad_token_id)
            frozen = contract_api.token_logps_batch(reference,sequences,tokenizer.pad_token_id)
            for candidate,b,r in zip(part,behavior,frozen):
                candidate['behavior_logps'],candidate['reference_logps'] = b.cpu().tolist(),r.cpu().tolist()
    write(directory/'optimizer_inputs.json', groups)
    write(directory/'coverage.json', dict(visited_ids=ids, correction_or_rehearsal_ids=[g['reaction_id'] for g in groups],
          solved_without_correction=[rid for rid in ids if indexed[rid]['structural_exact']],
          horizon=horizon, baseline_structural_hits=sum(v['structural_exact'] for v in baseline)))
    return groups


def optimize(model, optimizer, groups, cfg, rank, world, step, root, round_index):
    import torch
    import torch.distributed as dist
    device = next(model.parameters()).device
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()
    local_ids = [g['reaction_id'] for g in groups]
    gathered = [None]*world
    dist.all_gather_object(gathered, local_ids)
    all_ids = sorted(rid for part in gathered for rid in part)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('EARHO_DUPLICATE_OPTIMIZER_GROUP')
    byid = {g['reaction_id']:g for g in groups}
    for optimization_pass in range(cfg['optimization_passes']):
        random.Random(cfg['seed']+round_index*101+optimization_pass).shuffle(all_ids)
        for begin in range(0, len(all_ids), cfg['global_batch_size']):
            batch = all_ids[begin:begin+cfg['global_batch_size']]
            optimizer.zero_grad(set_to_none=True)
            counts = torch.zeros(4, device=device)
            jobs = []
            for rid in batch:
                group = byid.get(rid)
                if group is None:
                    continue
                for candidate in group['candidates']:
                    if candidate['advantage']:
                        output = candidate['first_output']
                        jobs.append(('policy',candidate,output['prompt_token_ids'],output['generated_token_ids']))
                if group['fallback_replay']:
                    jobs.append(('replay',None,group['replay_prompt'],group['replay_response']))
            for start in range(0,len(jobs),cfg.get('loss_batch_size',4)):
                part = jobs[start:start+cfg.get('loss_batch_size',4)]
                values = contract_api.token_logps_batch(model,[(p,r) for kind,c,p,r in part])
                losses = []
                for (kind,candidate,p,r),new in zip(part,values):
                    if kind == 'policy':
                        loss,pg,kl = contract_api.policy_loss(new,
                            torch.tensor(candidate['behavior_logps'],device=device),
                            torch.tensor(candidate['reference_logps'],device=device),candidate['advantage'],
                            cfg['clip_epsilon'],cfg['kl_beta'])
                        losses.append(candidate['policy_weight']*loss)
                        counts[0] += 1
                        counts[2] += pg.detach()
                        counts[3] += kl.detach()
                    else:
                        losses.append(-cfg['replay_weight']*new.mean())
                        counts[1] += 1
                combined = torch.stack(losses).sum()
                if not torch.isfinite(combined):
                    raise ValueError('EARHO_NONFINITE_BATCHED_LOSS')
                combined.backward()
            dist.all_reduce(counts)
            if not (counts[:2] > 0).any():
                continue
            contract_api.synchronize_gradients(model, len(batch))
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['max_grad_norm'])
            if not torch.isfinite(norm):
                raise ValueError('EARHO_NONFINITE_GRADIENT')
            optimizer.step()
            step += 1
            emit(root, rank, 'optimizer-update', round=round_index, optimization_pass=optimization_pass,
                 optimizer_step=step, global_groups=len(batch), policy_records=int(counts[0]),
                 fallback_replay_records=int(counts[1]), policy_loss_sum=float(counts[2]),
                 sampled_fixed_parent_kl_sum=float(counts[3]), gradient_norm=float(norm))
    model.eval()
    model.gradient_checkpointing_disable()
    return step


def save_checkpoint(model, optimizer, root, cfg_sha, next_round, horizon, step, rank, completed_ids, cfg):
    import torch
    import torch.distributed as dist
    import numpy as np
    from mechet.pr96_eval_runtime import sha
    destination = root/'checkpoints'/f'round-{next_round:05d}'
    staging_paths = [str(contract_api.checkpoint_staging(root, next_round)) if rank == 0 else None]
    dist.broadcast_object_list(staging_paths, src=0)
    staging = Path(staging_paths[0])
    if rank == 0:
        model.save_pretrained(staging, safe_serialization=True)
        torch.save(optimizer.state_dict(), staging/'optimizer.pt')
    dist.barrier()
    torch.save(dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                    python=random.getstate(), numpy=np.random.get_state()), staging/f'rng.rank{rank:02d}.pt')
    dist.barrier()
    if rank == 0:
        state = dict(next_round=next_round, horizon=horizon, optimizer_step=step,
                     train_denominator=cfg['train_reactions'], visited_train_roots=len(completed_ids),
                     completed_train_ids_sha256=hashlib.sha256(json.dumps(completed_ids).encode()).hexdigest())
        write(staging/'trainer_state.json', state)
        files = {p.name:sha(p) for p in staging.iterdir() if p.is_file()}
        write(staging/'complete.json', dict(contract_sha256=cfg_sha, **state,
              files=files, benchmark_eligible=True,
              training_contains_originalTEST=False, algorithm='pr96_earho_v1'))
        staging.rename(destination)
        write(root/'latest_checkpoint.json', dict(path=str(destination), next_round=next_round,
                                               optimizer_step=step, horizon=horizon))
    dist.barrier()


def validation_sources(source):
    from rdkit import Chem
    public_sources, unresolved = [], []
    for original in source:
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        mol = Chem.MolFromSmiles(original['target_smiles'], parser)
        if mol is None:
            raise ValueError('EARHO_INVALID_ORIGINAL_VALIDATION_SMILES')
        fragments = list(Chem.GetMolFrags(mol, asMols=True))
        organic = [f for f in fragments if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
        ordered = sorted(organic, key=lambda f:f.GetNumHeavyAtoms(), reverse=True)
        error = ('MAIN_PRODUCT_UNRESOLVED' if not ordered else
                 'MAIN_PRODUCT_AMBIGUOUS' if len(ordered)>1 and ordered[0].GetNumHeavyAtoms()==ordered[1].GetNumHeavyAtoms() else '')
        if error:
            unresolved.append(dict(reaction_id=str(original['source_id']), terminal=False,
                                   structural_exact=False, full_exact=False, generated_decisions=0, error=error))
            continue
        # Original expected precursor is untouched. Its mapped main component
        # remains the independent structural projection used in scoring.
        public_sources.append({**original, 'target_smiles': Chem.MolToSmiles(ordered[0])})
    return public_sources, unresolved


def full_validation(model, tokenizer, cfg, root, label, rank, world):
    source = list(rows(cfg['validation_file']))
    if len(source) != cfg['validation_reactions']:
        raise ValueError('EARHO_VALIDATION_DENOMINATOR_CHANGED')
    public_sources, unresolved = validation_sources(source[rank::world])
    scored, _ = run_episodes(public_sources, model, tokenizer, cfg, root/label/f'rank{rank:02d}')
    scored.extend(unresolved)
    write(root/label/f'rank{rank:02d}'/'scored.json', scored)
    import torch.distributed as dist
    dist.barrier()
    if rank == 0:
        parts = [json.loads((root/label/f'rank{r:02d}'/'scored.json').read_text()) for r in range(world)]
        report = evaluation.aggregate_results(parts, [str(s['source_id']) for s in source])
        write(root/label/'report.json', report)
        emit(root, rank, 'full-valid-complete', label=label, **report)
    dist.barrier()


def main():
    import torch
    import torch.distributed as dist
    import numpy as np
    import yaml
    from peft import PeftModel
    from transformers import AutoTokenizer
    from mechet.qwen35_protocol import load_language_model, OFFICIAL_REVISION
    from mechet.pr96_artifacts import verify_evaluation_bindings
    from mechet.pr96_eval_runtime import sha
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.contract.read_text())
    contract_api.validate(cfg)
    cfg_sha = sha(args.contract)
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != cfg['world_size'] or torch.cuda.device_count() != world:
        raise ValueError('EARHO_ACTUAL_GPU_COUNT_MISMATCH')
    contract_api.validate_hardware([torch.cuda.get_device_name(i) for i in range(world)], cfg.get('gpu_name', 'H20'), world)
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(cfg['cpu_threads_per_rank'])
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    root = Path(cfg['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    # Bootstrap authenticated all immutable inputs once before torchrun. Avoid
    # rereading 15GB from Ceph on every GPU rank. Direct entry still verifies.
    if os.environ.get('MECHET_EARHO_VERIFIED_CONTRACT_SHA256') != cfg_sha:
        if rank == 0:
            for path, expected in cfg['files_sha256'].items():
                if sha(path) != expected:
                    raise ValueError('EARHO_FROZEN_INPUT_CHANGED:'+path)
        dist.barrier()
    parent_cfg = yaml.safe_load(Path(cfg['parent_config']).read_text())
    verify_evaluation_bindings(parent_cfg, Path(cfg['parent']))
    base = json.loads(Path(cfg['base_manifest']).read_text())
    if base['revision'] != OFFICIAL_REVISION or sha(Path(cfg['parent'])/'adapter_model.safetensors') != cfg['parent_sha256']:
        raise ValueError('EARHO_WRONG_INITIAL_ACTOR')
    local = Path(os.environ['HF_HUB_CACHE'])/'models--Qwen--Qwen3.5-0.8B'/'snapshots'/OFFICIAL_REVISION
    for name, record in base['files'].items():
        if sha(local/name) != record['sha256']:
            raise ValueError('EARHO_LOCAL_BASE_CHANGED:'+name)
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    resumed = None
    checkpoint = contract_api.discover_checkpoint(root, cfg_sha, cfg)
    if checkpoint is not None:
        resumed = contract_api.verify_checkpoint(checkpoint, cfg_sha, cfg)
    else:
        checkpoint = Path(cfg['parent'])
    actor, _ = load_language_model(local, dtype=torch.bfloat16)
    actor = PeftModel.from_pretrained(actor, checkpoint, is_trainable=True).to(local_rank)
    reference, _ = load_language_model(local, dtype=torch.bfloat16)
    reference = PeftModel.from_pretrained(reference, cfg['parent'], is_trainable=False).to(local_rank).eval()
    for module in actor.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    actor.eval()
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=cfg['learning_rate'], weight_decay=0.)
    start, horizon, step = 0, cfg['initial_horizon'], 0
    if resumed:
        optimizer.load_state_dict(torch.load(checkpoint/'optimizer.pt', map_location=f'cuda:{local_rank}', weights_only=False))
        rng = torch.load(checkpoint/f'rng.rank{rank:02d}.pt', map_location='cpu', weights_only=False)
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'])
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        start, horizon, step = resumed['next_round'], resumed['horizon'], resumed['optimizer_step']
    else:
        torch.manual_seed(cfg['seed']+rank)
        random.seed(cfg['seed']+rank)
        np.random.seed(cfg['seed']+rank)
    # Broadcast the starting trainable tensors once; later updates reduce sums
    # in fixed order, independent of each rank's number of supported branches.
    for parameter in actor.parameters():
        if parameter.requires_grad:
            dist.broadcast(parameter.data, src=0)
    index_root = root/'indexes'
    if rank == 0 and not cfg.get('indexes'):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(storage.build_index, cfg[key], index_root/f'{key}.sqlite',
                                grouped=(key == 'history_file'), expected_reactions=cfg['train_reactions'])
                    for key in ('train_file','history_file')]
            for job in jobs:
                job.result()
    dist.barrier()
    indexes = cfg.get('indexes') or {key:str(index_root/f'{key}.sqlite') for key in ('train_file','history_file')}
    source_store = storage.IndexedJsonl(cfg['train_file'], indexes['train_file'])
    history_store = storage.IndexedJsonl(cfg['history_file'], indexes['history_file'])
    ids = sorted(source_store.ids(), key=lambda rid:hashlib.sha256(f"earho:{cfg['seed']}:{rid}".encode()).hexdigest())
    if set(ids) != set(history_store.ids()):
        raise ValueError('EARHO_MISSING_HISTORY_REACTIONS')
    chunks = list(contract_api.rounds(ids, cfg['products_per_round']))
    if resumed:
        completed_ids = [rid for chunk in chunks[:start] for rid in chunk]
        if resumed['completed_train_ids_sha256'] != hashlib.sha256(json.dumps(completed_ids).encode()).hexdigest():
            raise ValueError('EARHO_RESUME_COMPLETED_IDS_CHANGED')
        if not (root/'valid_before/report.json').exists():
            raise ValueError('EARHO_RESUME_BASELINE_REPORT_MISSING')
    emit(root, rank, 'actor-ready', parent_sha256=cfg['parent_sha256'], gpu=torch.cuda.get_device_name(local_rank),
         train_roots=len(ids), world_size=world, next_round=start, total_rounds=len(chunks),
         frozen_reference_sha256=cfg['parent_sha256'], resumed=bool(resumed))
    if not (root/'valid_before'/'report.json').exists():
        full_validation(actor, tokenizer, cfg, root, 'valid_before', rank, world)
    for index in range(start, len(chunks)):
        chunk = chunks[index]
        mine_ids = chunk[rank::world]
        # Bound resident JSON and replayed states to this round, not all TRAIN.
        source = {rid:source_store.get(rid)[0] for rid in mine_ids}
        history = {rid:history_store.get(rid) for rid in mine_ids}
        nodes = {rid:replay(source[rid],history[rid]) for rid in mine_ids}
        emit(root, rank, 'round-references-ready', round=index+1, owned_roots=len(mine_ids))
        full_count = max(1, round(len(chunk)*cfg['full_episode_fraction']))
        rehearsal = set(sorted(chunk, key=lambda rid:hashlib.sha256(('rehearsal:'+rid).encode()).hexdigest())[:full_count])
        directory = root/f'round-{index:05d}'/f'rank{rank:02d}'/f'attempt-{time.time_ns()}'
        groups = collect(mine_ids, source, history, nodes, actor, reference, tokenizer, cfg, directory, horizon, rehearsal)
        step = optimize(actor, optimizer, groups, cfg, rank, world, step, root, index)
        summaries = [{k:g[k] for k in ('effective', 'productive', 'rehearsal')} for g in groups]
        parts = [None]*world
        dist.all_gather_object(parts, summaries)
        before = horizon
        horizon = contract_api.promote(horizon, [g for part in parts for g in part], cfg['maximum_horizon'],
                                       cfg['minimum_effective_groups'], cfg['productive_support_threshold'])
        completed_ids = [rid for part in chunks[:index+1] for rid in part]
        save_checkpoint(actor, optimizer, root, cfg_sha, index+1, horizon, step, rank, completed_ids, cfg)
        del source, history, nodes, groups
        visited = sum(map(len, chunks[:index+1]))
        emit(root, rank, 'round-complete', round=index+1, total_rounds=len(chunks),
             visited_train_roots=visited, train_denominator=len(ids), percent=100*visited/len(ids),
             optimizer_step=step, horizon_before=before, horizon_after=horizon,
             effective_policy_groups=sum(g['effective'] for part in parts for g in part))
    full_validation(actor, tokenizer, cfg, root, 'valid_after', rank, world)
    if rank == 0:
        final_checkpoint = contract_api.discover_checkpoint(root, cfg_sha, cfg)
        final_state = contract_api.verify_checkpoint(final_checkpoint, cfg_sha, cfg)
        if final_state['visited_train_roots'] != len(ids):
            raise ValueError('EARHO_INCOMPLETE_TRAIN_COVERAGE')
        write(root/'completion.json', dict(algorithm=cfg['algorithm'], visited_train_roots=final_state['visited_train_roots'],
            optimizer_steps=step, final_horizon=horizon, test_used=False,
            parent_sha256=cfg['parent_sha256'], contract_sha256=cfg_sha,
            validation_before=json.loads((root/'valid_before/report.json').read_text()),
            validation_after=json.loads((root/'valid_after/report.json').read_text())))
    dist.barrier()
    source_store.close()
    history_store.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
