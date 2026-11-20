"""PR116 common step-0, independent G4 clean validation. No teacher or learner update."""
import os,sys,hashlib,json
from pathlib import Path
from datetime import timedelta

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else stream_sha(f)
def stream_sha(f):
    h=hashlib.sha256()
    for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def main(path):
    cfg=json.loads(Path(path).read_text())
    for p,h in cfg['stage0_input_hashes'].items():
        if sha(p)!=h:raise ValueError('STAGE0_INPUT_CHANGED '+p)
    cfg['_sha']=sha(path)
    rank=int(os.environ['LOCAL_RANK'])
    root=Path(cfg['output_dir']);root.mkdir(parents=True,exist_ok=True)
    from mechet.route_runtime_efficiency import configure_profile
    configure_profile(cfg,Path(cfg['code_root']))
    from mechet.route_transformers_import import import_transformers_indexed
    import_transformers_indexed(cfg)
    import torch,torch.distributed as dist
    from transformers import AutoTokenizer
    from peft import PeftModel
    from mechet.qwen35_protocol import load_language_model,OFFICIAL_REVISION
    from mechet.route_precision import configure_precision
    from mechet.route_training_contract import atomic_json
    from scripts.train_external_route_rl import full_validation
    torch.cuda.set_device(rank)
    if torch.cuda.device_count()!=cfg['world_size'] or cfg['gpu_name'] not in torch.cuda.get_device_name(rank):raise ValueError('DEVICE_CONTRACT_CHANGED')
    dist.init_process_group('nccl',timeout=timedelta(seconds=cfg['collective_timeout_seconds']))
    cfg['_effective_policy_precision']=configure_precision(cfg)
    tokenizer=AutoTokenizer.from_pretrained(cfg['base_path'],revision=OFFICIAL_REVISION,local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    cfg['pad_token_id']=tokenizer.pad_token_id
    actor,_=load_language_model(cfg['base_path'],dtype=torch.bfloat16)
    actor=PeftModel.from_pretrained(actor,cfg['parent'],is_trainable=False).to(rank).eval()
    cfg['_rollout_adapter_path']=cfg['parent']
    if cfg['inference_backend']=='vllm':
        from mechet.route_vllm_backend import start_vllm_process
        start_vllm_process(cfg)
    atomic_json(root/f'actor_ready_rank{rank}.json',dict(rank=rank,device=torch.cuda.get_device_name(rank),actor=cfg['parent_sha256'],contract=cfg['_sha']))
    ds=Path(cfg['dataset'])
    rows=[json.loads(l) for l in Path(cfg['clean_valid_file']).read_text().splitlines()][:cfg['validation_targets']]
    try:
        full_validation(rows,set((ds/'stock.txt').read_text().splitlines()),json.loads((ds/'validation_protected_identities.json').read_text()),actor,tokenizer,cfg,root,0,cfg['parent_sha256'],[0.0])
    finally:
        if '_vllm_process' in cfg:cfg['_vllm_process'].close()
        dist.destroy_process_group()

if __name__=='__main__':main(sys.argv[1])
