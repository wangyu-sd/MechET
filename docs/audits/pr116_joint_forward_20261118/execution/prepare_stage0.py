import sys,os,shutil,json,hashlib,shlex
from pathlib import Path
here=Path(__file__).resolve().parent
root=Path.cwd().resolve()
old=root/'outputs/submissions/pr111_full_native_d3_resume_20261118_04'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prepare(kind,world,count):
    label=f'pr116_step0_{world}{kind.lower()}_{count}_20261118_01'
    if kind=='A100' and '--hf' in sys.argv:label=label.replace('_01','_02')
    folder=root/'outputs/submissions'/label
    folder.mkdir(exist_ok=True)
    cfg=json.loads((old/'contract.json').read_text())
    code=Path(cfg['code_root'])
    out=root/'outputs/agent'/label
    new=folder/'contract.json'; cmd=folder/'bootstrap.sh'
    # The single-card admission panel is a prefix of the identical clean baseline cohort.
    state=folder/'baseline.py';shutil.copyfile(here/'baseline.py',state)
    shutil.copyfile(here/'clean_valid.jsonl',folder/'clean_valid.jsonl')
    ds=Path(cfg['dataset'])
    required=[state,folder/'clean_valid.jsonl',Path(cfg['parent'])/'adapter_model.safetensors',ds/'validation_protected_identities.json',ds/'stock.txt',ds/'references.sqlite']
    cfg.update(world_size=world,gpu_name=kind,validation_targets=count,products_per_round=8 if world==1 else 32,
               clean_valid_file=str(folder/'clean_valid.jsonl'),stage0_input_hashes={str(f):sha(f) for f in required},output_dir=str(out),
               stage0_role='frozen policy-only clean validation, before PR116 joint optimization')
    if kind=='A100' and '--hf' in sys.argv:
        hf=json.loads((root/'outputs/submissions/pr111_d1_accelerated_resume_20261117_01/contract.json').read_text())
        cfg.update(inference_backend='hf',acceleration=hf['acceleration'],generation_batch_size=8)
        cfg['stage0_role']='single A100 engineering fallback with original BF16 HF sampler; separate results from native comparison'
    if kind=='V100':raise ValueError('BF16 native runtime requires a distinct V100 compatibility profile')
    # Independent engine per rank, same generation distribution and frozen original G4.
    cfg['validation_protocol']={**cfg['validation_protocol'],'id':'pr116_clean_stage2_g4_seed17_v1'}
    new.write_text(json.dumps(cfg,indent=2)+'\n')
    bootstrap=(code/'scripts/taiji_external_route_bootstrap.sh').read_text().split('# Warm source reads')[0]
    bootstrap=bootstrap.replace('route_code=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)',f'route_code={shlex.quote(str(code))}')
    assert f'route_code={code}' in bootstrap
    bootstrap+=f'\nexec "$route_python" -S -u -m torch.distributed.run --standalone --nproc_per_node="$route_world" --no-python "$route_python" -S -u {state} "$route_contract"\n'
    cmd.write_text(bootstrap)
    flag='meteor_mechet_'+label
    donor=old/'task_config.private.json' if kind=='H20' else Path('/root/.taiji/rendered_configs/meteor_mechet_first_use_valid16_ckpt2500_k1_1a100_qy_20260910_02.json')
    task=json.loads(donor.read_text())
    task.update(task_flag=flag,readable_name=f'MechET PR116 frozen step-0 clean validation {world}x{kind}',host_num=1,host_gpu_num=world,GPUName=kind,keep_running_after_trainer_finish=False,
                task_description=f'PR116 frozen History89442 clean validation {count} targets G4, no teacher or optimizer update; independent allocation.',start_cmd=f"bash -lc 'TAIJI_HEARTBEAT_LABEL={flag} TAIJI_HEARTBEAT_SECONDS=45 TAIJI_MIRROR_PID1_STDOUT=1 exec bash {code}/scripts/taiji_run_with_heartbeat.sh bash {cmd} {new}'")
    token=json.loads(Path('/root/.taijiconfig').read_text())['user']['token']
    task.update(Token=token,token=token)
    private=folder/'task_config.private.json'
    with os.fdopen(os.open(private,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600),'w') as f:json.dump(task,f)
    (folder/'manifest.json').write_text(json.dumps(dict(flag=flag,contract_sha256=sha(new),bootstrap_sha256=sha(cmd),stage='baseline',device=kind,world=world,targets=count),indent=2)+'\n')
    print(folder)
if __name__=='__main__':prepare(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
