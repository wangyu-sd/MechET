"""Freeze one independent 64-update PR116 Taiji arm, without restarting previous jobs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,sys,tarfile
from pathlib import Path

base=Path(__file__).resolve().parent
root=base.parents[2]
sub=root/'outputs/submissions'
old=sub/'pr111_full_native_d3_resume_20261118_04'
stage0=sub/'pr116_step0_4h20_512_20261118_01'
release=root/'artifacts/pr116_joint_20261118'
code=release/'code'
def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as src:
        for b in iter(lambda:src.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def profile_label(mode):
    return f'pr116_teacher_profile_{mode}_4h20_20261119_01'


def freeze(arm,profile=None):
    if profile is not None and (arm not in ('JR', 'J', 'J-profile') or profile not in ('time_slice', 'colocated')):
        raise ValueError('select teacher profile only for a teacher-enabled arm')
    if arm not in ('R', 'JR', 'J', 'J-distill', 'J-profile'):
        raise ValueError('unknown arm')
    if arm == 'J-profile' and profile is None:
        raise ValueError('the profile trial must declare its teacher placement')
    profile_trial = arm == 'J-profile'
    distill = arm == 'J-distill'
    with_teacher = arm in ('JR', 'J', 'J-profile')
    profile_files = {}
    if with_teacher:
        teacher=json.loads((release/'teacher/bound_runtime.json').read_bytes())
        admission=json.loads((release/'teacher/forward_execution.json').read_bytes())
        if admission['protocol']!=teacher['teacher_protocol'] or not admission['scores']:
            raise ValueError('no actual forward run bound to this checkpoint and protocol')
        if sha(teacher['teacher_config'])!=teacher['teacher_protocol']['config_sha256']:
            raise ValueError('teacher runtime mismatch')
        opts=json.loads(Path(teacher['teacher_config']).read_bytes())
        if sha(opts['checkpoint'])!=opts['checkpoint_sha256']:
            raise ValueError('official weight mismatch')
        sys.path.insert(0, str(release/'code/src'))
        from mechet.route_joint_profile import MODES, bind_profile_source, choose_profile
        replay = bind_profile_source(sub/'pr116_r_64x32_4h20_20261118_01/contract.json',
            base/'full_first_train_batch_teacher_inputs/manifest.json')
        if profile_trial:
            profile_files.update(replay['group_files_sha256'])
            profile_files[replay['manifest']] = replay['manifest_sha256']
            profile_files[replay['optimizer_seed_path']] = replay['optimizer_seed_sha256']
        else:
            reports = []
            for mode in MODES:
                label = profile_label(mode)
                profile_contract = sub/label/'contract.json'
                profile_receipt = root/'outputs/agent'/label/'teacher_profile_receipt.json'
                measured = json.loads(profile_receipt.read_bytes())
                profile_files[str(profile_contract)] = sha(profile_contract)
                profile_files[str(profile_receipt)] = sha(profile_receipt)
                if measured['contract_sha256'] != profile_files[str(profile_contract)]:
                    raise ValueError('teacher placement receipt does not bind its Taiji contract')
                reports.append(measured)
            chosen = choose_profile(reports, teacher['teacher_protocol'], replay['manifest_sha256'],
                                    replay['expected_teacher_queries'])
            if profile is not None and profile != chosen:
                raise ValueError('formal arms must share the selected measured teacher placement')
            profile = chosen
    if distill:
        sys.path.insert(0, str(release/'code'))
        sys.path.insert(0, str(release/'code/src'))
        from mechet.route_joint_distillation import bind_distillation_source
        source = bind_distillation_source(sub/'pr116_j_64x32_4h20_20261118_01/contract.json')
    cfg=json.loads((old/'contract.json').read_bytes())
    arm_slug = arm.lower().replace('-', '_')
    label=profile_label(profile) if profile_trial else f'pr116_{arm_slug}_64x32_4h20_20261118_01'
    folder=sub/label;folder.mkdir(exist_ok=True)
    global code
    import shutil
    if (folder/'start.accepted').exists() or (folder/'registered.json').exists():
        raise ValueError('already registered: immutable')
    code=folder/'code'
    shutil.copytree(release/'code',code,dirs_exist_ok=True)
    archive=folder/'entry.tar.gz'
    with tarfile.open(archive.with_suffix('.building'),'w:gz') as out:
        for name in ('src','scripts','configs','tests'):out.add(code/name,arcname=name)
    archive.with_suffix('.building').replace(archive)
    parent_code=Path(cfg['code_root'])
    files=cfg['files_sha256']
    # Keep pinned dataset, student, tokenizer, and runtime. The previous optimizer,
    # old code snapshot and continuation diagnostics do not seed this fresh actor.
    removal=['/outputs/agent/pr109_standard_d3_refit','/outputs/agent/pr111_d3_accelerated',
             '/outputs/diagnostics/pr111_', '/outputs/submissions/pr111_', '/outputs/submissions/pr109_standard_d3']
    cfg['files_sha256']={k:v for k,v in files.items() if not Path(k).is_relative_to(parent_code.parent)
                         and not any(p in k for p in removal)}
    source=sorted(p for name in ('src/mechet','scripts') for p in (code/name).rglob('*.py'))
    with ThreadPoolExecutor(max_workers=12) as ex:
        cfg['code_files_sha256']=dict(zip([str(p.relative_to(code)) for p in source],ex.map(sha,source)))
    for k in ('acceleration_continuation','initial_baseline'):cfg.pop(k,None)
    cap=(.1 if with_teacher else 0.);lam=(.1 if arm in ('J','J-profile') else 0.)
    cfg['joint_forward']=dict(lambda_f=cap,lambda_d=lam,beta_d=.1,margin=.25)
    if with_teacher:
        snapshot=code/'scripts/pr116_flow_teacher.py'
        if sha(snapshot)!=teacher['teacher_protocol']['script_sha256']:raise ValueError('teacher adapter changed since admission')
        cfg['joint_forward'].update(teacher,teacher_script=str(snapshot),profile=profile)
        cfg['files_sha256'].update(profile_files)
        cfg['files_sha256'][teacher['teacher_config']]=teacher['teacher_protocol']['config_sha256']
    cfg.update(step_budget=dict(arm=arm,roots=str(base/'root_stream.jsonl'),valid=str(base/'clean_valid.jsonl'),
                    baseline_contract=str(stage0/'contract.json'),baseline_sha256=sha(stage0/'contract.json')),
        experimental_condition='PR116_'+arm,authorization='PR116 fresh History-89442, four independent H20',
        implementation='fixed 64x32 ReFIT, optional frozen forward evidence and joint DPO',
        products_per_round=32,validation_targets=512,epochs=1,entry_archive=str(archive),
        entry_archive_sha256=sha(archive),code_root=str(code),output_dir=str(root/'outputs/agent'/label),
        native_admission=dict(required=True,mode='fresh_stage2_native_baseline'))
    if profile_trial:
        cfg['profile_replay'] = replay
        cfg['implementation'] = 'PR116 teacher device measurement on the original complete TRAIN batch'
    if distill:
        cfg['distill_source'] = source
        cfg['files_sha256'][source['source_contract']] = source['source_contract_sha256']
        cfg['implementation'] = 'frozen Joint actor ReFIT search and native weighted sequence CE'
        cfg['authorization'] = 'PR116 final TRAIN-only search-to-policy distillation, four H20'
    cfg['validation_protocol']['id']='pr116_clean_stage2_g4_seed17_v1'
    cfg['acceleration']['refit_lanes']=dict(credit_forks_per_tree=2,exploration_forks_per_tree=0)
    cfg['refit']['max_forks_per_tree']=2
    for f in (old/'contract.json',archive,stage0/'contract.json',base/'root_stream.jsonl',base/'clean_valid.jsonl',base/'cohort.json'):
        cfg['files_sha256'][str(f)]=sha(f)
    conf=folder/'contract.json';write(conf,cfg)
    job=json.loads((old/'task_config.private.json').read_bytes())
    flag='meteor_mechet_'+label
    rounds = '64 TRAIN batches' if distill else '64 updates'
    if profile_trial: rounds = f'one frozen {profile} TRAIN profile'
    job.update(task_flag=flag,readable_name=f'MechET PR116 {arm} {rounds} 32 roots 4 H20',
               task_description=f'Fixed TRAIN root stream. Four GPUs, arm {arm}.',
               host_num=1,host_gpu_num=4,keep_running_after_trainer_finish=False,
               start_cmd=f"bash -lc 'TAIJI_HEARTBEAT_LABEL={flag} TAIJI_HEARTBEAT_SECONDS=45 TAIJI_MIRROR_PID1_STDOUT=1 exec bash {code}/scripts/taiji_run_with_heartbeat.sh bash {code}/scripts/taiji_external_route_bootstrap.sh {conf}'")
    token=json.loads(Path('/root/.taijiconfig').read_bytes())['user']['token']
    job.update(Token=token,token=token)
    with os.fdopen(os.open(folder/'task_config.private.json',os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600),'w') as f:json.dump(job,f)
    write(folder/'manifest.json',dict(flag=flag,stage=arm,contract_sha256=sha(conf),device='H20',world=4,targets=32 if profile_trial else 2048))
    return folder
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',choices=('R','JR','J','J-distill','J-profile'))
    parser.add_argument('--profile',choices=('colocated','time_slice'))
    args=parser.parse_args()
    print(freeze(args.arm,args.profile))
