"""Bind the original Figshare model release to a frozen, reproducible forward-sampling protocol."""
import argparse,hashlib,json,os,sys,zipfile,tarfile
from pathlib import Path

def sha(path,algo='sha256'):
    h=hashlib.new(algo)
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,record):
    temp=path.with_name(path.name+'.building')
    temp.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    os.replace(temp,path)

def bind(archive):
    here=Path(__file__).resolve().parent
    release=here.parents[2]/'artifacts/pr116_joint_20261118'
    dest=release/'teacher';source=dest/'official';runtime=dest/'python'
    if archive.stat().st_size!=444244997 or sha(archive,'md5')!='2ab43d1956aafc12ead0b707b2b3c9db':
        raise ValueError('expected the original Figshare checkpoints.zip, file 55904912')
    if sha(dest/'source.tgz')!='b5a0a682804422bbada2c16a411df6176197c75ebddb4d122540690918b0fd9c':
        raise ValueError('unexpected source release')
    hashes={}
    with tarfile.open(dest/'source.tgz') as tar:
        for member in tar:
            path=Path(member.name)
            if not member.isfile() or path.suffix!='.py':continue
            name=str(Path(*path.parts[1:]))
            expected=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            if sha(source/name)!=expected:raise ValueError('changed teacher code '+name)
            hashes[name]=expected
    name='checkpoints/flower_new_dataset/best_large_hyperparam/model.2880000_95.pt'
    weight=dest/'model.2880000_95.pt'
    with zipfile.ZipFile(archive) as z:
        entries=[n for n in z.namelist() if n.lstrip('./').endswith(name) and '__MACOSX' not in n]
        if len(entries)!=1:raise ValueError('original release member ambiguous')
        import shutil
        with z.open(entries[0]) as src, weight.with_suffix('.building').open('wb') as dst:shutil.copyfileobj(src,dst)
        weight.with_suffix('.building').replace(weight)
    work=dest/'work';work.mkdir(exist_ok=True)
    score_script=release/'code/scripts/pr116_flow_teacher.py'
    cfg=dict(official_code=str(source),official_files=hashes,checkpoint=str(weight),checkpoint_sha256=sha(weight),
             work_dir=str(work),samples=4,max_elementary_steps=15,token_batch=4096,seed=17)
    config=dest/'scoring.json';write(config,cfg)
    protocol=dict(upstream='00b93aa7f46c266a94d249066edfb1ee833cb1bc',figshare_file=55904912,
        config_sha256=sha(config),script_sha256=sha(score_script),weights_sha256=cfg['checkpoint_sha256'],
        samples=4,max_elementary_steps=15,seed=17,normalization='full_mapped_dense_H_project_stereo_mask_isotopes_v2',
        recovery='terminal_self_loop_product_component_heavy_maps_charge_preserved')
    write(dest/'bound_runtime.json',dict(teacher_protocol=protocol,teacher_python='/root/miniconda3/envs/meteor/bin/python',
          teacher_local_rank=3,teacher_config=str(config),teacher_script=str(score_script),
          teacher_pythonpath=os.pathsep.join([str(runtime),str(source)]),work_dir=str(work),
          cache_dir=str(dest/'scores'),timeout_seconds=1800))
    print(weight,cfg['checkpoint_sha256'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('official_zip',type=Path)
    bind(p.parse_args().official_zip)
