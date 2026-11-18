"""Run the bound authors' model on an actual accepted reaction; record the device and score."""
import argparse,hashlib,json,math,os,subprocess,tempfile,time
from pathlib import Path

def file_sha(p):
    d=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):d.update(part)
    return d.hexdigest()

def verify(config,sample,device):
    opts=json.loads(config.read_bytes()); protocol=opts['teacher_protocol']
    scorer=Path(opts['teacher_script']); config_file=Path(opts['teacher_config']); wd=Path(opts['work_dir'])
    if protocol['script_sha256']!=file_sha(scorer) or protocol['config_sha256']!=file_sha(config_file):
        raise ValueError('teacher binding changed')
    env=dict(os.environ,PYTHONPATH=opts['teacher_pythonpath'],CUDA_VISIBLE_DEVICES=device,PYTHONDONTWRITEBYTECODE='1')
    gpu=json.loads(subprocess.check_output([opts['teacher_python'],'-B','-c',
        "import json,torch; print(json.dumps({'devices':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(0)}))"],env=env))
    if gpu['devices']!=1:raise ValueError('a single visible test device must be selected')
    start=time.monotonic()
    with tempfile.TemporaryDirectory(dir=wd) as temp:
        answer=Path(temp)/'observed.json'
        subprocess.run([opts['teacher_python'],'-B',str(scorer),str(config_file),str(sample),str(answer)],env=env,
                       check=True,timeout=opts['timeout_seconds'])
        scores=json.loads(answer.read_bytes())
    if (len(scores)!=len(json.loads(sample.read_bytes())) or not scores
            or any(type(s) not in (int,float) or not math.isfinite(s) or not 0<=s<=1 for s in scores)):
        raise ValueError('required an observed forward model score for each admission input')
    report=dict(device=gpu,protocol=protocol,real_input_sha256=file_sha(sample),scores=scores,
                elapsed_seconds=time.monotonic()-start)
    out=config.parent/'forward_execution.json'
    tmp=out.with_suffix('.building');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
    print(out)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bound_config',type=Path);p.add_argument('real_input',type=Path);p.add_argument('--device',required=True)
    a=p.parse_args();verify(a.bound_config,a.real_input,a.device)
