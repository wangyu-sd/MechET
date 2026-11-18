"""Run the bound authors' model on accepted TRAIN reactions and record coverage and time."""
import argparse,hashlib,json,math,os,subprocess,tempfile,time
from pathlib import Path

def file_sha(p):
    d=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):d.update(part)
    return d.hexdigest()

def score_counts(scores, expected, *, allow_unavailable=False):
    if not isinstance(scores, list) or expected < 1 or len(scores) != expected:
        raise ValueError('forward score count does not match the frozen inputs')
    observed = [s for s in scores if s is not None]
    if (not observed or (not allow_unavailable and len(observed) != expected)
            or any(type(s) not in (int, float) or not math.isfinite(s) or not 0 <= s <= 1 for s in observed)):
        raise ValueError('forward runtime did not produce the required observed scores')
    return dict(observed_scores=len(observed), unavailable_scores=expected-len(observed))


def verify(config,sample,device,batch_manifest=None):
    opts=json.loads(config.read_bytes()); protocol=opts['teacher_protocol']
    scorer=Path(opts['teacher_script']); config_file=Path(opts['teacher_config']); wd=Path(opts['work_dir'])
    if protocol['script_sha256']!=file_sha(scorer) or protocol['config_sha256']!=file_sha(config_file):
        raise ValueError('teacher binding changed')
    inputs = json.loads(sample.read_bytes())
    input_digest = file_sha(sample)
    if batch_manifest is not None:
        batch = json.loads(batch_manifest.read_bytes())
        if (batch['source_round'] != 0 or batch['train_targets'] != 32
                or batch['distinct_teacher_queries'] != len(inputs) or batch['queries_sha256'] != input_digest):
            raise ValueError('profile inputs do not match the frozen 32-target TRAIN batch')
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
    counts=score_counts(scores,len(inputs),allow_unavailable=batch_manifest is not None)
    report=dict(device=gpu,protocol=protocol,real_input_sha256=input_digest,scores=scores,
                elapsed_seconds=time.monotonic()-start,**counts)
    filename='forward_execution.json'
    if batch_manifest is not None:
        report.update(train_batch_manifest_sha256=file_sha(batch_manifest),train_targets=32,
                      scope='single-device teacher execution on the complete frozen TRAIN input batch')
        filename='forward_full_batch_execution.json'
    out=config.parent/filename
    tmp=out.with_suffix('.building');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
    print(out)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bound_config',type=Path);p.add_argument('real_input',type=Path);p.add_argument('--device',required=True)
    p.add_argument('--full-train-batch-manifest',type=Path)
    a=p.parse_args();verify(a.bound_config,a.real_input,a.device,a.full_train_batch_manifest)
