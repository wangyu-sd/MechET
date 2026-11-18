"""Place forward-model queries on the declared four-device allocation."""
import atexit
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def release_for_teacher(cfg, profile, has_new_queries):
    if profile not in ('time_slice','colocated'):
        raise ValueError('unknown forward device profile')
    if profile=='time_slice' and has_new_queries:
        process=cfg.pop('_vllm_process',None)
        if process is not None:
            process.close()
            atexit.unregister(process.close)
            return True
    return False


def sample_on_devices(rows,devices,score):
    if not devices or len(set(devices))!=len(devices):
        raise ValueError('distinct teacher GPUs required')
    assignments={d:[] for d in devices}
    loads=dict.fromkeys(devices,0)
    for index in sorted(range(len(rows)),key=lambda i:-len(rows[i]['teacher_reactants_mapped'])):
        device=min(devices,key=lambda d:loads[d])
        assignments[device].append(index)
        loads[device]+=max(1,len(rows[index]['teacher_reactants_mapped']))
    def execute(device,indices):
        started=time.monotonic()
        result=score(device,[rows[i] for i in indices])
        if len(result)!=len(indices):raise ValueError('forward device returned wrong sample count')
        return indices,result,time.monotonic()-started
    answers=[None]*len(rows);seconds={}
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        tasks={d:pool.submit(execute,d,indices) for d,indices in assignments.items() if indices}
        for d,task in tasks.items():
            indices,result,seconds[d]=task.result()
            for index,score_value in zip(indices,result):answers[index]=score_value
    return answers,seconds


def run_teacher_device(opts,device,rows):
    with tempfile.TemporaryDirectory(dir=opts['work_dir']) as folder:
        wd=Path(folder)
        (wd/'input').write_text(json.dumps(rows))
        env=os.environ.copy()
        visible=env.get('CUDA_VISIBLE_DEVICES')
        env['CUDA_VISIBLE_DEVICES']=visible.split(',')[device] if visible else str(device)
        for key in ('LOCAL_RANK','RANK','WORLD_SIZE'):env.pop(key,None)
        env['PYTHONPATH']=opts['teacher_pythonpath']
        subprocess.run([opts['teacher_python'],opts['teacher_script'],opts['teacher_config'],str(wd/'input'),
                        str(wd/'output')],env=env,check=True,timeout=opts['timeout_seconds'])
        return json.loads((wd/'output').read_bytes())
