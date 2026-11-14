import json
import os
from pathlib import Path
import subprocess
import sysconfig


def test_torchrun_workers_keep_site_disabled(tmp_path):
    worker = tmp_path/'worker.py'
    worker.write_text("import json,os,sys,pathlib;pathlib.Path(os.environ['AUDIT_DIR'],os.environ['RANK']+'.json').write_text(json.dumps({'no_site':sys.flags.no_site}))")
    wrapper = Path(__file__).resolve().parents[1]/'scripts/qwen3_runtime_bin/python'
    env = dict(os.environ, PYTHONPATH=sysconfig.get_path('purelib'), AUDIT_DIR=str(tmp_path),
               CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1')
    env.pop('PYTHON_EXEC', None)
    result = subprocess.run([str(wrapper), '-m', 'torch.distributed.run', '--standalone',
                             '--nproc_per_node=2', str(worker)], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-2000:]
    assert [json.loads((tmp_path/f'{rank}.json').read_text())['no_site'] for rank in range(2)] == [1, 1]
