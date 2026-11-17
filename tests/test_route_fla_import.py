import json
import os
from pathlib import Path
import subprocess
import sys


def test_optional_model_registry_is_skipped_while_real_core_imports_remain(tmp_path):
    root=tmp_path/'packages';fla=root/'fla';fla.mkdir(parents=True)
    (fla/'__init__.py').write_text('''import importlib
__version__='0.5.2'
def _import_optional_public_module(name):
 try:return importlib.import_module(name)
 except ModuleNotFoundError as e:
  if e.name==name:return None
  raise
_layers=_import_optional_public_module('fla.layers')
_models=_import_optional_public_module('fla.models')
''')
    for name in ['layers','models']:
        p=fla/name;p.mkdir();(p/'__init__.py').write_text('VALUE=99\n')
    p=fla/'ops';p.mkdir();(p/'__init__.py').write_text('def original_op(x):return x*x\n')
    code=Path(__file__).resolve().parents[1]
    script='import runpy,os\nimport_core_fla=runpy.run_path('+repr(str(code/'src/mechet/route_fla_import.py'))+")[\"import_core_fla\"]\n"+'''import sys,json,importlib
module=importlib.import_module('fla') if os.environ.get('FLA_IMPORT_BASELINE_CONTROL') else import_core_fla()
before=[n for n in sys.modules if n in ('fla.layers','fla.models')]
from fla.ops import original_op
later=importlib.import_module('fla.models')
print(json.dumps(dict(version=module.__version__,before=before,value=original_op(7),later=later.VALUE)))
'''
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root),str(code/'src')]),PYTHONDONTWRITEBYTECODE='1')
    result=json.loads(subprocess.check_output([sys.executable,'-S','-c',script],env=env,text=True))
    assert result==dict(version='0.5.2',before=[],value=49,later=99)


def test_failed_core_import_always_removes_temporary_blocker(tmp_path):
    # A broken core package must raise; it cannot turn into a silent fallback.
    root=tmp_path/'packages';fla=root/'fla';fla.mkdir(parents=True)
    (fla/'__init__.py').write_text("raise RuntimeError('broken core')\n")
    code=Path(__file__).resolve().parents[1]
    script='import runpy\nimport_core_fla=runpy.run_path('+repr(str(code/'src/mechet/route_fla_import.py'))+")[\"import_core_fla\"]\n"+'''import sys,json
before=list(sys.meta_path)
try:import_core_fla()
except RuntimeError:pass
else:raise AssertionError('hidden core failure')
assert sys.meta_path==before
print('PASS')
'''
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root),str(code/'src')]),PYTHONDONTWRITEBYTECODE='1')
    assert subprocess.check_output([sys.executable,'-S','-c',script],env=env,text=True).strip()=='PASS'
