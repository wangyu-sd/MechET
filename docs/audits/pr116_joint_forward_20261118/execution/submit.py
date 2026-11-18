"""Register, launch and read back an independent immutable task; never restart another instance."""
import json,sys,subprocess,datetime,os
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
folder=Path(sys.argv[1]).resolve()
manifest=json.loads((folder/'manifest.json').read_text())
flag=manifest['flag']; spec=spec_from_file_location('mon','outputs/submissions/pr111_full_native_d3_resume_20261118_04/monitor_qwen3_task.py');mon=module_from_spec(spec);spec.loader.exec_module(mon)
config=folder/'task_config.private.json'
client=str(Path('artifacts/taiji_mount_bootstrap/taiji_client').resolve())
def call(op,*args):
    result=subprocess.run([client,op,*map(str,args)],capture_output=True,timeout=55)
    data=(result.stdout+result.stderr).decode(errors='replace')
    path=folder/f'{op}.private.log'
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as f:f.write(data)
    if result.returncode or '[error]' in data.lower():raise RuntimeError(op+' failed; see private receipt')
    return data
import hashlib
def register():
    record=folder/'registered.json'
    digest=hashlib.sha256((folder/'contract.json').read_bytes()).hexdigest()
    if digest!=manifest['contract_sha256']:raise RuntimeError('frozen contract changed')
    if record.exists():
        if json.loads(record.read_bytes())!=dict(flag=flag,contract_sha256=digest):raise RuntimeError('registration changed')
        return
    call('create','--type','task','--scfg',config)
    detail=call('task_detail',flag)
    if str(folder/'contract.json') not in detail:raise ValueError('remote registration path mismatch')
    record.write_text(json.dumps(dict(flag=flag,contract_sha256=digest))+'\n')
if sys.argv[2] in ('register','start'):
    register()
    if sys.argv[2]=='register':
        print((folder/'registered.json').read_text())
        sys.exit(0)
    if (folder/'start.accepted').exists():raise RuntimeError('start already accepted, use observe')
    call('start','--scfg',config,'--task_flag',flag)
    (folder/'start.accepted').write_text(datetime.datetime.now().astimezone().isoformat())
listing=call('instance_list',flag)
try:r=mon.parse_instances(listing)
except ValueError as exc:
    if str(exc)!='NO_PLATFORM_INSTANCE_ROW' or not (folder/'registered.json').exists():raise
    r=dict(instance=None,platform_state='REGISTERED_WITHOUT_INSTANCE',success=None)
r.update(task_flag=flag,observed_at=datetime.datetime.now().astimezone().isoformat())
(folder/'platform_receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(r)
