"""Start the registered R arm after the successful four-H20 step-0 allocation releases its GPUs."""
import fcntl,json,subprocess,sys,time,hashlib,datetime
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[2]
sub=root/'outputs/submissions'
stage=sub/'pr116_step0_4h20_512_20261118_01'
arm=sub/'pr116_r_64x32_4h20_20261118_01'
conf=json.loads((arm/'contract.json').read_bytes())
parent=json.loads((stage/'contract.json').read_bytes())
def call(folder,mode):
    return subprocess.run([sys.executable,'-B',str(here/'submit.py'),str(folder),mode],
                          cwd=root,capture_output=True,text=True,timeout=100,check=True)
def status(message):
    r={'time':datetime.datetime.now().astimezone().isoformat(),'status':message}
    tmp=arm/'chain_status.building'
    tmp.write_text(json.dumps(r));tmp.replace(arm/'chain_status.json')
with (arm/'chain.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bound=hashlib.sha256((arm/'contract.json').read_bytes()).hexdigest()
    if json.loads((arm/'preflight.json').read_bytes())['contract_sha256']!=bound:raise ValueError('preflight changed')
    while not (arm/'start.accepted').exists():
        summary=Path(parent['output_dir'])/'validation/checkpoint-000000/summary.json'
        if not summary.is_file():
            status('waiting for all 512 initial target outcomes')
            time.sleep(45);continue
        try:
            report=json.loads(summary.read_bytes())
            if not report.get('complete') or report['denominator']!=512:raise ValueError('baseline not complete')
            call(stage,'observe')
            state=json.loads((stage/'platform_receipt.json').read_bytes())
            if state['platform_state']=='END' and state['success'] is True:
                # Check platform first to avoid duplicate launch after a lost acknowledgement.
                seen=call(arm,'observe')
                previous=json.loads((arm/'platform_receipt.json').read_bytes())
                if previous.get('instance'):
                    status('existing R instance '+previous['instance']);break
                call(arm,'start');status('R start accepted');break
            status('waiting for baseline GPU release')
        except (OSError,subprocess.SubprocessError) as exc:
            status('retrying platform readback: '+type(exc).__name__)
        time.sleep(45)
