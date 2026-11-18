"""Process-safe batched forward evidence. Only completed numeric observations are cached."""
import contextlib,fcntl,hashlib,json,math,os,tempfile
from pathlib import Path

def content_key(data):
    return hashlib.sha256(json.dumps(data,separators=(',',':'),sort_keys=True,allow_nan=False).encode()).hexdigest()

def score_key(protocol,query):
    return content_key([protocol,query['teacher_reactants_mapped'],query['target_product'],query['target_heavy_teacher_maps']])

def get_scores(inputs,protocol,storage,score_batch):
    storage=Path(storage)
    storage.mkdir(parents=True,exist_ok=True)
    keys=[score_key(protocol,q) for q in inputs]
    pending=dict(zip(keys,inputs))
    evidence={}
    # Stable ordering prevents lock inversion when two ranks request overlapping batches.
    with contextlib.ExitStack() as stack:
        for key in sorted(set(keys)):
            path=storage/(key+'.json')
            fd=stack.enter_context((storage/(key+'.lock')).open('a+b'))
            fcntl.flock(fd.fileno(),fcntl.LOCK_EX)
            if path.exists():
                row=json.loads(path.read_bytes())
                if row['key']!=key or row['protocol']!=content_key(protocol):
                    raise ValueError('forward cache provenance mismatch')
                evidence[key]=row
                pending.pop(key)
        if pending:
            values=score_batch(list(pending.values()))
            if len(values)!=len(pending):raise ValueError('forward batch output count mismatch')
            for key,v in zip(pending,values):
                if v is not None and (not math.isfinite(v) or not 0<=v<=1):
                    raise ValueError('invalid forward score')
                row={'key':key,'protocol':content_key(protocol),'score':v}
                evidence[key]=row
                if v is not None:
                    with tempfile.NamedTemporaryFile('w',dir=storage,delete=False) as f:
                        json.dump(row,f,allow_nan=False);f.flush();os.fsync(f.fileno())
                    os.replace(f.name,storage/(key+'.json'))
    return [evidence[k] for k in keys]
