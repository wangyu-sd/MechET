"""Freeze nested target-disjoint cohorts and independent seed-17 root order for all arms."""
import hashlib, json, sqlite3
from collections import defaultdict
from pathlib import Path
out=Path(__file__).resolve().parent
cfg=json.loads(Path('outputs/submissions/pr111_full_native_d3_resume_20261118_04/contract.json').read_bytes())
d=Path(cfg['dataset'])
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def order(*parts):
    return hashlib.sha256(json.dumps([17,*parts],separators=(',',':')).encode()).hexdigest()
def inspect_tree(node,stock):
    _,children=node
    if not children:return False,node[0] in stock
    kids=[inspect_tree(c,stock) for c in children]
    return sum(bool(c[1]) for c in children)>1 or any(k[0] for k in kids),all(k[1] for k in kids)
for name in ['stock.txt','manifest.json','references.sqlite','training.jsonl','validation.jsonl']:
    assert sha(d/name)==cfg['files_sha256'][str(d/name)], name
stock=set((d/'stock.txt').read_text().splitlines())
src=sqlite3.connect('file:'+str(d/'references.sqlite')+'?mode=ro',uri=True)
by={}; public={}
for split in ('training','validation'):
    rows={}
    for tid,canonical,obs in src.execute('select id,target,public_target from tasks where split=?',(split,)):
        if canonical not in rows or order(tid)<rows[canonical][0]:rows[canonical]=order(tid),tid
        public[tid]={'task_id':tid,'target':obs}
    by[split]={m:t for m,(_,t) in rows.items()}
valid=sorted(by['validation'].keys()-by['training'].keys(),key=lambda m:order('valid',m))[:512]
assert len(valid)==512
strata=defaultdict(list)
chosen=set(by['training'].values())
for tid,tree,n,depth,provenance in src.execute("select r.task_id,r.tree,r.reaction_count,r.depth,s.provenance from routes r join route_sources s on r.task_id=s.task_id where s.split='training'"):
    if tid not in chosen:continue
    convergent,reachable=inspect_tree(json.loads(tree),stock)
    patents=json.loads(provenance)['route']['annotations'].get('source_patent_ids',[])
    key=(min(n,6),min(depth,5),convergent,reachable)
    strata[key].append((order('source', sorted(patents)),order('root',tid),tid))
# Proportional stratification at every prefix, with deterministic source-aware ties.
stream=[]
for key,values in strata.items():
    for i,(_,_,tid) in enumerate(sorted(values)):
        stream.append(((i+.5)/len(values),order('interleave',tid),tid))
train=[public[tid] for _,_,tid in sorted(stream)[:8192]]
valid=[public[by['validation'][m]] for m in valid]
def save(name,rows):
    path=out/name
    text=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows)
    if path.exists() and path.read_text()!=text:raise ValueError(name)
    path.write_text(text);return sha(path)
manifest={'source_manifest':sha(d/'manifest.json'),'source_train_rows':cfg['train_targets'],'train_unique':len(chosen),'valid_all_unique':len(by['validation']),'canonical_train_valid_overlap_source':len(by['training'].keys()&by['validation'].keys()),'clean_valid_excludes_all_train_targets':True,'strata':{str(k):len(v) for k,v in sorted(strata.items())},'pool':len(train),'main_steps':64,'roots_per_step':32,'validation_nested':[256,512],'train_sha256':save('root_stream.jsonl',train),'valid_sha256':save('clean_valid.jsonl',valid)}
(out/'cohort.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({k:v for k,v in manifest.items() if k!='strata'},flush=True)
