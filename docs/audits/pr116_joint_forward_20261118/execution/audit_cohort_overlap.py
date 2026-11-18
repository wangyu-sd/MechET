"""Audit exact product, reaction, route and source patent overlap without changing any split."""
import hashlib,json,sqlite3
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda:src.read(4<<20),b''):digest.update(block)
    return digest.hexdigest()

def run(directory,contract):
    config=json.loads(contract.read_bytes());frozen=config['step_budget']
    database=Path(config['dataset'])/'references.sqlite'
    expected=config['files_sha256']
    files=[Path(frozen['roots']),Path(frozen['valid']),database]
    if any(sha(p)!=expected[str(p)] for p in files):raise ValueError('changed experiment inputs')
    train=[json.loads(x)['task_id'] for x in files[0].read_text().splitlines()][:2048]
    valid=[json.loads(x)['task_id'] for x in files[1].read_text().splitlines()]
    if len(set(train))!=2048 or len(set(valid))!=512 or set(train)&set(valid):raise ValueError('unexpected cohorts')
    db=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TEMP TABLE picked(id TEXT PRIMARY KEY, split TEXT)')
    db.executemany('INSERT INTO picked VALUES (?,?)',[(v,'training') for v in train]+[(v,'validation') for v in valid])
    rows={}
    for tid,split,product,route,meta in db.execute(
        'SELECT p.id,p.split,t.target,r.route_signature,r.provenance FROM picked p '
        'JOIN tasks t ON t.id=p.id AND t.split=p.split JOIN route_sources r ON r.task_id=p.id AND r.split=p.split'):
        patents=json.loads(meta)['route']['annotations'].get('source_patent_ids',[])
        rows[tid]=dict(product={product},route={route},patent=set(patents),reaction=set())
    for tid,sig in db.execute('SELECT p.id,e.signature FROM picked p JOIN edges e ON p.id=e.task_id'):
        rows[tid]['reaction'].add(sig)
    db.close()
    if len(rows)!=2560 or any(not r['reaction'] for r in rows.values()):raise ValueError('missing reference rows')
    def compare(ids):
        result={};retained={}
        for k in ('product','reaction','route','patent'):
            tr=set().union(*(rows[i][k] for i in train))
            va=set().union(*(rows[i][k] for i in ids))
            retained[k]=[i for i in ids if rows[i][k] and rows[i][k].isdisjoint(tr)]
            result[k]=dict(distinct_train=len(tr),distinct_valid=len(va),shared=len(tr&va),
                affected_valid_targets=sum(bool(rows[i][k]&tr) for i in ids),
                unknown_valid_targets=sum(not rows[i][k] for i in ids))
        certified=sorted(set(retained['reaction'])&set(retained['patent']))
        return dict(denominator=len(ids),overlap=result,reaction_disjoint_count=len(retained['reaction']),
                    reaction_and_patent_disjoint_count=len(certified)),retained['reaction'],certified
    summary,strict,both=compare(valid);nested,_,_=compare(valid[:256])
    def store(name,data):
        tmp=directory/(name+'.building')
        tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
        tmp.replace(directory/name)
    store('overlap_subsets.json',dict(reaction_disjoint=strict,reaction_and_patent_disjoint=both))
    result=dict(arm_training_draws=2048,cohort_file_sha256={p.name:expected[str(p)] for p in files},
        subset_sha256=sha(directory/'overlap_subsets.json'),full=summary,nested=nested)
    store('cohort_overlap.json',result)
    return result
if __name__=='__main__':
    here=Path(__file__).resolve().parent
    contract=here.parents[2]/'outputs/submissions/pr116_r_64x32_4h20_20261118_01/contract.json'
    print(json.dumps(run(here,contract)['full']))
