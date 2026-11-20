"""Compare the actual Stage-II training file to the frozen validation targets, in one sequential read."""
import hashlib,json
from pathlib import Path

def main(directory,contract):
    from mechet.route_training_contract import chemical_identity
    cfg=json.loads(contract.read_bytes())
    lineage=cfg['actor_lineage']
    manifests=[Path(p) for p in cfg['files_sha256'] if '/data/pr96_flower_mainonly_history_' in p and p.endswith('/manifest.json')]
    if len(manifests)!=1:raise ValueError('ambiguous Stage-II data')
    manifest=manifests[0]
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=cfg['files_sha256'][str(manifest)]:raise ValueError('SFT manifest changed')
    valid=Path(cfg['step_budget']['valid'])
    if hashlib.sha256(valid.read_bytes()).hexdigest()!=cfg['files_sha256'][str(valid)]:raise ValueError('validation cohort changed')
    expected={chemical_identity(r['target']):r['task_id'] for r in map(json.loads,valid.read_text().splitlines())}
    if len(expected)!=512:raise ValueError('unexpected clean validation size')
    observed=set();source_ids=set();count=0;digest=hashlib.sha256()
    try:
        from orjson import loads
    except ImportError:loads=json.loads
    with (manifest.parent/'train.jsonl').open('rb',buffering=8<<20) as src:
        for line in src:
            digest.update(line)
            r=loads(line);observed.add(r['target_smiles']);source_ids.add(r['source_id']);count+=1
    if digest.hexdigest()!=lineage['train_file_sha256']:raise ValueError('Stage-II TRAIN file changed')
    unique={chemical_identity(s) for s in observed}
    matched=sorted(expected[s] for s in unique.intersection(expected))
    result=dict(scope='Stage-II TRAIN target_smiles vs PR116 clean validation final target; canonical stereo-preserving identity',
        actor_sha256=lineage['adapter_sha256'],history_train_sha256=digest.hexdigest(),history_train_rows=count,
        history_train_source_ids=len(source_ids),history_train_unique_products=len(unique),valid_denominator=len(expected),
        valid_file_sha256=cfg['files_sha256'][str(valid)],matched_count=len(matched),unmatched_count=len(expected)-len(matched))
    for name,data in [('sft_target_overlap.json',result),('sft_target_overlap_ids.json',matched)]:
        f=directory/name;tmp=f.with_suffix('.building');tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n');tmp.replace(f)
    print(result)
if __name__=='__main__':
    here=Path(__file__).resolve().parent
    main(here,here.parents[2]/'outputs/submissions/pr116_r_64x32_4h20_20261118_01/contract.json')
