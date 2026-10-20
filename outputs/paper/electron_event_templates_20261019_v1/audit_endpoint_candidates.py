"""Read-only re-score and candidate census of the already frozen full 3120 run."""
import csv
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
import statistics

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.warning')
ROOT = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
HERE = Path(__file__).parent
OUT = HERE/'endpoint_candidates'
FREEZE = ROOT/'outputs/paper/nmi_literature_evidence/source_checked_20261019_v18/prediction_freeze_manifest.json'
DATA = ROOT/'data/mech_uspto_31k_full_endpoint_rxnmapper/test.jsonl'


@lru_cache(maxsize=50000)
def normalized(value):
    parts=[]
    for s in value.split('.'):
        if not s.strip():
            continue
        p=Chem.SmilesParserParams();p.removeHs=False
        m=Chem.MolFromSmiles(s,p)
        if m is None:
            raise ValueError('unparseable structural endpoint')
        m=Chem.RemoveHs(m)
        for atom in m.GetAtoms():
            atom.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(m,canonical=True,isomericSmiles=True))
    return tuple(sorted(parts))


def main():
    freeze=json.loads(FREEZE.read_text())
    source=next(r for r in freeze['source_reports'] if r['run_id']=='mech31k_mcts_pr88')
    path=Path(source['report_path']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==source['report_sha256']
    report=json.loads(raw)
    data_raw=DATA.read_bytes()
    manifest=json.loads((DATA.parent/'manifest.json').read_text())
    assert hashlib.sha256(data_raw).hexdigest()==manifest['splits']['test']['endpoint_sha256']
    assert report['conditions']['source_sha']==manifest['splits']['test']['endpoint_sha256']
    data={r['id']:r for r in map(json.loads,data_raw.decode().splitlines())}
    rows=report['rows']
    assert len(rows)==len(data)==3120
    assert {r['id'] for r in rows}==set(data)
    OUT.mkdir(exist_ok=False)
    records=[];dist=Counter();matching=0;flag_changes=[];total=0
    for row in rows:
        reference=normalized(data[row['id']]['structural_precursor'])
        assert reference==normalized(row['expected_structural_precursor'])
        candidates=row['ranked_candidates']
        keys=[normalized(c['structural_precursor']) for c in candidates]
        assert len(keys)==len(set(keys))==row['terminal_count']
        hits=[k==reference for k in keys]
        for c,h in zip(candidates,hits):
            if h!=c['matches_reference']:
                flag_changes.append(row['id'])
        hit=any(hits)
        assert hit==row['any_endpoint_hit']
        matching+=hit;total+=len(keys);dist[len(keys)]+=1
        records.append({'reaction_id':row['id'], 'product':data[row['id']]['product_unmapped'],
                        'unique_structural_endpoints':len(keys), 'contains_reference':int(hit),
                        'matching_endpoints':sum(hits), 'reference_structural_precursor':'.'.join(reference),
                        'candidate_endpoint_sets_json':json.dumps(['.'.join(k) for k in keys]),
                        'completed_rollouts':row.get('mcts',{}).get('completed_rollouts'),
                        'collected_programs':row.get('collected_programs')})
    with (OUT/'per_reaction_candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    counts=[r['unique_structural_endpoints'] for r in records]
    result={'scope':'same frozen full3120 MCTS run as prior event-template analysis, no new generation',
            'endpoint_protocol':freeze['single_reaction_protocol'],
            'denominator':3120,'contains_reference':matching,'reference_coverage':matching/3120,
            'nonempty_candidate_sets':sum(x>0 for x in counts),'candidate_endpoint_total':total,
            'candidate_endpoints_mean':statistics.mean(counts),'candidate_endpoints_median':statistics.median(counts),
            'candidate_endpoints_min':min(counts),'candidate_endpoints_max':max(counts),
            'candidate_count_distribution':dict(sorted(dist.items())),
            'stored_match_flags_disagree':len(flag_changes),
            'all_references_match_frozen_dataset':True,
            'matching':'map-free, ordinary explicit-H normalized, canonical isomeric component MULTISET; charge retained; not neutralized; no candidate ranking used',
            'source_report':str(path),'source_report_sha256':source['report_sha256'],
            'source_dataset':str(DATA),'source_dataset_sha256':manifest['splits']['test']['endpoint_sha256'],
            'freeze_manifest_sha256':hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
            'csv_sha256':hashlib.sha256((OUT/'per_reaction_candidates.csv').read_bytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
