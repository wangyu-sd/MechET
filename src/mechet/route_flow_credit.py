"""Conservative persistent-edit attribution; never an extra FLOW reward.

Cancellation, ambiguous mapping or changed inventory keeps progress at FINISH.
These bookkeeping witnesses do not establish physical mechanistic causality.
"""
from collections import Counter
import math


def validate_alpha(alpha):
    if type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('INVALID_FLOW_CREDIT_ALPHA')
    return float(alpha)


def features(state):
    from rdkit import Chem
    parser=Chem.SmilesParserParams();parser.removeHs=False
    mol=Chem.MolFromSmiles(state,parser)
    if mol is None:raise ValueError('FLOW_CREDIT_INVALID_STATE')
    atoms={};values={}
    for atom in mol.GetAtoms():
        key=atom.GetAtomMapNum()
        if key<=0 or key in atoms:raise ValueError('FLOW_CREDIT_MAPPING')
        atoms[key]=(atom.GetAtomicNum(),atom.GetIsotope())
        values[('atom',key)]=(atom.GetFormalCharge(),atom.GetNumRadicalElectrons(),
                             atom.GetNumExplicitHs(),int(atom.GetChiralTag()))
    for bond in mol.GetBonds():
        pair=sorted([bond.GetBeginAtom().GetAtomMapNum(),bond.GetEndAtom().GetAtomMapNum()])
        values[('bond',*pair)]=(bond.GetBondTypeAsDouble(),int(bond.GetStereo()))
    return atoms,values


def changed(before,after):
    return {key for key in before.keys()|after.keys() if before.get(key)!=after.get(key)}


def contributors(records,finish):
    from .route_action_feedback import semantic_action
    endpoint=records[finish];occurrence=endpoint['occurrence']
    local=[(i,r,semantic_action(r.get('raw',''))) for i,r in enumerate(records[:finish])
           if r.get('occurrence')==occurrence and r.get('accepted') is True]
    flows=[(i,r) for i,r,action in local if action and action[1]['name']=='apply_electron_flow']
    if not flows:return []
    try:
        inventory,initial=features(flows[0][1]['state_before'])
        end_inventory,final=features(endpoint['state_before'])
        if end_inventory!=inventory:return []
        net=changed(initial,final);counts=Counter();out=[];previous=flows[0][1]['state_before']
        for index,record in flows:
            if previous!=record['state_before']:return []
            ia,a=features(record['state_before']);ib,b=features(record['state_after'])
            if ia!=inventory or ib!=inventory:return []
            edits=changed(a,b)
            if not edits or not edits<=net:return []
            counts.update(edits);out.append(index);previous=record['state_after']
        if previous!=endpoint['state_before'] or set(counts)!=net or any(v!=1 for v in counts.values()):
            return []
        return out
    except (KeyError,TypeError,ValueError):return []


def redistribute_progress(records,events,increments,*,alpha=0.):
    alpha=validate_alpha(alpha)
    if alpha==0:return list(events),[]
    result=list(events);evidence=[]
    for finish,amount in increments:
        indices=contributors(records,finish);shifted=alpha*amount if indices else 0.
        if indices:
            share=shifted/len(indices)
            for index in indices:result[index]+=share
            result[finish]-=shifted
        evidence.append(dict(finish_index=finish,progress=amount,flow_indices=indices,
                             shifted=shifted,attribution='unique_persistent_mapped_edit_v1'))
    if not math.isclose(math.fsum(result),math.fsum(events),rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError('FLOW_CREDIT_NOT_CONSERVED')
    return result,evidence
