"""Incremental offline author-rule repair. Historical v2 is immutable."""
from pathlib import Path
import sys
from collections import Counter,defaultdict
import hashlib

from rdkit import Chem

V2=Path(__file__).resolve().parent.parent/'flower_author_template_audit_20261020_v2'
sys.path.insert(0,str(V2))
import representation_match as legacy

compile_rule=legacy.compile_rule


class Event(legacy.Event):
    def match(self,rule,max_work=100000):
        # No radical-count blanket rejection. The unchanged v2 replay still
        # requires the FULL serialized target, including its radical state.
        if self.unchanged or self.sigma_key!=rule.sigma_key:
            return None
        for side in (0,1):
            for kekule in (False,True):
                proof=self.view(side,kekule).match(rule,side,max_work)
                if proof:
                    proof['radical_counts_checked_via_full_state']=True
                    return proof
        return None


def match_event(before,after,templates):
    try:
        event=Event(before,after)
    except ValueError:
        return []
    return [dict(reaction_smarts=s,**proof) for s in templates
            if (proof:=event.match(compile_rule(s)))]


def group_rules(rules):
    groups=defaultdict(list)
    for rule in rules:
        groups[(rule.sigma_key,rule.typed_key)].append(rule)
    return groups


def _edit(m,bonds,charges,aromatic):
    rw=Chem.RWMol(m)
    ix={a.GetAtomMapNum():a.GetIdx() for a in rw.GetAtoms()}
    for (a,b),desired in bonds.items():
        if rw.GetBondBetweenAtoms(ix[a],ix[b]):rw.RemoveBond(ix[a],ix[b])
        if desired is not None:
            rw.AddBond(ix[a],ix[b],desired)
            rw.GetBondBetweenAtoms(ix[a],ix[b]).SetIsAromatic(desired==Chem.BondType.AROMATIC)
    for i,q in charges.items():
        a=rw.GetAtomWithIdx(ix[i]);a.SetFormalCharge(a.GetFormalCharge()+q)
    for i,value in aromatic.items():rw.GetAtomWithIdx(ix[i]).SetIsAromatic(value)
    x=rw.GetMol()
    Chem.SanitizeMol(x)
    return x


def match_composition(before,after,rules,max_states=4096):
    """Exact author-rule composition, reported SEPARATELY from a single rule.

    Decompose only identifiable mapped proton transfers plus one residual
    event. Every component needs its own whole-state author-rule replay; no
    free splitting of individual electron arrows, no intermediate atom import.
    The number of units is fixed by the observed transition, not a search
    depth heuristic. Safety exhaustion is inconclusive, never a negative.
    """
    try:
        event=Event(before,after)
    except ValueError:
        return None
    source,target=event.raw[1],event.raw[0]  # saved inverse direction
    a,b=map(legacy.atom_dict,(source,target))
    sb,tb=map(legacy.bond_dict,(source,target))
    changed={k:(tb[k].GetBondType() if k in tb else None)
             for k in sb.keys()|tb.keys()
             if k not in sb or k not in tb or sb[k].GetBondType()!=tb[k].GetBondType()}
    charges=Counter({i:b[i].GetFormalCharge()-a[i].GetFormalCharge() for i in a})
    parts=[]
    for h,x in a.items():
        y=b[h]
        if x.GetAtomicNum()!=1 or x.GetDegree()!=1 or y.GetDegree()!=1:continue
        if x.GetFormalCharge() or y.GetFormalCharge() or x.GetNumRadicalElectrons() or y.GetNumRadicalElectrons():continue
        d=x.GetNeighbors()[0].GetAtomMapNum();c=y.GetNeighbors()[0].GetAtomMapNum()
        if d==c or a[d].GetAtomicNum()==1 or b[c].GetAtomicNum()==1:continue
        old,new=tuple(sorted((d,h))),tuple(sorted((c,h)))
        if old not in changed or changed.get(new)!=Chem.BondType.SINGLE:continue
        if sb[old].GetBondType()!=Chem.BondType.SINGLE:continue
        delta={d:-1,c:1}
        parts.append((f'proton:{h}',{old:None,new:Chem.BondType.SINGLE},delta,{}))
        del changed[old];del changed[new]
        for i,q in delta.items():charges[i]-=q
    if not parts:return None
    aromatic={i:b[i].GetIsAromatic() for i in a if a[i].GetIsAromatic()!=b[i].GetIsAromatic()}
    charges={i:q for i,q in charges.items() if q}
    if changed or charges or aromatic:parts.append(('residual_core',changed,charges,aromatic))
    if len(parts)<2:return None
    groups=rules if isinstance(rules,dict) else group_rules(rules)
    full=(1<<len(parts))-1
    visited=set();work=0

    def search(m,mask,steps):
        nonlocal work
        key=legacy.normalized(Chem.AddHs(m),event.original_maps)
        if mask==full:
            if key==event.keys[0]:
                return dict(kind='author_template_composition',steps=steps,
                            normalized_full_state_sha256=hashlib.sha256(key.encode()).hexdigest())
            return None
        if (mask,key) in visited:return None
        visited.add((mask,key));work+=1
        if work>max_states:raise legacy.MatchWorkLimit('composition states exhausted; inconclusive')
        for j,(name,bonds,dq,arom) in enumerate(parts):
            if mask&(1<<j):continue
            try:
                nxt=_edit(m,bonds,dq,arom)
                s0,s1=(Chem.MolToSmiles(x,canonical=True,isomericSmiles=True) for x in (m,nxt))
                sub=Event(s0,s1)
            except (ValueError,RuntimeError):continue
            for rule in sub.candidate_rules(groups):
                proof=sub.match(rule)
                if proof:
                    result=search(nxt,mask|(1<<j),steps+[dict(unit=name,state_before=s0,state_after=s1,
                                  template_id=hashlib.sha256(rule.raw.encode()).hexdigest(),**proof)])
                    if result:return result
                    # Same state/remaining units regardless of template witness.
                    break
        return None

    return search(source,0,[])
