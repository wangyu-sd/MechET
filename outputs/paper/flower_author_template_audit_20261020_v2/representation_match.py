"""Representation-aware author-template replay (offline analysis only).

Acceptance requires a WHOLE mapped-molecule replay certificate. Query matching
alone, ignoring aromatic bonds, or dropping H/charges is NEVER acceptance.
The original author hydrogen-expansion routine is pinned and reused. Ordinary
implicit H is materialized as unlabelled context, while every original mapped
H, isotope, charge and stereocentre remains part of the comparison.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'flower_author_template_audit_20261019_v1'
sys.path.insert(0,str(PREVIOUS))
from author_template_match import query_charge, fixed_h, read_literal, MatchWorkLimit, UnsupportedTemplate

AUTHOR = Path('/aaa/fionafyang/buddy1/whaleywang/datasets/flower_original_mechanistic_templates_20261019')
H_SOURCE = AUTHOR/'Mechanistic_dataset/utils/explicit_H.py'
H_SPEC = importlib.util.spec_from_file_location('pinned_flower_author_explicit_h',H_SOURCE)
H_MOD = importlib.util.module_from_spec(H_SPEC)
H_SPEC.loader.exec_module(H_MOD)
RDLogger.DisableLog('rdApp.warning')


def atom_dict(m):
    return {a.GetAtomMapNum():a for a in m.GetAtoms()}


def bond_dict(m):
    return {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum()))):b
            for b in m.GetBonds()}


def parse(s):
    p=Chem.SmilesParserParams()
    p.removeHs=False
    m=Chem.MolFromSmiles(s,p)
    if m is None:
        raise ValueError('unparseable saved state')
    ids=[a.GetAtomMapNum() for a in m.GetAtoms()]
    if not all(ids) or len(ids)!=len(set(ids)):
        raise ValueError('requires unique positive ORIGINAL maps')
    return m


def hydrogen_projection(m):
    p=Chem.RemoveHsParameters()
    p.removeMapped=True
    p.removeHydrides=False
    p.removeIsotopes=False
    x=Chem.RemoveHs(m,p,sanitize=False)
    x.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(x)
    return x


def normalized(m, original_maps):
    """Only representation transforms. No neutralization or tautomer changes."""
    x=Chem.Mol(m)
    x.UpdatePropertyCache(strict=False)
    Chem.Kekulize(x,clearAromaticFlags=True)
    Chem.SanitizeMol(x)
    for a in x.GetAtoms():
        if a.GetAtomMapNum() not in original_maps:
            a.SetAtomMapNum(0)
    return Chem.MolToSmiles(x,canonical=True,isomericSmiles=True)


@dataclass
class ReplayRule:
    raw: str
    expanded: str
    q: tuple
    original: tuple
    b: tuple
    dq: dict
    dh: dict
    changed_bonds: set
    centre: set
    context_h: set
    sigma_key: tuple
    typed_key: tuple
    refs: list


def sigma_key(b0,b1,dq,atoms):
    # Presence changes among heavy atoms only. Pi/aromatic representation is
    # left to the full replay certificate, not erased in the final comparison.
    broken=sum(k not in b1 and all(atoms[i].GetAtomicNum()!=1 for i in k) for k in b0)
    formed=sum(k not in b0 and all(atoms[i].GetAtomicNum()!=1 for i in k) for k in b1)
    return broken,formed,tuple(sorted(v for v in dq.values() if v))


def sort_typed(rows):
    return tuple(sorted(rows,key=lambda x:(x[0],x[1],-999 if x[2] is None else x[2])))


@lru_cache(maxsize=65536)
def compile_rule(raw):
    expanded,_=H_MOD.modify_explicit_H(raw)
    q=tuple(Chem.MolFromSmarts(s) for s in expanded.split('>>'))
    original=tuple(Chem.MolFromSmarts(s) for s in raw.split('>>'))
    if any(m is None for m in (*q,*original)):
        raise UnsupportedTemplate('author SMARTS expansion parse failure')
    a=tuple(atom_dict(m) for m in q)
    orig=tuple(atom_dict(m) for m in original)
    if any(0 in x or len(x)!=m.GetNumAtoms() for x,m in zip(a,q)) or a[0].keys()!=a[1].keys():
        raise UnsupportedTemplate('unmapped/repeated/added/deleted expanded atom')
    if orig[0].keys()!=orig[1].keys():
        raise UnsupportedTemplate('unbalanced original mapped atom inventory')
    # Original predicates (notably D) are evaluated in their implicit-H query
    # representation. Added H vertices use their expanded explicit predicates.
    queries=tuple({i:orig[s].get(i,x) for i,x in a[s].items()} for s in (0,1))
    b=tuple(bond_dict(m) for m in q)
    changed={k for k in b[0].keys()|b[1].keys()
             if k not in b[0] or k not in b[1] or b[0][k].GetBondType()!=b[1][k].GetBondType()}
    for k in changed:
        if any(k in side and side[k].GetBondTypeAsDouble() not in (1,1.5,2,3)
               for side in b):
            raise UnsupportedTemplate('changed non-concrete query bond')
    dq={i:query_charge(a[1][i])-query_charge(a[0][i]) for i in a[0]}
    dh={}
    for i,x in a[0].items():
        h0,h1=fixed_h(queries[0][i]),fixed_h(queries[1][i])
        dh[i]=h1-h0 if x.GetAtomicNum()!=1 and h0 is not None and h1 is not None else None
    centre={i for k in changed for i in k}|{i for i,v in dq.items() if v}
    if not centre:
        raise UnsupportedTemplate('no expanded bond/charge event')
    context_h={i for i in a[0] if i not in centre and a[0][i].GetAtomicNum()==1}
    typed=sort_typed((a[0][i].GetAtomicNum(),v,dh[i]) for i,v in dq.items() if v)
    return ReplayRule(raw,expanded,queries,orig,b,dq,dh,changed,centre,context_h,
                      sigma_key(b[0],b[1],dq,a[0]),typed,[])


class Event:
    def __init__(self,before,after):
        self.raw=(parse(after),parse(before))  # FORWARD orientation
        atoms=tuple(atom_dict(x) for x in self.raw)
        if atoms[0].keys()!=atoms[1].keys():
            raise ValueError('ORIGINAL atom inventory differs')
        self.original_maps=set(atoms[0])
        for i in self.original_maps:
            if (atoms[0][i].GetAtomicNum(),atoms[0][i].GetIsotope()) != (atoms[1][i].GetAtomicNum(),atoms[1][i].GetIsotope()):
                raise ValueError('element/isotope transmutation')
        self.radical_changed=any(atoms[0][i].GetNumRadicalElectrons()!=atoms[1][i].GetNumRadicalElectrons() for i in atoms[0])
        self.dq={i:atoms[1][i].GetFormalCharge()-atoms[0][i].GetFormalCharge() for i in atoms[0]}
        self.dh={i:atoms[1][i].GetTotalNumHs(includeNeighbors=True)-atoms[0][i].GetTotalNumHs(includeNeighbors=True) for i in atoms[0]}
        self.sigma_key=sigma_key(*[bond_dict(x) for x in self.raw],self.dq,atoms[0])
        self.m=tuple(Chem.AddHs(x) for x in self.raw)
        base=max(self.original_maps)+1
        rank={i:r for r,i in enumerate(sorted(self.original_maps))}
        for m in self.m:
            n=Counter()
            for a in m.GetAtoms():
                if a.GetAtomMapNum():
                    continue
                parent=a.GetNeighbors()[0].GetAtomMapNum()
                if n[parent]>=32:
                    raise ValueError('unsupported implicit H multiplicity')
                a.SetAtomMapNum(base+32*rank[parent]+n[parent])
                n[parent]+=1
        self.keys=tuple(normalized(x,self.original_maps) for x in self.m)
        self.unchanged=self.keys[0]==self.keys[1]
        self.views={}
        self.atoms=atoms

    def candidate_rules(self,groups):
        from itertools import product
        changes=[(self.atoms[0][i].GetAtomicNum(),v,self.dh[i]) for i,v in self.dq.items() if v]
        choices=[[(z,q,h),(0,q,h),(z,q,None),(0,q,None)] for z,q,h in changes]
        for key in {sort_typed(v) for v in product(*choices)}:
            yield from groups.get((self.sigma_key,key),())

    def view(self,side,kekule):
        key=(side,kekule)
        if key not in self.views:
            x=Chem.Mol(self.m[side])
            if kekule:
                Chem.Kekulize(x,clearAromaticFlags=True)
            self.views[key]=View(x,self)
        return self.views[key]

    def match(self,rule,max_work=100000):
        if self.radical_changed or self.unchanged or self.sigma_key!=rule.sigma_key:
            return None
        for side in (0,1):
            for kekule in (False,True):
                result=self.view(side,kekule).match(rule,side,max_work)
                if result:
                    return result
        return None


class View:
    def __init__(self,m,event):
        self.m=m
        self.event=event
        self.heavy=hydrogen_projection(m)
        self.atoms=atom_dict(m)
        self.bonds=bond_dict(m)
        self.cache={}

    def query_atoms(self,q,original):
        key=q.GetSmarts(),original
        if key not in self.cache:
            view=self.heavy if original and q.GetAtomicNum()!=1 else self.m
            query=Chem.MolFromSmarts(q.GetSmarts())
            self.cache[key]={view.GetAtomWithIdx(x[0]).GetAtomMapNum() for x in
                             view.GetSubstructMatches(query,uniquify=False,maxMatches=0)}
        return self.cache[key]

    def match(self,rule,side,max_work):
        event=self.event
        candidates={}
        for i,q in rule.q[side].items():
            c=self.query_atoms(q,i in rule.original[side])
            c={j for j in c if rule.dq[i]==event.dq.get(j,0) and
               (rule.dh[i] is None or rule.dh[i]==event.dh.get(j,0))}
            if not c:
                return None
            candidates[i]=c
        assigned,used={},set()
        work=0
        edges=rule.b[side]
        def source_pair_ok(i,j,x,y):
            k=tuple(sorted((i,j)))
            b=self.bonds.get(tuple(sorted((x,y))))
            return k not in edges or b is not None and edges[k].Match(b)

        def search():
            nonlocal work
            work+=1
            if work>max_work:
                raise MatchWorkLimit('replay binding safety guard; inconclusive')
            if len(assigned)==len(candidates):
                return self.replay(rule,side,assigned)
            rest=[i for i in candidates if i not in assigned]
            i=min(rest,key=lambda x:(x not in rule.centre,
                  -sum(tuple(sorted((x,j))) in edges for j in assigned),len(candidates[x])))
            for x in sorted(candidates[i]-used):
                if all(source_pair_ok(i,j,x,y) for j,y in assigned.items()):
                    assigned[i]=x
                    used.add(x)
                    r=search()
                    used.remove(x)
                    del assigned[i]
                    if r:
                        return r
                    if i in rule.context_h:
                        break  # unchanged interchangeable H cannot affect edits
            return None
        return search()

    def replay(self,rule,side,binding):
        target=1-side
        rw=Chem.RWMol(self.m)
        ix={a.GetAtomMapNum():a.GetIdx() for a in rw.GetAtoms()}
        for a,b in rule.changed_bonds:
            x,y=ix[binding[a]],ix[binding[b]]
            existing=rw.GetBondBetweenAtoms(x,y)
            desired=rule.b[target].get(tuple(sorted((a,b))))
            if existing:
                rw.RemoveBond(x,y)
            if desired:
                rw.AddBond(x,y,desired.GetBondType())
                rw.GetBondBetweenAtoms(x,y).SetIsAromatic(desired.GetBondType()==Chem.BondType.AROMATIC)
        sign=1 if side==0 else -1
        for i,q in rule.dq.items():
            atom=rw.GetAtomWithIdx(ix[binding[i]])
            atom.SetFormalCharge(atom.GetFormalCharge()+sign*q)
        projected=rw.GetMol()
        projected.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(projected)
        other=View(projected,self.event)
        # Check the author's counterpart predicates on the symbolic result,
        # BEFORE valid aromatic/Kekule normalization of a sigma complex.
        if any(binding[i] not in other.query_atoms(q,i in rule.original[target]) for i,q in rule.q[target].items()):
            return None
        try:
            key=normalized(projected,self.event.original_maps)
        except Exception:
            return None
        if key!=self.event.keys[target]:
            return None
        return dict(template_to_state_atom_map=dict(binding), replay_start_side=side,
                    replay_start_representation='aromatic' if any(a.GetIsAromatic() for a in self.m.GetAtoms()) else 'kekule_or_nonaromatic',
                    normalized_full_state_sha256=hashlib.sha256(key.encode()).hexdigest(),
                    original_atom_maps_preserved=True,virtual_context_hydrogens_unlabelled=True)


def match_event(before,after,templates):
    try:
        event=Event(before,after)
    except ValueError:
        return []
    out=[]
    for s in templates:
        rule=compile_rule(s)
        match=event.match(rule)
        if match:
            out.append(dict(reaction_smarts=s,expanded_smarts=rule.expanded,**match))
    return out
