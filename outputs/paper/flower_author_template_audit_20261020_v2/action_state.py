"""Resolve only EXPLICITLY RECORDED step imports for an offline GT audit."""
from rdkit import Chem


def maps(s):
    p=Chem.SmilesParserParams();p.removeHs=False
    m=Chem.MolFromSmiles(s,p)
    if m is None:
        raise ValueError('invalid recorded import/state')
    ids=[a.GetAtomMapNum() for a in m.GetAtoms()]
    if not all(ids) or len(ids)!=len(set(ids)):
        raise ValueError('nonunique or absent original maps')
    return set(ids)


def before_action(state_before,imports):
    present=maps(state_before)
    parts=[state_before]
    for s in imports:
        ids=maps(s)
        if ids<=present:
            continue
        if ids&present:
            raise ValueError('partial map overlap in recorded imports')
        parts.append(s)
        present|=ids
    return '.'.join(parts)
