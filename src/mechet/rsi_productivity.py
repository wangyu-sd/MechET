"""Private net-reaction audit, not a physical feasibility oracle or actor input.

Reference and actor map namespaces are deliberately classified independently.
Only terminal net changes matter: elementary proton transfers remain legal.
"""
from rdkit import Chem


def net_reaction(target, precursor):
    def unknown(reason):
        return {'kind': 'unknown', 'reason': reason, 'confidence': 'unknown'}
    p = Chem.SmilesParserParams(); p.removeHs = False
    a, b = Chem.MolFromSmiles(target, p), Chem.MolFromSmiles(precursor, p)
    if a is None or b is None:
        return unknown('unparseable_state')
    def atoms(m):
        aa = {x.GetAtomMapNum(): x for x in m.GetAtoms()}
        return aa if len(aa) == m.GetNumAtoms() and 0 not in aa else None
    aa, bb = atoms(a), atoms(b)
    if aa is None or bb is None:
        return unknown('missing_or_duplicate_maps')
    origin = {k for k, x in aa.items() if x.GetAtomicNum() > 1}
    if any(k not in bb or aa[k].GetAtomicNum() != bb[k].GetAtomicNum() for k in origin):
        return unknown('target_origin_atom_not_conserved')
    def edges(m):
        result = {}
        for edge in m.GetBonds():
            u, v = edge.GetBeginAtom(), edge.GetEndAtom()
            if u.GetAtomicNum() <= 1 or v.GetAtomicNum() <= 1:
                continue
            key = tuple(sorted((u.GetAtomMapNum(), v.GetAtomMapNum())))
            if set(key) & origin:
                result[key] = (edge.GetBondTypeAsDouble(), str(edge.GetStereo()))
        return result
    ea, eb = edges(a), edges(b)
    evidence = {'added_bonds': sorted(set(eb)-set(ea)), 'removed_bonds': sorted(set(ea)-set(eb))}
    if set(ea) != set(eb):
        kind = 'bond_change'
    elif any(ea[k][0] != eb[k][0] for k in ea):
        kind = 'functional_group_or_redox'
    else:
        Chem.AssignStereochemistry(a, cleanIt=True, force=True)
        Chem.AssignStereochemistry(b, cleanIt=True, force=True)
        stereo = any((aa[k].GetProp('_CIPCode') if aa[k].HasProp('_CIPCode') else '') !=
                     (bb[k].GetProp('_CIPCode') if bb[k].HasProp('_CIPCode') else '') for k in origin)
        stereo |= any(ea[k][1] != eb[k][1] for k in ea)
        changes = [(bb[k].GetTotalNumHs(includeNeighbors=True)-aa[k].GetTotalNumHs(includeNeighbors=True),
                    bb[k].GetFormalCharge()-aa[k].GetFormalCharge(),
                    bb[k].GetNumRadicalElectrons()-aa[k].GetNumRadicalElectrons()) for k in origin]
        if stereo:
            kind = 'stereochemical'
        elif any(h or q or r for h, q, r in changes):
            kind = 'acid_base' if all(h == q and r == 0 for h, q, r in changes) else 'functional_group_or_redox'
        else:
            kind = 'identity'
    return {'kind': kind, 'reason': 'mapped_target_origin_terminal_net_change',
            'confidence': 'mapped_net_class_not_chemical_truth', **evidence}


def productivity(episode, reference):
    candidate = net_reaction(episode['initial_state'], episode['final_state'])
    kind, task = candidate['kind'], reference.get('kind', 'unknown')
    progress = None
    if episode.get('terminal') and kind != 'unknown' and task != 'unknown':
        progress = kind != 'identity' and not (kind == 'acid_base' and task != 'acid_base')
    return {'inverse_executor_valid': bool(episode.get('terminal')),
            'forward_roundtrip_success': bool(episode.get('outcome', {}).get('success')),
            'task_compatible_net_progress': progress, 'candidate': candidate,
            'reference': reference, 'label_scope': 'private_net_task_compatibility_not_experimental_feasibility'}
