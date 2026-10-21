"""Replay-verified state inversion; no generated reverse arrow text is inverted."""
from __future__ import annotations
from rdkit import Chem
from .forward_qwen_events import (initialize_forward_state,build_transition_row,
                                  execute_forward_call,forward_prompt,SYSTEM)
from .proof_program import sides_equal


def mol(text):
    p=Chem.SmilesParserParams();p.removeHs=False
    m=Chem.MolFromSmiles(text,p)
    if m is None:raise ValueError('invalid bridge state')
    return m


def maps(m):return {a.GetAtomMapNum() for a in m.GetAtoms()}


def text(m):return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)


def _bridge(episode):
    if not episode['terminal']:raise ValueError('inverse_not_terminal')
    accepted=[s for s in episode['steps'] if s['accepted']]
    if not any(s['name']=='apply_electron_flow' for s in accepted):raise ValueError('no_inverse_flow')
    endpoint=mol(episode['final_state']);universe=maps(endpoint)
    if 0 in universe or len(universe)!=endpoint.GetNumAtoms():raise ValueError('invalid_private_maps')
    imported=[]
    for step in accepted:
        if step['name']=='import_fragments':
            old=maps(mol(step['state_before']))
            for fragment in Chem.GetMolFrags(mol(step['state_after']),asMols=True):
                if not (maps(fragment)&old):imported.append(fragment)

    def lift(state):
        m=mol(state);present=maps(m)
        if present-universe:raise ValueError('atom_deleted_by_inverse')
        missing=universe-present
        for component in imported:
            cm=maps(component)
            if cm<=missing:
                m=Chem.CombineMols(m,component);missing-=cm
        if missing:raise ValueError('unexplained_missing_atoms')
        return m

    # Make implicit hydrogens explicit with stable identity, held through every
    # state. If their attachment/count changes without a recorded graph H,
    # refuse a witness rather than inventing an H trajectory.
    expanded=Chem.AddHs(endpoint);next_map=max(universe)+1;hslots={}
    for atom in expanded.GetAtoms():
        if atom.GetAtomMapNum()==0:
            if atom.GetAtomicNum()!=1 or len(atom.GetNeighbors())!=1:raise ValueError('anonymous_nonH')
            parent=atom.GetNeighbors()[0].GetAtomMapNum()
            atom.SetAtomMapNum(next_map);hslots.setdefault(parent,[]).append(next_map);next_map+=1
    forward_initial=initialize_forward_state(text(expanded))
    initial=mol(forward_initial)
    match=initial.GetSubstructMatch(expanded,useChirality=True)
    if len(match)!=expanded.GetNumAtoms():raise ValueError('forward_remap_not_isomorphic')
    remap={a.GetAtomMapNum():initial.GetAtomWithIdx(match[a.GetIdx()]).GetAtomMapNum() for a in expanded.GetAtoms()}

    def normalize(state):
        m=Chem.AddHs(lift(state));used={}
        for atom in m.GetAtoms():
            if atom.GetAtomMapNum()==0:
                parent=atom.GetNeighbors()[0].GetAtomMapNum();index=used.get(parent,0)
                if index>=len(hslots.get(parent,[])):raise ValueError('implicit_H_transfer_unavailable')
                atom.SetAtomMapNum(hslots[parent][index]);used[parent]=index+1
        if maps(m)!=set(remap):raise ValueError('hydrogen_inventory_mismatch')
        for atom in m.GetAtoms():atom.SetAtomMapNum(remap[atom.GetAtomMapNum()])
        return text(m)

    current=forward_initial;history=[];witness=[]
    for step in reversed(accepted):
        if step['name']!='apply_electron_flow':continue
        source,destination=normalize(step['state_after']),normalize(step['state_before'])
        if not sides_equal(current,source,ignore_maps=False):raise ValueError('noncontiguous_forward_witness')
        row=build_transition_row(f'{source}>>{destination}|rsi_witness',reaction_id='rsi',decision_index=len(witness),history=history)
        call=row['messages'][2]['tool_calls'][0]['function']
        replay=execute_forward_call(current,call['name'],call['arguments'])
        if not replay.get('ok'):raise ValueError('witness_replay_failed')
        witness.append({'state_before':source,'state_after':replay['state_smiles'],
                        'name':call['name'],'arguments':call['arguments'],'messages':row['messages'][:2]})
        history.append({'name':call['name'],'arguments':call['arguments']});current=replay['state_smiles']
    witness.append({'state_before':current,'state_after':current,'name':'finish_forward','arguments':{},
                    'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':forward_prompt(current,history)}]})
    return {'status':'verified','initial_state':forward_initial,'witness':witness,
            'terminal_state':current,'policy':'full_inverse_terminal_mixture_v1',
            'import_components_lifted':len(imported),'explicit_H_count':sum(a.GetAtomicNum()==1 for a in initial.GetAtoms())}


def bridge_reverse_episode(episode):
    try:return _bridge(episode)
    except Exception as exc:
        # Autonomous forward checking remains available even without a witness.
        try:initial=initialize_forward_state(episode['final_state'])
        except Exception:initial=None
        return {'status':'bridge_failed','error':f'{type(exc).__name__}:{exc}',
                'initial_state':initial,'witness':[],'policy':'full_inverse_terminal_mixture_v1'}
