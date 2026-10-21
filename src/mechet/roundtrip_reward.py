"""Conservative autonomous outcomes and exact-state preference attribution."""
from collections import Counter,defaultdict
from rdkit import Chem
from .rsi_contract import fingerprint


def components(smiles):
    p=Chem.SmilesParserParams();p.removeHs=False
    m=Chem.MolFromSmiles(smiles,p)
    if m is None:raise ValueError('invalid endpoint')
    for atom in m.GetAtoms():atom.SetAtomMapNum(0)
    m=Chem.RemoveHs(m)
    return Counter(Chem.MolToSmiles(c,canonical=True,isomericSmiles=True) for c in Chem.GetMolFrags(m,asMols=True))


def component_match(state,target):
    got,want=components(state),components(target)
    return bool(want) and all(got[k]>=v for k,v in want.items())


def forward_outcome(episode,target):
    if component_match(episode['initial_state'],target):
        return {'status':'nonproductive','success':False}
    if not episode.get('terminal'):return {'status':'censored','success':False}
    if episode.get('flow_count',0)==0:return {'status':'nonproductive','success':False}
    hit=component_match(episode['final_state'],target)
    return {'status':'roundtrip_hit' if hit else 'forward_mismatch','success':hit}


def preference_pairs(episodes):
    pairs=[];seen=set()
    for i,chosen in enumerate(episodes):
        if not chosen.get('outcome',{}).get('success'):continue
        for j,rejected in enumerate(episodes):
            if i==j:continue
            outcome=rejected.get('outcome',{}).get('status')
            for a,b in zip(chosen['steps'],rejected['steps']):
                if a['prompt_ids']!=b['prompt_ids']:break
                if a.get('successor')==b.get('successor') and a.get('accepted')==b.get('accepted'):continue
                if not a.get('accepted'):break
                evidence=('executor_rejected' if not b.get('accepted') else outcome)
                if evidence not in {'executor_rejected','nonproductive','forward_mismatch'}:break
                if not a.get('completion_ids') or not b.get('completion_ids'):break
                if not a.get('complete',True) or not b.get('complete',True):break
                if a['completion_ids']==b['completion_ids']:break
                key=fingerprint([a['prompt_ids'],a.get('successor'),b.get('successor'),b.get('error')])
                if key not in seen:
                    seen.add(key);pairs.append({'id':key,'prompt_ids':a['prompt_ids'],
                        'prompt_text':a.get('prompt_text'),'chosen_ids':a['completion_ids'],'rejected_ids':b['completion_ids'],
                        'chosen_raw':a['raw'],'rejected_raw':b['raw'],'chosen_episode':chosen['episode_id'],
                        'rejected_episode':rejected['episode_id'],'chosen_actor_sha':chosen['actor_sha'],
                        'rejected_actor_sha':rejected['actor_sha'],'tool':a['name'],'evidence':evidence})
                break
    return pairs
