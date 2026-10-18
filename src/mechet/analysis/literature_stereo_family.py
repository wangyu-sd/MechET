"""Versioned, full-stereo preserving compatibility with immutable v1 indices.

Legacy partial SMILES can contain an invalid local @/@@ tag whose parity
depends on the full graph traversal. Only such undefined *partial* tags get
lookup aliases. Every analogue is then rechecked using full-graph CIP labels
and an explicit-H canonical local graph. Exact reaction identity is unchanged.
"""
from functools import lru_cache
import itertools
import json
import re

from rdkit import Chem

from .literature_index import source_family
from .literature_reaction_keys import _mapped_graph, _environment, make_reaction, stable_hash

POLICY='full_stereo_local_v2'
ALIAS_LIMIT=256  # below SQLite's portable parameter limit; never a reaction filter
_STEREO_TOKENS=re.compile(r'@@|@|[/\\]')


def _parse_partial(text):
    parser=Chem.SmilesParserParams();parser.removeHs=False;parser.sanitize=False
    molecule=Chem.MolFromSmiles(text,parser)
    if molecule is None:raise ValueError('unparseable local graph')
    # Cutting an aromatic ring creates legitimate dangling aromatic boundary
    # atoms. This descriptor is a graph fragment, not an isolated molecule to
    # kekulize; aromatic flags and original H counts must remain intact.
    Chem.SanitizeMol(molecule,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    Chem.AssignStereochemistry(molecule,cleanIt=True,force=True)
    return molecule


def _typed_environment(graph,key,radius):
    if key not in graph[1]:return None
    mol=Chem.Mol(graph[0]);root=graph[1][key]
    Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    selected={root}
    for _ in range(radius):
        selected|={n.GetIdx() for i in tuple(selected) for n in mol.GetAtomWithIdx(i).GetNeighbors()}
    # These are descriptor colour labels, NOT physical atom addresses. They
    # retain full-graph stereo even when a truncated neighbourhood is achiral.
    labels={'R':101,'S':102,'r':103,'s':104}
    for atom in mol.GetAtoms():
        label=0
        if atom.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED:
            if not atom.HasProp('_CIPCode') or atom.GetProp('_CIPCode') not in labels:
                raise ValueError('unsupported full-graph stereochemical descriptor')
            label=labels[atom.GetProp('_CIPCode')]
        atom.SetAtomMapNum(label+(1000 if atom.GetIdx()==root else 0))
    partial=Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(selected),canonical=True,
                                   isomericSmiles=True,allHsExplicit=True)
    canonical=_parse_partial(partial)
    return Chem.MolToSmiles(canonical,canonical=True,isomericSmiles=True,allHsExplicit=True)


def _legacy_variants(text,limit):
    if text is None:return (None,)
    tokens=list(_STEREO_TOKENS.finditer(text))
    if not tokens:return (text,)
    if 2**len(tokens)>limit:return None
    baseline=Chem.MolToSmiles(_parse_partial(text),canonical=True,isomericSmiles=True,allHsExplicit=True)
    opposites={'@':'@@','@@':'@','/':'\\','\\':'/'}
    choices=[(m.group(),opposites[m.group()]) for m in tokens]
    result=[]
    for values in itertools.product(*choices):
        pieces=[];position=0
        for match,value in zip(tokens,values):
            pieces.extend((text[position:match.start()],value))
            position=match.end()
        pieces.append(text[position:]);candidate=''.join(pieces)
        canonical=Chem.MolToSmiles(_parse_partial(candidate),canonical=True,isomericSmiles=True,allHsExplicit=True)
        if canonical==baseline:result.append(candidate)
    return tuple(sorted(set(result)))


@lru_cache(maxsize=50000)
def family_context(precursors,product,*,radius=1,alias_limit=ALIAS_LIMIT):
    if radius!=1 or type(alias_limit) is not int or not 1<=alias_limit<=ALIAS_LIMIT:
        raise ValueError('unsupported frozen radius or legacy-lookup work budget')
    chemical=make_reaction(product,[precursors])
    base={'reaction_key':chemical['reaction_key'],'policy':POLICY,
        'family_key_v2':None,'legacy_family_keys':(),'legacy_alias_space_complete':False,
        'legacy_alias_limit':alias_limit,'radius':radius}
    before,after=_mapped_graph(precursors),_mapped_graph(product)
    if before is None or after is None or not set(after[1])<=set(before[1]):
        return {**base,'status':'mapping_unknown'}
    common=before[1].keys()&after[1].keys()
    for key in common:
        a=before[0].GetAtomWithIdx(before[1][key]);b=after[0].GetAtomWithIdx(after[1][key])
        if (a.GetAtomicNum(),a.GetIsotope())!=(b.GetAtomicNum(),b.GetIsotope()):
            return {**base,'status':'mapping_unknown'}
    pairs=[pair for pair in sorted(before[2].keys()|after[2].keys()) if before[2].get(pair)!=after[2].get(pair)]
    charges=[key for key in sorted(common) if before[0].GetAtomWithIdx(before[1][key]).GetFormalCharge()!=
             after[0].GetAtomWithIdx(after[1][key]).GetFormalCharge()]
    if not pairs and not charges:return {**base,'status':'no_reaction_center'}
    atoms=sorted({k for pair in pairs for k in pair}|set(charges))
    def payload(environments):
        edits=[]
        for pair in pairs:
            ends=sorted([(environments[0,k],environments[1,k]) for k in pair],key=str)
            edits.append({'ends':ends,'before':before[2].get(pair),'after':after[2].get(pair)})
        delta=[[environments[0,k],environments[1,k],
                before[0].GetAtomWithIdx(before[1][k]).GetFormalCharge(),
                after[0].GetAtomWithIdx(after[1][k]).GetFormalCharge()] for k in charges]
        return {'radius':radius,'bonds':sorted(edits,key=str),'charges':sorted(delta,key=str)}
    try:
        typed={(side,k):_typed_environment(graph,k,radius) for side,graph in enumerate((before,after)) for k in atoms}
        full_key=stable_hash({'policy':POLICY,'environment_atom_labels':'full-CIP colour + root, not atom addresses',
                              'edits':payload(typed)})
        old={(side,k):_environment(graph,k,radius) for side,graph in enumerate((before,after)) for k in atoms}
        options={key:_legacy_variants(text,alias_limit) for key,text in old.items()}
    except (ValueError,RuntimeError) as error:
        return {**base,'status':'local_stereo_unresolved','reason':str(error)[:300]}
    count=1
    for values in options.values():
        if values is None:count=alias_limit+1;break
        count*=len(values)
        if count>alias_limit:break
    if count>alias_limit:
        return {**base,'family_key_v2':full_key,'status':'alias_space_unresolved'}
    keys=tuple(options)
    aliases={stable_hash(payload(dict(zip(keys,values)))) for values in itertools.product(*(options[k] for k in keys))}
    return {**base,'status':'available','family_key_v2':full_key,
            'legacy_family_keys':tuple(sorted(aliases)),'legacy_alias_space_complete':True}


def query_sources(index,reaction,context,*,cap=5):
    if not 1<=cap<=5:raise ValueError('primary-source cap must be1..5')
    if context['reaction_key']!=reaction['reaction_key']:
        raise ValueError('stereo retrieval context belongs to another complete reaction')
    audit={'policy':POLICY,'context_status':context['status'],
        'legacy_alias_space_complete':context['legacy_alias_space_complete'],
        'legacy_family_lookups':0,'incompatible_family_candidates':0,'source_records_considered':0}
    exact=index.connection.execute('SELECT record FROM reactions WHERE reaction_key=? ORDER BY source_id',
                                   (reaction['reaction_key'],))
    result=[];families={}
    def accept(record,match):
        family=source_family(record)
        if family in families:
            families[family]['source_representation_ids'].append(record['source_id']);return
        selected={**record,'match_type':match,'source_family_key':family,
            'source_representation_ids':[record['source_id']],'retrieval_status':'indexed_candidate',
            'verification_status':'machine_structural_match','family_compatibility_policy':POLICY,
            'representation_inventory':'encountered before distinct-source cap'}
        families[family]=selected;result.append(selected)
    for (encoded,) in exact:
        record=json.loads(encoded);audit['source_records_considered']+=1;accept(record,'exact')
        if len(result)==cap:return result,audit
    if context['status']=='available':
        aliases=context['legacy_family_keys'];audit['legacy_family_lookups']=len(aliases)
        sql='SELECT record FROM reactions WHERE family_key IN ('+','.join('?' for _ in aliases)+') AND reaction_key!=? ORDER BY source_id'
        for (encoded,) in index.connection.execute(sql,(*aliases,reaction['reaction_key'])):
            record=json.loads(encoded);audit['source_records_considered']+=1
            other=family_context('.'.join(record['precursors']),record['product'])
            if other['family_key_v2'] is None or other['family_key_v2']!=context['family_key_v2']:
                audit['incompatible_family_candidates']+=1;continue
            accept(record,'local_edit_analogue_candidate')
            if len(result)==cap:break
    return result,audit
