"""Source-anchored controls of structure retrieval, not chemical yield validation.

The source reactions are recorded experiments; counterfactual query variants
are explicitly hypothetical and never become MechET source evidence grades.
"""
import gzip
import json
from pathlib import Path

from rdkit import Chem

from .literature_evidence import assess
from .literature_index import ReactionIndex
from .literature_reaction_keys import make_reaction
from .literature_sources import file_sha256, iter_ord_messages, ord_records


def variant_smiles(text, variant):
    parser=Chem.SmilesParserParams();parser.removeHs=False
    molecule=Chem.MolFromSmiles(text,parser)
    if molecule is None:raise ValueError('invalid independent control structure')
    if variant=='consistent_map_permutation':
        for atom in molecule.GetAtoms():atom.SetAtomMapNum(atom.GetAtomMapNum()+1000)
    elif variant in {'remote_benzyl_fluoro','reactive_environment_element_change'}:
        target=7 if variant=='remote_benzyl_fluoro' else 2
        selected=[a for a in molecule.GetAtoms() if a.GetAtomMapNum()==target]
        if len(selected)!=1:raise ValueError('missing unique predeclared control atom')
        atom=selected[0]
        if variant=='remote_benzyl_fluoro':
            if atom.GetAtomicNum()!=6 or not atom.GetIsAromatic() or atom.GetTotalNumHs()!=1:
                raise ValueError('remote control atom must be the declared aryl CH')
            editable=Chem.RWMol(molecule);atom=editable.GetAtomWithIdx(atom.GetIdx())
            atom.SetNumExplicitHs(0);atom.SetNoImplicit(True)
            fluorine=Chem.Atom(9);fluorine.SetAtomMapNum(10000)
            if any(a.GetAtomMapNum()==10000 for a in molecule.GetAtoms()):
                raise ValueError('control fluorine address already present')
            f=editable.AddAtom(fluorine);editable.AddBond(atom.GetIdx(),f,Chem.BondType.SINGLE)
            molecule=editable.GetMol()
        else:
            if atom.GetAtomicNum()!=7:raise ValueError('reactive-environment control requires mapped N2')
            atom.SetAtomicNum(15)
    elif variant!='recorded_structures':raise ValueError('unsupported predeclared control variant')
    Chem.SanitizeMol(molecule)
    return Chem.MolToSmiles(molecule,canonical=True,isomericSmiles=True)


def evaluate_controls(spec,index_path):
    """Execute every predeclared source/variant against a real local index."""
    index_path=Path(index_path)
    if index_path.exists():raise FileExistsError('independent control index already exists')
    if spec['radius']!=1 or spec['primary_search_cap']!=5:
        raise ValueError('controls validate the unchanged radius1/cap5 policy only')
    sources=spec['sources'];variants=spec['variants']
    if (not sources or not variants or len({r['id'] for r in sources})!=len(sources) or
        len({r['id'] for r in variants})!=len(variants)):
        raise ValueError('empty or duplicate predeclared controls')
    index=ReactionIndex(index_path)
    results=[]
    try:
        for source in sources:
            index.add({'source_id':source['id'],'product':source['mapped_product'],
                'precursors':[source['mapped_precursors']],'license':'CC-BY-SA-4.0',
                'mapping_basis':'explicit independent control correspondence, not MechET GT'})
        index.commit()
        for source in sources:
            for variant in variants:
                reaction=make_reaction(variant_smiles(source['mapped_product'],variant['id']),
                    [variant_smiles(source['mapped_precursors'],variant['id'])],radius=1)
                hits=index.query(reaction,cap=5)
                own=[h for h in hits if h['source_id']==source['id']]
                observed=own[0]['match_type'] if own else 'unresolved'
                # A changed reactive environment must not leak through another
                # source in the control index either.
                passed=observed==variant['expected_retrieval'] and not (
                    variant['expected_retrieval']=='unresolved' and hits)
                result={'source_id':source['id'],'variant':variant['id'],
                    'query_reaction_key':reaction['reaction_key'],'query_product':reaction['product_raw'],
                    'query_precursors':reaction['precursors_raw'],'query_family_key':reaction['reaction_family_key'],
                    'expected_retrieval':variant['expected_retrieval'],'observed_retrieval':observed,
                    'hits':[{'source_id':h['source_id'],'match_type':h['match_type']} for h in hits],
                    'passed':passed,'machine_only_tier':assess(reaction,hits)['tier']}
                results.append(result)
    finally:index.close()
    passed=sum(r['passed'] for r in results)
    return {'schema':'lgrv_independent_structure_policy_results_v1','total':len(results),
        'passed':passed,'failed':len(results)-passed,'all_passed':passed==len(results),
        'controls':results,'chemical_success_rate_established':False,
        'new_model_inference':False,'mechet_evidence_grade_supplied':False,
        'scope':'source-anchored structural retrieval controls; hypothetical variants are not performed reactions'}


def run_controls(spec_path,source_manifest,codec_file,codec_sha,freeze_root,output_root):
    """Bind every source, original frozen query and resulting control artifact."""
    import importlib.util
    from google.protobuf.json_format import MessageToDict
    from .literature_mapping import _authenticated_manifest
    spec_path,source_manifest,codec_file,freeze_root,output_root=map(Path,
        (spec_path,source_manifest,codec_file,freeze_root,output_root))
    spec=json.loads(spec_path.read_bytes());catalogue=json.loads(source_manifest.read_bytes())
    if catalogue.get('download_complete') is not True or file_sha256(codec_file)!=codec_sha:
        raise ValueError('complete source catalogue and pinned codec required')
    relative=Path(spec['ord_relative_path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe control source path')
    selected=[r for r in catalogue['files'] if r['path']==relative.as_posix()]
    source_path=source_manifest.parent/relative
    if len(selected)!=1 or file_sha256(source_path)!=selected[0]['sha256']:
        raise ValueError('control source does not match original catalogue bytes')
    module_spec=importlib.util.spec_from_file_location('lgrv_control_ord_codec',codec_file)
    codec=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(codec)
    def decode(payload):
        message=codec.Reaction();message.ParseFromString(payload)
        return MessageToDict(message,preserving_proto_field_name=True)
    required={r['ord_row']:r for r in spec['sources']}
    if len(required)!=len(spec['sources']):raise ValueError('duplicate independent source row')
    actual={};source_keys=set();source_count=0
    for number,dataset,raw in iter_ord_messages(source_path,decode):
        source_count+=1
        if number not in required:continue
        source=required[number]
        if dataset['dataset_id']!=spec['ord_dataset_id'] or raw['reaction_id']!=source['ord_reaction_id']:
            raise ValueError('independent source row identity differs')
        records=ord_records(raw,dataset=dataset,row_index=number)
        reaction=make_reaction(source['mapped_product'],[source['mapped_precursors']])
        matches=[r for r in records if make_reaction(r['product'],r['precursors'])['reaction_key']==reaction['reaction_key']]
        if not matches or reaction['mapping_status']!='mapped_center_available':
            raise ValueError('control mapped structures do not reconcile to actual source reaction')
        actual[source['id']]={'source_id':matches[0]['source_id'],'ord_row':number,
            'ord_reaction_id':raw['reaction_id'],'reaction_key':reaction['reaction_key'],
            'primary_locator':source['primary_locator'],'primary_url':spec['primary_url']}
        source_keys.add(reaction['reaction_key'])
    if len(actual)!=len(required):raise ValueError('independent control source rows missing')
    _,freeze_sha=_authenticated_manifest(freeze_root,'prediction_freeze_manifest.json','unique_reaction_queries.jsonl.gz')
    with gzip.open(freeze_root/'unique_reaction_queries.jsonl.gz','rt') as f:
        pool={r['reaction_key'] for r in map(json.loads,f)}
    overlap=sorted(source_keys&pool)
    output_root.mkdir(parents=True,exist_ok=False)
    result=evaluate_controls(spec,output_root/'control_sources.sqlite')
    result.update(source_key_overlap_with_frozen_mechet_pool=overlap,
        source_independence_passed=not overlap,source_rows_reconciled=actual,
        source_file_total_rows=source_count,policy_tuned_on_mechet_outcomes=False,
        primary_read_status=spec.get('primary_read_status','not documented'),
        universal_analogue_precision_established=False)
    result['all_passed']=result['all_passed'] and not overlap
    output=output_root/'policy_control_results.json';output.write_text(json.dumps(result,indent=2)+'\n')
    manifest={'schema':'lgrv_independent_structure_policy_manifest_v1','complete':True,
        'all_controls_passed':result['all_passed'],'control_spec_sha256':file_sha256(spec_path),
        'source_catalogue_sha256':file_sha256(source_manifest),'source_file_sha256':file_sha256(source_path),
        'source_file_path':str(source_path.resolve()),'codec_sha256':codec_sha,
        'prediction_freeze_manifest_sha256':freeze_sha,'new_model_inference':False,
        'producer_sha256':file_sha256(Path(__file__)),
        'output_files':{p.name:{'sha256':file_sha256(p)} for p in output_root.iterdir() if p.is_file()}}
    (output_root/'policy_control_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result,manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control-spec','source-manifest','codec-file','prediction-freeze-root','output-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--codec-sha256',required=True);a=p.parse_args()
    result,_=run_controls(a.control_spec,a.source_manifest,a.codec_file,a.codec_sha256,a.prediction_freeze_root,a.output_root)
    print(json.dumps({'phase':'independent_structure_controls_complete','total':result['total'],
        'passed':result['passed'],'failed':result['failed'],'source_independence':result['source_independence_passed']}),flush=True)
