"""Source-bound drawings, not synthetic model performance experiments."""
import importlib
import xml.etree.ElementTree as ET

import pytest


def implementation():
    try:
        return importlib.import_module('mechet.analysis.literature_figures')
    except ModuleNotFoundError:
        pytest.fail('source-bound literature case renderer is not implemented')


def case_fixture():
    # Mutation targets: another terminal, changed raw actions, swapped reaction
    # identity, lost import count, or reversed/absent electron arrows.
    from mechet.analysis.literature_reaction_keys import make_reaction, stable_hash
    actions = [
        {'name':'import_fragments','state_before':'[CH3:1][OH:2]',
         'arguments':{'fragments':[{'smiles':'O','count':2,'purpose':'endpoint_context'}]},
         'result':{'ok':True,'current_state':'CO.O.O'}},
        {'name':'apply_electron_flow','state_before':'[CH3:1][OH:2].[OH2:3].[OH2:4]',
         'arguments':{'direction':'retrosynthetic','electron_flow':[
             {'source':'the bond between atoms A01 and A02','destination':'atom A02'}]},
         'result':{'ok':True,'current_state':'[CH3+].[OH-].O.O'}},
        {'name':'finish_trace','state_before':'[CH3+:1].[OH-:2].[OH2:3].[OH2:4]',
         'arguments':{},'result':{'ok':True,'derived_precursor':'[CH3+].[OH-].O.O'}}]
    context={'id':'end','terminal':True,'state':actions[-1]['state_before'],'actions':actions}
    tree={'case_id':'t','target':'CO','reaction_trees':{'CO':{'root_context':'root',
          'contexts':[{'id':'root','state':'[CH3:1][OH:2]'},context]}}}
    reaction=make_reaction('CO',['[CH3+]','[OH-]'])
    occurrence={**reaction,'run_id':'r','target_id':'t','occurrence_id':'o'}
    program={'run_id':'r','target_id':'t','reaction_key':reaction['reaction_key'],
             'program_record_id':'p','native_context_id':'end','reaction_product':'CO',
             'program_source_pointer':'/reaction_trees/CO/contexts/1/actions',
             'program_payload_sha256':stable_hash(actions),'exposure_class':'unseen_ordinary',
             'complete_signature':'fixture-only-not-a-result','actor_exposure_authenticated':True}
    route={'run_id':'r','target_id':'t','route_key':'route','reaction_keys':[reaction['reaction_key']],
           'stock_verified':True,'stock_leaves':['[CH3+]','[OH-]'],'already_in_stock':False}
    return route,[occurrence],[program],tree


def test_case_preparation_does_not_promote_unchecked_net_chemistry_or_arrows():
    route,occurrences,programs,tree=case_fixture()
    result=implementation().prepare_case(route,occurrences,programs,tree,[])
    assert result['precedent_covered'] is False
    assert result['physical_mechanism_verified'] is False
    assert result['steps'][0]['evidence']['tier']=='C'
    assert result['steps'][0]['cells'][0]['fragments'][0]['count']==2
    moves=result['steps'][0]['cells'][1]['moves']
    assert len(moves)==1
    assert moves[0]['source']=={'kind':'BOND','atoms':[1,2],'id':'BOND:1,2'}
    assert moves[0]['sink']=={'kind':'ATOM','atoms':[2],'id':'ATOM:2'}
    assert moves[0]['electrons']==2
    assert result['steps'][0]['cells'][-1]['state']=='[CH3+].[OH-]'


@pytest.mark.parametrize('mutation', ['payload','terminal','root','reaction','mixed_target','missing_edge'])
def test_case_preparation_rejects_unbound_programs_and_incomplete_route(mutation):
    route,occurrences,programs,tree=case_fixture()
    if mutation=='payload':tree['reaction_trees']['CO']['contexts'][1]['actions'][0]['arguments']['fragments'][0]['count']=3
    if mutation=='terminal':tree['reaction_trees']['CO']['contexts'][1]['terminal']=False
    if mutation=='root':tree['reaction_trees']['CO']['contexts'][0]['state']='[CH3:1][NH2:2]'
    if mutation=='reaction':occurrences[0]['structural_components']=['C','[OH-]']
    if mutation=='mixed_target':programs[0]['target_id']='other'
    if mutation=='missing_edge':route['reaction_keys'].append('absent')
    with pytest.raises(ValueError):
        implementation().prepare_case(route,occurrences,programs,tree,[])


def test_case_draws_exact_moves_imports_and_source_scope_as_editable_vectors():
    import yaml
    from pathlib import Path
    style=yaml.safe_load((Path(__file__).parents[1]/'configs/figures/mechet_nmi_tree_style_v1.yaml').read_text())
    route,occurrences,programs,tree=case_fixture()
    prepared=implementation().prepare_case(route,occurrences,programs,tree,[])
    main,details,metadata=implementation().render_case(prepared,style,title='Fixture only')
    ns={'s':'http://www.w3.org/2000/svg'}
    assert len(details)==1
    root=ET.fromstring(details[0]);flow=root.findall('.//*[@data-electron-flow]',ns)
    assert len(flow)==1
    assert flow[0].attrib['data-source-maps']=='1,2'
    assert flow[0].attrib['data-sink-maps']=='2'
    assert root.findall('.//*[@data-import-fragment]',ns)[0].attrib['data-count']=='2'
    assert len(metadata['steps'][0]['action_indices'])==3
    for svg in [main,*details]:
        root=ET.fromstring(svg)
        assert root.attrib['width']=='183mm'
        assert float(root.attrib['height'].removesuffix('mm'))<=170
        assert root.findall('.//s:path',ns)
        assert root.findall('.//s:text',ns)
        assert not root.findall('.//s:image',ns)
    texts=' '.join(n.text or '' for n in ET.fromstring(main).iter())
    assert 'Unresolved' in texts
    assert 'Tier A' not in texts
    assert 'Primary sources support net transformations' not in texts


def frozen_case_files(tmp_path):
    import gzip
    import json
    from mechet.analysis.literature_reaction_keys import stable_hash
    from mechet.analysis.literature_sources import file_sha256
    route,occurrences,programs,tree=case_fixture()
    rank=tmp_path/'rank00';(rank/'trees').mkdir(parents=True);(rank/'evaluations').mkdir()
    tree_path=rank/'trees/t.json';tree_path.write_text(json.dumps(tree))
    report_path=rank/'evaluations/t.json';report_path.write_text(json.dumps({
        'id':'t','target':'CO','tree_artifact':'trees/t.json'}))
    source={'run_id':'r','target_id':'t','report_path':str(report_path),
            'report_sha256':file_sha256(report_path),'actor_sha256':'b'*64}
    route['source']=source.copy();occurrences[0]['source']=source.copy()
    freeze=tmp_path/'freeze';freeze.mkdir();files={}
    for name,records in [('generated_reaction_steps',occurrences),('frozen_routes',[route])]:
        path=freeze/(name+'.jsonl.gz')
        with gzip.open(path,'wt') as handle:
            for record in records:handle.write(json.dumps(record)+'\n')
        files[path.name]={'sha256':file_sha256(path)}
    manifest=freeze/'prediction_freeze_manifest.json'
    # Real source inventory carries additional tree/provenance fields absent
    # from a route's projection. Identity equality is not whole-dict equality.
    inventory_source={**source,'target_tree_sha256':file_sha256(tree_path)}
    manifest.write_text(json.dumps({'complete':True,'actor_sha256':'b'*64,'source_reports':[inventory_source],
                                    'output_files':files}))
    exposure=tmp_path/'exposure.json';exposure.write_text(json.dumps({
        'model_sha256':'b'*64,'signature_version':'source_to_sink_execution_moves_v1',
        'actual_train_exposure':{'program_counts':{},'primitive_counts':{}}}))
    spec=tmp_path/'case.json';spec.write_text(json.dumps({
        'title':'Fixture only','run_id':'r','target_id':'t','route_key':'route',
        'prediction_freeze_manifest_sha256':file_sha256(manifest),
        'native_tree_sha256':file_sha256(tree_path),
        'actor_exposure_manifest_sha256':file_sha256(exposure),
        'program_record_ids':[stable_hash(['r','t','CO','end'])]}))
    return spec,freeze,exposure,report_path


def test_case_cli_exports_hash_bound_vectors_and_rejects_mutated_report(tmp_path):
    from pathlib import Path
    try:
        run=importlib.import_module('scripts.render_nmi_literature_figures').render_selected_case
    except ModuleNotFoundError:
        pytest.fail('literature figure delivery CLI is not implemented')
    spec,freeze,exposure,report=frozen_case_files(tmp_path)
    style=Path(__file__).parents[1]/'configs/figures/mechet_nmi_tree_style_v1.yaml'
    manifest=run(spec,freeze,exposure,[],style,tmp_path/'out')
    assert len(manifest['figures'])==2
    assert all(f['qa']['machine_checks_passed'] for f in manifest['figures'])
    assert manifest['physical_mechanism_verified'] is False
    assert (tmp_path/'out/captions.md').is_file()
    assert 'This is a source-checked case' not in (tmp_path/'out/captions.md').read_text()
    report.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        run(spec,freeze,exposure,[],style,tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()


def test_contradiction_source_is_not_captioned_as_positive_reaction_precedent(tmp_path):
    from pathlib import Path
    import json
    from test_nmi_literature_protocol import primary
    from scripts.render_nmi_literature_figures import render_selected_case
    spec,freeze,exposure,report=frozen_case_files(tmp_path)
    route,occurrences,programs,tree=case_fixture()
    record=primary(tmp_path,contradicted_reaction_key=occurrences[0]['reaction_key'],
        contradiction_class='fixture_only',contradiction_evidence='Synthetic software-test contradiction, not empirical chemistry')
    registry=tmp_path/'registry.json';registry.write_text(json.dumps({'records':[record]}))
    style=Path(__file__).parents[1]/'configs/figures/mechet_nmi_tree_style_v1.yaml'
    manifest=render_selected_case(spec,freeze,exposure,[registry],style,tmp_path/'out')
    text=(tmp_path/'out/captions.md').read_text()
    assert manifest['precedent_covered'] is False
    assert 'documented contradiction' in text.lower()
    assert 'Exact/analogue net-transformation evidence: Tier D' not in text


def test_net_precedent_gallery_draws_only_checked_frozen_generated_pairs(tmp_path):
    import yaml
    from pathlib import Path
    from test_nmi_literature_protocol import primary
    from mechet.analysis.literature_reaction_keys import make_reaction
    reaction=make_reaction('CO',['CBr','[OH-]']);source=primary(tmp_path)
    style=yaml.safe_load((Path(__file__).parents[1]/'configs/figures/mechet_nmi_tree_style_v1.yaml').read_text())
    renderer=getattr(implementation(),'render_net_precedent_gallery',None)
    assert renderer is not None,'source-checked complete-reaction gallery is missing'
    svg,metadata=renderer([reaction],[source],style,reaction_keys=[reaction['reaction_key']])
    root=ET.fromstring(svg);ns={'s':'http://www.w3.org/2000/svg'}
    assert root.findall('.//s:path',ns) and not root.findall('.//s:image',ns)
    text=' '.join(n.text or '' for n in root.iter())
    assert 'Tier A' in text and source['source_id'] in text
    assert 'selected cases' in text.lower()
    assert metadata['reaction_keys']==[reaction['reaction_key']]
    assert metadata['physical_mechanism_verified'] is False
    source['document_sha256']='0'*64
    with pytest.raises(ValueError):renderer([reaction],[source],style,reaction_keys=[reaction['reaction_key']])
    with pytest.raises(ValueError):renderer([],[],style,reaction_keys=[reaction['reaction_key']])
