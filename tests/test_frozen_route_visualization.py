"""Renderer unit fixtures from executor calls, NEVER model-generated evidence."""
import copy
import xml.etree.ElementTree as ET
import pytest
from mechet.dynamic_electron_tree import Proposal
from mechet.reverse_route_network import RouteNetwork
from mechet.natural_language_electron_flow import render_event_arguments


def executed_case():
    net=RouteNetwork('UNIT_ONLY','CO',{'CCl','[OH-]'},mode='greedy')
    context=net.next_context()
    net.incorporate(context.key,[Proposal('import_fragments',{'fragments':[
        {'smiles':'[Cl-]','count':1,'purpose':'electron_participant'}]},'fixture-import',-1,4)])
    context=net.next_context()
    moves=[{'source':{'kind':'BOND','atoms':[1,2]},'sink':{'kind':'ATOM','atoms':[2]},'electrons':2},
           {'source':{'kind':'LP','atoms':[3]},'sink':{'kind':'BOND','atoms':[1,3]},'electrons':2}]
    arguments=render_event_arguments(context.node.state,moves)
    net.incorporate(context.key,[Proposal('apply_electron_flow',arguments,'fixture-flow',-2,8)])
    context=net.next_context()
    net.incorporate(context.key,[Proposal('finish_trace',{},'fixture-finish',-1,3)])
    assert net.report()['stock_solved']
    return net.to_dict(),net.report()


def test_saved_route_panels_use_actual_native_states_and_ground_recorded_arrows():
    from scripts.visualize_frozen_route_case import route_panels
    graph,report=executed_case();panels=route_panels(graph,report)
    assert [p['kind'] for p in panels]==['target','import','flow','reaction_endpoint']
    flow=next(p for p in panels if p['kind']=='flow')
    assert len(flow['moves'])==2
    assert '[Cl-' in flow['state']
    assert flow['moves'][0]['source']['kind']=='BOND'
    assert panels[-1]['state']==report['complete_reaction_edges'][0]['program']['full_terminal_state']


def test_case_reader_refuses_missing_native_state_instead_of_replaying_or_guessing():
    from scripts.visualize_frozen_route_case import route_panels
    graph,report=executed_case();broken=copy.deepcopy(report)
    del broken['complete_reaction_edges'][0]['program']['actions'][0]['state_before']
    with pytest.raises(ValueError,match='saved'):
        route_panels(graph,broken)


def test_svg_is_self_contained_and_has_real_rdkit_molecules_bonds_and_arrows():
    from scripts.visualize_frozen_route_case import render_case_svg
    graph,report=executed_case();svg=render_case_svg(graph,report)
    root=ET.fromstring(svg);ns={'s':'http://www.w3.org/2000/svg'}
    assert not root.findall('.//s:image',ns)  # no broken external molecule image paths
    assert len(root.findall('.//*[@data-electron-flow]',ns))==2
    assert root.findall('.//*[@data-molecule]',ns)
    assert root.findall('.//*[@data-stock-leaf]',ns)
    assert 'UNIT_ONLY' not in svg  # IDs belong in provenance, not paper labels


def test_already_stock_zero_step_is_not_drawn_as_a_generated_reaction_route():
    from scripts.visualize_frozen_route_case import render_case_svg
    net=RouteNetwork('UNIT_ONLY','CO',{'CO'},mode='tree')
    with pytest.raises(ValueError,match='nontrivial'):
        render_case_svg(net.to_dict(),net.report())


def test_svg_pdf_export_uses_existing_conversion_environment_not_model_packages(tmp_path):
    from scripts.visualize_frozen_route_case import render_case_svg,write_svg_pdf
    graph,report=executed_case()
    output=tmp_path/'UNIT_FIXTURE_NOT_MODEL_RESULT'
    write_svg_pdf(render_case_svg(graph,report),output,conversion_python='/root/miniconda3/bin/python')
    assert output.with_suffix('.pdf').read_bytes().startswith(b'%PDF-')
    ET.fromstring(output.with_suffix('.svg').read_text())


def test_historical_endpoint_subset_cannot_be_presented_as_full_route_case(tmp_path):
    import json
    from scripts.visualize_frozen_route_case import load_saved_case
    from scripts.infer_dynamic_electron_trees import ADAPTER_SHA
    folder=tmp_path/'tree/rank00';folder.mkdir(parents=True)
    (folder/'complete.json').write_text(json.dumps({'manifest':{'adapter_sha':ADAPTER_SHA,
        'full_denominator':128,'evaluation_unit':'single_reaction_endpoint'},'rows':[]}))
    with pytest.raises(ValueError,match='full-route'):
        load_saved_case(tmp_path,'tree','UNIT_ONLY')
