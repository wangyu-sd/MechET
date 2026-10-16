"""AND-route controller semantics; fixture states are not model predictions."""
from scripts.run_natural_language_value_search import Node
from mechet.dynamic_electron_tree import SearchContext, chemical_key, digest


def terminal(state, target):
    # A boundary fixture for an already accepted native terminal. Tests below
    # verify closure/ancestry, not the physical plausibility of this fixture.
    action={'name':'apply_electron_flow','arguments':{},'result':{'ok':True,'code':'PASS'}}
    node=Node(target,state,10,actions=[action],terminal=True,logprob=-2,tokens=4)
    return SearchContext(digest([state,target]),node,digest(chemical_key(state)))


def test_every_precursor_branch_must_close_not_just_one_stock_leaf():
    from mechet.reverse_route_network import stock_closure
    edges=[{'id':'r','product':'CCO','precursors':['C','CO'],'cost':1,'tokens':1}]
    assert 'CCO' not in stock_closure({'C'},edges)
    edges.append({'id':'s','product':'CO','precursors':['C','O'],'cost':1,'tokens':1})
    closure=stock_closure({'C','O'},edges)
    assert closure['CCO']['cost'] == 2
    assert closure['CCO']['edge'] == 'r'


def test_cycles_do_not_create_stock_connected_certificates():
    from mechet.reverse_route_network import stock_closure
    edges=[{'id':'a','product':'CCO','precursors':['CO'],'cost':1,'tokens':1},
           {'id':'b','product':'CO','precursors':['CCO'],'cost':1,'tokens':1}]
    assert set(stock_closure({'O'},edges)) == {'O'}


def test_already_stock_targets_are_not_nontrivial_model_solutions():
    from mechet.reverse_route_network import RouteNetwork
    net=RouteNetwork('target','CO',{'CO'},mode='tree')
    assert net.next_context() is None
    report=net.report()
    assert report['already_in_stock'] is True
    assert report['stock_solved'] is False
    assert report['complete_reaction_edges'] == []


def test_terminal_recursively_expands_only_structural_not_environment_fragments():
    from mechet.reverse_route_network import RouteNetwork
    net=RouteNetwork('target','CCO',{'C'},mode='tree')
    net._register_terminal('CCO',terminal('[CH4:1].[CH3:2][OH:3].[OH2:4]','CCO'))
    assert set(net.trees) == {'CCO','CO'}
    assert net.reaction_edges[0]['precursors'] == ['C','CO']
    assert net.reaction_edges[0]['auxiliary_fragments']
    assert not net.report()['stock_solved']
    context=net.next_context()
    assert context is not None
    assert context.node.target in {'CCO','CO'}


def test_two_recursive_electronic_terminals_form_complete_and_route():
    from mechet.reverse_route_network import RouteNetwork
    net=RouteNetwork('target','CCO',{'C','O'},mode='tree')
    net._register_terminal('CCO',terminal('[CH4:1].[CH3:2][OH:3]','CCO'))
    net._register_terminal('CO',terminal('[CH4:1].[OH2:2]','CO'))
    result=net.report()
    assert result['stock_solved']
    assert len(result['complete_reaction_edges']) == 2
    assert sorted(result['stock_leaves']) == ['C','C','O']
    assert net.next_context() is not None  # first solution does not stop tree search


def test_import_and_finish_without_electronic_transform_cannot_solve_route():
    from mechet.reverse_route_network import RouteNetwork
    from mechet.dynamic_electron_tree import Proposal
    net=RouteNetwork('target','CCO',{'C','O'},mode='greedy')
    context=net.next_context()
    net.incorporate(context.key,[Proposal('import_fragments',{'fragments':[
        {'smiles':'O','count':1,'purpose':'electron_participant'}]},'import',-1,3)])
    context=net.next_context()
    net.incorporate(context.key,[Proposal('finish_trace',{},'finish',-1,3)])
    assert net.next_context() is None
    assert not net.report()['stock_solved']
    assert net.rejected_route_terminals[0]['reason'] == 'NONPRODUCTIVE_REACTION'


def test_and_closure_has_no_fixed_synthesis_depth_cutoff():
    from mechet.reverse_route_network import stock_closure
    edges=[{'id':f'r{i}','product':'C'*(i+2),'precursors':['C'*(i+1)],'cost':1,'tokens':1}
           for i in range(50)]
    result=stock_closure({'C'},edges)
    assert result['C'*51]['cost'] == 50


def test_batched_greedy_scheduler_continues_to_next_precursor_obligation():
    from mechet.reverse_route_network import RouteNetwork
    from scripts.infer_dynamic_electron_trees import select_context
    net=RouteNetwork('target','CCO',{'C'},mode='greedy')
    net._register_terminal('CCO',terminal('[CH4:1].[CH3:2][OH:3]','CCO'))
    # The parent reaction's one greedy trajectory has ended. Its unresolved
    # precursor must still be sampled by the shared batched scheduler.
    net.trees['CCO'].contexts[net.forced['CCO']].resource_closed='GREEDY_TERMINAL'
    context=select_context(net,'greedy',None)
    assert context is not None
    assert context.node.target == 'CO'
