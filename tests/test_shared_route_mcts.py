"""CPU-only MCTS algorithm/contract tests; no new model experiments."""
import pytest


def store(tmp_path,stock='stock-a'):
    from scripts.shared_route_mcts import SharedMCTSStore
    return SharedMCTSStore(tmp_path,{'actor':'fixed','stock_sha':stock})


def test_shared_edge_values_and_exactly_once_path_backup(tmp_path):
    s=store(tmp_path);root=s.intern({'position':'root'});child=s.intern({'position':'child'})
    s.add_edges(root,[{'id':'a','child':child,'count':2},{'id':'b','child':None,'count':1}],observation='batch0')
    lease=s.start_rollout();s.reserve(lease.uid,root,'a')
    s.backup(lease.uid,[(root,'a')],1.)
    s.backup(lease.uid,[(root,'a')],1.)
    second=store(tmp_path);record=second.snapshot(root)
    assert record['visits']==1
    assert record['edges']['a']['n']==1
    assert record['edges']['a']['w']==1.
    assert record['edges']['b']['n']==0
    assert not record['reservations']
    lease.close()
    with pytest.raises(ValueError,match='backup'):
        s.backup(lease.uid,[(root,'b')],0.)


def test_policy_batch_reuse_does_not_inflate_priors(tmp_path):
    s=store(tmp_path);root=s.intern({'x':1})
    edges=[{'id':'a','child':None,'count':3}]
    s.add_edges(root,edges,observation='same')
    s.add_edges(root,edges,observation='same')
    assert s.snapshot(root)['edges']['a']['count']==3


def test_duplicate_only_widening_yields_to_selection(tmp_path):
    s=store(tmp_path);root=s.intern({'x':1})
    edges=[{'id':'a','child':None,'count':1}]
    s.add_edges(root,edges,observation='first')
    lease=s.start_rollout();s.reserve(lease.uid,root,'a')
    s.backup(lease.uid,[(root,'a')],0.);lease.close()
    assert s.needs_expansion(root)
    s.add_edges(root,edges,observation='duplicates')
    assert not s.needs_expansion(root)
    assert s.select_edge(root)['id']=='a'


def test_checkpoint_restores_native_graph_and_censors_crashed_rollout(tmp_path):
    from scripts.shared_route_mcts import MCTSRouteNetwork
    from mechet.shared_electron_search import SearchJournal
    from mechet.dynamic_electron_tree import Proposal
    path=tmp_path/'checkpoint.json';s=store(tmp_path)
    c=MCTSRouteNetwork('case','CO',{'C'},mcts_store=s,checkpoint_path=path)
    j=SearchJournal(tmp_path/'journals','case',{'stock_sha':'stock-a'})
    j.initialize(c)
    context=c.next_context()
    proposals=[Proposal('import_fragments',{'fragments':[{'smiles':'[Cl-]','count':1,
                            'purpose':'electron_participant'}]},'import',-1.,4)]
    edges=c.incorporate(context.key,proposals);j.record(c,context.key,edges)
    c.close()  # simulate released OS leases after a crash; saved paths remain
    restored=MCTSRouteNetwork.restore(j,stock={'C'},mcts_store=s,checkpoint_path=path)
    assert len(restored.contexts)==len(c.contexts)
    assert restored.censored_rollouts>=1
    assert restored.completed_rollouts==0
    assert restored.has_pending
    assert restored.next_context() is not None
    restored.close()


def test_crashed_rollout_releases_virtual_visits_without_failure_backup(tmp_path):
    s=store(tmp_path);root=s.intern({'x':1})
    s.add_edges(root,[{'id':'a','child':None,'count':1}],observation='b')
    lease=s.start_rollout();s.reserve(lease.uid,root,'a');lease.close()
    record=s.snapshot(root)
    assert not record['reservations']
    assert record['visits']==0 and record['edges']['a']['n']==0


def test_different_stock_cannot_share_mcts_values(tmp_path):
    a=store(tmp_path);b=store(tmp_path,'stock-b')
    assert a.root!=b.root


def test_puct_explores_after_a_failed_high_prior_action(tmp_path):
    s=store(tmp_path);root=s.intern({'x':1})
    s.add_edges(root,[{'id':'a','child':None,'count':3},{'id':'b','child':None,'count':1}],observation='batch')
    for _ in range(8):
        lease=s.start_rollout();s.reserve(lease.uid,root,'a')
        s.backup(lease.uid,[(root,'a')],0.);lease.close()
    assert s.select_edge(root)['id']=='b'


def test_finish_boundary_retains_all_nonstock_obligations_and_rejects_cycles(tmp_path):
    from scripts.shared_route_mcts import finish_obligations
    # This tests planning semantics, not chemical feasibility of a reaction.
    remaining=finish_obligations('CCO',['C','CO'],[],(),{'C'})
    assert remaining==[('CO',('CCO',))]
    with pytest.raises(ValueError,match='cycle'):
        finish_obligations('CO',['CCO'],[],('CCO',),{'C'})
    assert finish_obligations('CO',['C','O'],[],('CCO',),{'C','O'})==[]


def test_planning_transpositions_keep_pending_ancestry_and_native_history(tmp_path):
    s=store(tmp_path)
    a=s.intern({'active':'context-a','pending':[('CO',[])]})
    assert a==s.intern({'active':'context-a','pending':[('CO',[])]})
    assert a!=s.intern({'active':'context-b','pending':[('CO',[])]})
    assert a!=s.intern({'active':'context-a','pending':[('C',[])]})
    assert a!=s.intern({'active':'context-a','pending':[('CO',['CCO'])]})


def test_controller_import_and_finish_is_not_a_stock_solution(tmp_path):
    from scripts.shared_route_mcts import MCTSRouteNetwork
    from mechet.dynamic_electron_tree import Proposal
    c=MCTSRouteNetwork('case','CCO',{'C','O'},mcts_store=store(tmp_path))
    context=c.next_context()
    c.incorporate(context.key,[Proposal('import_fragments',{'fragments':[
        {'smiles':'O','count':1,'purpose':'electron_participant'}]},'import',-1,3)])
    # Selection may revisit root to expand another alternative; test that an
    # import/finish rollout never becomes a completed reaction or positive return.
    for _ in range(12):
        context=c.next_context()
        if context is None:break
        c.incorporate(context.key,[Proposal('finish_trace',{},'finish',-1,3)])
    assert not c.report()['stock_solved']
    assert c.mcts_metrics()['positive_backups']==0
    c.close()


def test_real_executor_rollout_only_rewards_full_route_after_finish(tmp_path):
    from scripts.shared_route_mcts import MCTSRouteNetwork
    from mechet.dynamic_electron_tree import Proposal
    from mechet.natural_language_electron_flow import render_event_arguments
    c=MCTSRouteNetwork('case','CO',{'CCl','[OH-]'},mcts_store=store(tmp_path),
                       checkpoint_path=tmp_path/'case.json')
    for _ in range(100):
        context=c.next_context()
        if context is None:
            if not c.has_pending:break
            continue
        actions=context.node.actions
        if not actions:
            p=Proposal('import_fragments',{'fragments':[{'smiles':'[Cl-]','count':1,
                       'purpose':'electron_participant'}]},'import chloride',-1,4)
        elif actions[-1]['name']=='import_fragments':
            moves=[{'source':{'kind':'LP','atoms':[3]},'sink':{'kind':'BOND','atoms':[1,3]},'electrons':2},
                   {'source':{'kind':'BOND','atoms':[1,2]},'sink':{'kind':'ATOM','atoms':[2]},'electrons':2}]
            p=Proposal('apply_electron_flow',render_event_arguments(context.node.state,moves),'reverse SN2',-1,4)
        else:
            p=Proposal('finish_trace',{},'finish',-1,3)
        c.incorporate(context.key,[p])
        assert c.positive_backups==0  # expansion itself must never fabricate a rollout return
    assert c.report()['stock_solved']
    assert c.mcts_metrics()['positive_backups']==1
    assert c.mcts_metrics()['root_value_sum']==1.
    assert sorted(c.report()['stock_leaves'])==['CCl','[OH-]']
    assert c.first_solution_work['rollout_path']
    from mechet.shared_electron_search import read_record
    checkpoint=read_record(tmp_path/'case.json')
    assert checkpoint['search_complete'] is True
    assert c.to_dict()['mcts_graph']
    c.close()


def test_recover_half_applied_backup_does_not_double_count_first_node(tmp_path):
    from mechet.shared_electron_search import durable_json
    s=store(tmp_path);root=s.intern({'p':'root'});child=s.intern({'p':'child'})
    s.add_edges(root,[{'id':'a','child':child,'count':1}],observation='r')
    s.add_edges(child,[{'id':'b','child':None,'count':1}],observation='c')
    lease=s.start_rollout()
    record={'uid':lease.uid,'path':[[root,'a'],[child,'b']],'reward':1.}
    durable_json(s.root/'backups'/(lease.uid+'.json'),record)
    with s.edit(root) as value:
        value['visits']=1;value['value_sum']=1.
        value['edges']['a']['n']=1;value['edges']['a']['w']=1.
        value['backed_up'][lease.uid]=['a',1.]
    lease.close()
    recovered=store(tmp_path)
    assert recovered.snapshot(root)['visits']==1
    assert recovered.snapshot(child)['visits']==1
    assert recovered.snapshot(child)['edges']['b']['w']==1.


def test_parallel_backups_do_not_lose_updates(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    s=store(tmp_path);root=s.intern({'p':'r'})
    s.add_edges(root,[{'id':'a','child':None,'count':1}],observation='r')
    def trial(index):
        lease=s.start_rollout();s.reserve(lease.uid,root,'a')
        s.backup(lease.uid,[(root,'a')],float(index%2));lease.close()
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(trial,range(8)))
    value=s.snapshot(root)
    assert value['visits']==8 and value['value_sum']==4.
    assert value['edges']['a']['n']==8 and value['edges']['a']['w']==4.


def test_busy_expansion_is_deferred_and_recovers_after_owner_closes(tmp_path):
    s=store(tmp_path);root=s.intern({'x':1})
    first=s.claim_expansion(root)
    assert store(tmp_path).claim_expansion(root) is None
    first.close()
    retry=store(tmp_path).claim_expansion(root)
    assert retry is not None
    retry.close()


def test_uncertified_success_never_enters_shared_positive_values(tmp_path):
    from scripts.shared_route_mcts import MCTSRouteNetwork
    s=store(tmp_path);c=MCTSRouteNetwork('case','CO',{'C'},mcts_store=s)
    s.add_edges(c.mcts_root,[{'id':'a','child':None,'count':1}],observation='unit')
    lease=s.start_rollout();rollout={'lease':lease,'sid':c.mcts_root,'path':[(c.mcts_root,'a')]}
    with pytest.raises(ValueError,match='certificate'):c._complete(rollout,1.)
    assert s.snapshot(c.mcts_root)['visits']==0
    lease.close();c.close()


def test_real_two_reaction_rollout_crosses_finish_and_closes_every_branch(tmp_path):
    from scripts.shared_route_mcts import MCTSRouteNetwork
    from mechet.dynamic_electron_tree import Proposal
    from mechet.natural_language_electron_flow import render_event_arguments
    c=MCTSRouteNetwork('case','CO',{'CBr','[OH-]','[Cl-]'},mcts_store=store(tmp_path))
    saw_second_reaction=False
    for _ in range(200):
        context=c.next_context()
        if context is None:
            if not c.has_pending:break
            continue
        actions=context.node.actions
        second=context.node.target=='CCl'
        if second:
            saw_second_reaction=True
            assert c.positive_backups==0
        if not actions:
            fragment='[Br-]' if second else '[Cl-]'
            p=Proposal('import_fragments',{'fragments':[{'smiles':fragment,'count':1,
                'purpose':'electron_participant'}]},'import',-1,4)
        elif actions[-1]['name']=='import_fragments':
            moves=[{'source':{'kind':'LP','atoms':[3]},'sink':{'kind':'BOND','atoms':[1,3]},'electrons':2},
                   {'source':{'kind':'BOND','atoms':[1,2]},'sink':{'kind':'ATOM','atoms':[2]},'electrons':2}]
            p=Proposal('apply_electron_flow',render_event_arguments(context.node.state,moves),'flow',-1,4)
        else:p=Proposal('finish_trace',{},'finish',-1,3)
        c.incorporate(context.key,[p])
    assert saw_second_reaction and c.report()['stock_solved']
    assert len(c.report()['complete_reaction_edges'])==2
    assert sorted(c.report()['stock_leaves'])==['CBr','[Cl-]','[OH-]']
    assert len(c.first_solution_work['rollout_path'])==6
    c.close()


def test_host_resource_rejection_does_not_become_shared_failed_action(tmp_path):
    from scripts.shared_route_mcts import MCTSRouteNetwork
    from mechet.dynamic_electron_tree import Proposal
    s=store(tmp_path);c=MCTSRouteNetwork('case','CO',{'C'},mcts_store=s)
    context=c.next_context()
    c.incorporate(context.key,[Proposal('import_fragments',{},'resource error',-1,3,
        error='STATE_ALLOCATION_RESOURCE_FAILURE')])
    assert not s.snapshot(c.mcts_root)['edges']
    assert s.snapshot(c.mcts_root)['visits']==0
    assert c.has_pending
    c.close()


def test_root_context_limit_finishes_as_censored_not_a_failed_rollout(tmp_path):
    from scripts.shared_route_mcts import MCTSRouteNetwork
    c=MCTSRouteNetwork('case','CO',{'C'},mcts_store=store(tmp_path))
    context=c.next_context();context.resource_closed='CONTEXT_MEMORY_BUDGET'
    c.censor_selected('CONTEXT_MEMORY_BUDGET')
    assert not c.has_pending
    assert c.mcts_metrics()['completed_rollouts']==0
    c.close()


def test_inert_wording_is_merged_but_real_atom_addresses_are_not():
    from scripts.shared_route_mcts import planning_edge_id
    from mechet.dynamic_electron_tree import Proposal
    a=Proposal('apply_electron_flow',{'site':'A01','instruction':'first wording'},'',-1,1)
    b=Proposal('apply_electron_flow',{'site':'A01','instruction':'other wording'},'',-2,2)
    c=Proposal('apply_electron_flow',{'site':'A02','instruction':'first wording'},'',-1,1)
    assert planning_edge_id('parent','child',a,'')==planning_edge_id('parent','child',b,'')
    assert planning_edge_id('parent','child',a,'')!=planning_edge_id('parent','child',c,'')
