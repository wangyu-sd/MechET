"""Executor-backed code verification; no model experiments or GT-guided search."""
def test_single_reaction_finish_returns_before_recursive_stock_expansion(tmp_path):
    from scripts.single_reaction_mcts import SingleReactionMCTS,SingleReactionMCTSStore
    from mechet.dynamic_electron_tree import Proposal
    from mechet.natural_language_electron_flow import render_event_arguments
    s=SingleReactionMCTSStore(tmp_path/'shared',{'reward':'productive_strict_finish_v1'})
    c=SingleReactionMCTS('case','CO',mcts_store=s,program_goal=1,checkpoint_path=tmp_path/'cp.json')
    for _ in range(100):
        context=c.next_context()
        if context is None:
            if not c.has_pending:break
            continue
        actions=context.node.actions
        if not actions:
            p=Proposal('import_fragments',{'fragments':[{'smiles':'[Cl-]','count':1,
                'purpose':'electron_participant'}]},'import',-1,4)
        elif actions[-1]['name']=='import_fragments':
            moves=[{'source':{'kind':'LP','atoms':[3]},'sink':{'kind':'BOND','atoms':[1,3]},'electrons':2},
                   {'source':{'kind':'BOND','atoms':[1,2]},'sink':{'kind':'ATOM','atoms':[2]},'electrons':2}]
            p=Proposal('apply_electron_flow',render_event_arguments(context.node.state,moves),'event',-1,4)
        else:p=Proposal('finish_trace',{},'finish',-1,3)
        c.incorporate(context.key,[p])
    assert not c.has_pending
    assert len(c.completed_programs)==1
    assert set(c.trees)=={'CO'}  # NO recursive precursor reactions/stock supplied
    assert c.mcts_metrics()['root_value_sum']==1.
    assert any(x.node.terminal for x in c.terminals())
    from mechet.shared_electron_search import read_record
    assert read_record(tmp_path/'cp.json')['search_complete'] is True
    c.close()


def test_import_finish_without_electron_event_does_not_earn_reward(tmp_path):
    from scripts.single_reaction_mcts import SingleReactionMCTS,SingleReactionMCTSStore
    from mechet.dynamic_electron_tree import Proposal
    c=SingleReactionMCTS('case','CO',mcts_store=SingleReactionMCTSStore(tmp_path,{}),program_goal=10)
    for _ in range(12):
        ctx=c.next_context()
        if ctx:c.incorporate(ctx.key,[Proposal('finish_trace',{},'finish',-1.,3)])
    assert c.has_pending and not c.completed_programs
    assert c.mcts_metrics()['root_value_sum']==0.
    c.close()


def test_single_reaction_widening_does_not_freeze_after_one_success(tmp_path):
    from scripts.single_reaction_mcts import SingleReactionMCTSStore
    s=SingleReactionMCTSStore(tmp_path,{})
    root=s.intern({'x':1});s.add_edges(root,[{'id':'x','child':None,'count':1}],observation='first')
    lease=s.start_rollout();s.reserve(lease.uid,root,'x');s.backup(lease.uid,[(root,'x')],1.);lease.close()
    assert s.needs_expansion(root)


def test_attempt_goal_counts_failed_rollouts_not_just_successes(tmp_path):
    from scripts.single_reaction_mcts import SingleReactionMCTS,SingleReactionMCTSStore
    from mechet.dynamic_electron_tree import Proposal
    c=SingleReactionMCTS('case','CO',mcts_store=SingleReactionMCTSStore(tmp_path,{}),program_goal=3)
    for _ in range(30):
        ctx=c.next_context()
        if ctx:c.incorporate(ctx.key,[Proposal('finish_trace',{},'finish',-1.,3)])
        if not c.has_pending:break
    assert not c.has_pending and c.completed_rollouts==3
    assert not c.rollouts  # no hidden speculative attempts beyond the requested budget
    assert c.positive_backups==0 and not c.completed_programs
    assert c.mcts_metrics()['root_visits']==3
    c.close()


def test_last_nonroot_context_censor_does_not_pop_empty_rollout_queue(tmp_path):
    from scripts.single_reaction_mcts import SingleReactionMCTS,SingleReactionMCTSStore
    from mechet.dynamic_electron_tree import Proposal
    c=SingleReactionMCTS('case','CO',mcts_store=SingleReactionMCTSStore(tmp_path,{}),program_goal=1)
    ctx=c.next_context()
    c.incorporate(ctx.key,[Proposal('import_fragments',{'fragments':[{'smiles':'[Cl-]','count':1,
        'purpose':'electron_participant'}]},'import',-1.,4)])
    ctx=c.next_context()
    assert c.selected['sid']!=c.mcts_root
    c.censor_selected('CONTEXT_MEMORY_BUDGET')
    assert c.next_context() is None
    assert not c.has_pending and c.censored_rollouts==1 and c.completed_rollouts==0
    assert not c.rollouts
    c.close()
