from copy import deepcopy
import pytest


def test_fork_preserves_exact_prompt_prefix_and_restores_new_receipt_namespace():
    from mechet.route_refit import RefitRouteEpisode
    from test_route_first_rejection import episode
    from test_route_native_observation import flow,send
    base=episode();ep=RefitRouteEpisode(base.cfg,'CO',set(),'root',17)
    ep.next_decision();send(ep,'apply_electron_flow',flow(ep))
    original=deepcopy(ep.snapshot());request=ep.next_decision()
    fork=ep.fork('sibling-1')
    assert fork.records==ep.records
    assert fork.next_decision().messages==request.messages
    assert fork.next_decision().request_id!=request.request_id
    send(fork,'unknown',{})
    assert ep.snapshot()==original and ep.status is None
    assert fork.status=='rejected_action'
    assert RefitRouteEpisode.restore(fork.snapshot()).snapshot()==fork.snapshot()


def conditioning():
    from mechet.route_refit import decision_identity
    return decision_identity(target_id='target',tree_id='tree',occurrence='0',
        prompt_token_ids=[1,2],tools=[{'name':'FLOW'}],accepted_history=[],
        route_context={'pending':['C']},behavior_version='b0',executor_contract='v1',
        mapped_state='[CH4:1]')


def test_conditioning_never_merges_histories_prompt_or_occurrences():
    from mechet.route_refit import decision_identity
    import inspect
    args=dict(target_id='target',tree_id='tree',occurrence='0',prompt_token_ids=[1,2],
        tools=[{'name':'FLOW'}],accepted_history=[],route_context={'pending':['C']},
        behavior_version='b0',executor_contract='v1',mapped_state='[CH4:1]')
    key=decision_identity(**args)
    for field,value in [('prompt_token_ids',[2,1]),('occurrence','0.1'),
                         ('accepted_history',[{'name':'FLOW'}]),('behavior_version','b1'),
                         ('route_context',{'pending':['O']}),('mapped_state','[CH4:2]')]:
        assert decision_identity(**dict(args,**{field:value}))!=key


def test_rejected_edge_is_terminal_but_parent_remains_open_and_duplicates_skip_executor():
    from mechet.route_refit import DecisionNode
    node=DecisionNode(conditioning(),max_expansions=3,max_duplicates=2)
    node.add('bad','r0',error='SOURCE_MISSING')
    assert node.status=='OPEN_WITH_REJECTIONS'
    assert node.edges['bad']['status']=='REJECTED_TERMINAL'
    assert node.edges['bad']['successor'] is None
    assert node.known_rejection('bad')=='SOURCE_MISSING'
    node.add('bad','r1',error='SOURCE_MISSING',executor_skipped=True)
    assert node.expansions==1 and node.duplicate_draws==1
    node.add('good','r2',successor='child')
    assert node.edges['good']['successor']=='child'
    assert node.known_rejection('good') is None


def test_resource_draw_is_censored_and_does_not_chemically_exhaust_parent():
    from mechet.route_refit import DecisionNode
    node=DecisionNode(conditioning(),max_expansions=1,max_duplicates=1)
    node.add('unknown','r0',resource=True)
    assert node.status=='OPEN' and node.expansions==0
    assert node.known_rejection('unknown') is None


def test_conditional_credit_mean_centers_semantic_actions_not_draw_multiplicity():
    from mechet.route_refit import branch_advantages
    rows=[dict(action='a',receipt_id='r0',value=1.,complete=True,proposal='original_policy'),
          dict(action='a',receipt_id='r1',value=1.,complete=True,proposal='original_policy'),
          dict(action='b',receipt_id='r2',value=0.,complete=True,proposal='original_policy')]
    result=branch_advantages(rows,matched_continuations=1)
    assert result=={'r0':.5,'r1':.5,'r2':-.5}
    for broken in [dict(rows[2],value=None),dict(rows[2],complete=False),
                   dict(rows[2],proposal='deterministic_puct')]:
        assert branch_advantages(rows[:2]+[broken],matched_continuations=1) is None
    assert branch_advantages(rows[:2],matched_continuations=1) is None


def test_unique_credit_replaces_g4_and_never_duplicates_shared_prefix():
    from mechet.route_refit import unique_credit
    rec=dict(request_id='prefix',prompt_token_ids=[1],generated_token_ids=[2,3],
             behavior_logprobs=[-.1,-.2])
    branch=dict(rec,request_id='branch')
    output=unique_credit([(rec,.2),(deepcopy(rec),.2),(branch,.3)],{'branch':-.5})
    assert [(r['request_id'],a) for r,a in output]==[('prefix',.2),('branch',-.5)]
    with pytest.raises(ValueError,match='RECEIPT'):
        unique_credit([(rec,.2),(dict(rec,generated_token_ids=[4]),.2)],{})


def test_grounded_semantics_ignore_explanation_but_preserve_move_order():
    import json
    from mechet.route_refit import grounded_action_key
    from test_route_native_observation import episode,flow
    ep=episode();ep.next_decision();args=flow(ep)
    def raw(a):return '<tool_call>'+json.dumps(dict(name='apply_electron_flow',arguments=a))+'</tool_call>'
    key=grounded_action_key(ep.node.state,raw(args))
    changed=deepcopy(args);changed['electron_flow'][0]['instruction']='different explanation'
    assert grounded_action_key(ep.node.state,raw(changed))==key
    changed['electron_flow'][0]['destination']='atom A99'
    assert grounded_action_key(ep.node.state,raw(changed))!=key


def test_failed_duplicate_real_draw_is_saved_but_executor_not_repeated():
    import json
    from mechet.route_refit import RefitRouteEpisode,grounded_action_key
    from test_route_first_rejection import episode
    from test_route_native_observation import send
    base=episode();ep=RefitRouteEpisode(base.cfg,'CO',set(),'root',17)
    ep.next_decision();raw='<tool_call>'+json.dumps(dict(name='finish_trace',arguments={}))+'</tool_call>'
    ep.set_rejections({grounded_action_key(ep.node.state,raw):'NONPRODUCTIVE_FINISH'})
    original=ep.chemistry.step
    ep.chemistry.step=lambda *a: (_ for _ in ()).throw(AssertionError('duplicate Executor call'))
    try:send(ep,'finish_trace',{})
    finally:ep.chemistry.step=original
    assert ep.records[-1]['executor_skipped']=='repeated_rejected_action'
    assert ep.records[-1]['generated_token_ids']==[2]
    assert ep.status=='rejected_action'


def test_pending_sibling_reserves_parent_slot_and_cannot_reuse_request_id():
    from mechet.route_refit_runtime import RefitCollection
    from mechet.route_refit import RefitRouteEpisode
    from test_route_first_rejection import episode
    from test_route_native_observation import flow,send
    base=episode();ep=RefitRouteEpisode(base.cfg,'CO',set(),'root',17)
    controller=RefitCollection([ep],dict(max_nodes_per_tree=1,max_expansions=2,max_duplicates=2,
                                      max_forks_per_tree=4,matched_continuations=1))
    root=controller.initial[0];root.next_decision()
    send(root,'apply_electron_flow',flow(root))
    pending=controller.completed(root)
    assert len(pending)==1
    first=pending[0].next_decision().request_id
    # The original suffix fails at an unretained descendant while the parent
    # already has an outstanding draw. Backtracking must not dispatch twice.
    send(root,'unknown',{})
    assert controller.completed(root)==[]
    assert controller.owner(first) is pending[0]


def test_parallel_worker_candidates_cannot_overshoot_retained_node_ceiling():
    from mechet.route_refit_runtime import RefitCollection
    from mechet.route_refit import RefitRouteEpisode
    from test_route_first_rejection import episode
    from test_route_native_observation import flow,send
    base=episode();ep=RefitRouteEpisode(base.cfg,'CO',set(),'root',17)
    controller=RefitCollection([ep],dict(max_nodes_per_tree=2,max_expansions=2,max_duplicates=2,
                                      max_forks_per_tree=5,matched_continuations=1))
    original=controller.initial[0];original.next_decision();args=flow(original)
    send(original,'apply_electron_flow',args);sibling=controller.completed(original)[0]
    args['electron_flow'][0]['instruction']='same chemical event, another explanation'
    send(sibling,'apply_electron_flow',args);controller.completed(sibling)
    # Both workers retain a prospective node before either result is harvested.
    for candidate in (original,sibling):send(candidate,'unknown',{})
    for candidate in (original,sibling):controller.completed(candidate)
    assert len(controller.nodes)==2


@pytest.mark.parametrize('collector_name',['rolling','async'])
def test_rolling_refit_collector_backtracks_without_changing_original_prompt(tmp_path,collector_name):
    import json
    from mechet.route_refit_runtime import RefitCollection
    from mechet.route_refit import RefitRouteEpisode
    from mechet.route_policy_sampling import collect_route_groups
    if collector_name=='async':
        from mechet.route_async_collection import collect_async_route_groups as collect_route_groups
    from test_route_first_rejection import episode
    from test_route_native_observation import flow
    base=episode();ep=RefitRouteEpisode(base.cfg,'CO',{'[CH3+]','[OH-]'},'root',17)
    controller=RefitCollection([ep],dict(max_nodes_per_tree=4,max_expansions=2,max_duplicates=2,
                                      max_forks_per_tree=4,matched_continuations=1))
    class Sampler:
        cfg=dict(generation_batch_size=8,executor_coalesce_seconds=0.)
        def generate(self,requests):
            out=[]
            for request in requests:
                owner=next(e for e in controller.all if
                    (e.ep.cfg.get('refit_request_namespaces') or [{'namespace':e.ep.uid}])[-1]['namespace']==request.episode_id)
                if not owner.ep.node.actions:
                    args=flow(owner.ep);name='apply_electron_flow'
                    if '/fork-' in request.request_id:args['electron_flow'][0]['source']='the bond between atoms A99 and A98'
                else:name='finish_trace';args={}
                out.append(dict(request_id=request.request_id,behavior_version=request.behavior_version,
                    prompt_token_ids=[1,2],generated_token_ids=[3,4],behavior_logprobs=[-.1,-.2],
                    raw='<tool_call>'+json.dumps(dict(name=name,arguments=args))+'</tool_call>'))
            return out
        def submit(self,request):
            from concurrent.futures import Future
            result=Future();result.set_result(self.generate([request])[0]);return result
    journal=[]
    reports=collect_route_groups(controller.initial,Sampler(),workers=2,journal=journal.append,
                                on_completed_action=controller.completed)
    packed=controller.pack(reports)
    assert len(packed)==1 and packed[0]['status']=='stock_solved'
    assert packed[0]['refit']['suffix_reports']
    suffix=packed[0]['refit']['suffix_reports'][0]
    assert suffix['status']=='rejected_action'
    assert suffix['records'][0]['messages']==packed[0]['records'][0]['messages']
    # This same collected payload must be usable by the real reward/credit path,
    # including a FINISH which mutates the route obligation lists.
    import sqlite3
    from mechet.external_route_data import edge_signature
    from mechet.route_process_reward import ProcessRouteReward
    from mechet.route_refit_credit import training_records
    from scripts.train_external_route_rl import score_groups
    from test_route_native_observation import send
    db=tmp_path/'references.sqlite'
    with sqlite3.connect(db) as conn:
        conn.executescript('CREATE TABLE tasks(id TEXT,target TEXT,split TEXT,reference_steps INTEGER); CREATE TABLE edges(task_id TEXT,signature TEXT);')
        conn.execute('INSERT INTO tasks VALUES(?,?,?,?)',('t','CO','training',1))
        edge=packed[0]['reactions'][0]
        conn.execute('INSERT INTO edges VALUES(?,?)',('t',edge_signature(edge['product'],edge['precursors'])))
    failed=episode('other');send(failed,'finish_trace',{})
    with ProcessRouteReward(db,{'[CH3+]','[OH-]'},failure_policy=base.cfg['route_failure_policy']) as oracle:
        groups=score_groups([dict(task_id='t',target='CO')],[packed[0],failed.finish_report()],2,oracle)
    unique,stats=training_records(groups[0])
    assert stats['conditional_groups']==1 and unique
    from mechet.route_refit_runtime import unique_collection_counts
    counts=unique_collection_counts(controller.all)
    assert counts['model_decision_requests']==len(journal)
    assert counts['executor_calls']==len([r for r in journal if not r.get('executor_skipped')])
    # Journal-only recovery recreates forks, IDs, branch metadata and outcomes.
    replay=RefitCollection([RefitRouteEpisode(base.cfg,'CO',{'[CH3+]','[OH-]'},'root',17)],controller.cfg)
    replay.replay(journal)
    assert replay.pack([e.finish_report() for e in replay.all])==packed
