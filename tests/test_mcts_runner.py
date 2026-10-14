"""MCTS runner wiring/progress, without loading models or allocating GPUs."""
from types import SimpleNamespace


def test_runner_builds_actual_mcts_not_adaptive_route_network(tmp_path):
    from scripts.infer_dynamic_electron_trees import make_route_controller
    from scripts.shared_route_mcts import MCTSRouteNetwork,SharedMCTSStore
    s=SharedMCTSStore(tmp_path,{'stock_sha':'s'})
    controller=make_route_controller({'id':'a','product_unmapped':'CO'}, {'C','O'},
                                    algorithm='mcts',shared_store=None,mcts_store=s,
                                    checkpoint_path=tmp_path/'checkpoint.json')
    assert isinstance(controller,MCTSRouteNetwork)
    assert controller.next_context() is not None
    assert controller.mcts_metrics()['algorithm']=='shared_sampled_puct_mcts_v1'
    controller.close()


def test_progress_distinguishes_finished_from_initialized_and_pending_rollouts():
    from scripts.infer_dynamic_electron_trees import search_progress
    active={'c':SimpleNamespace(first_solution_work=None,completed_rollouts=3,
                positive_backups=0,censored_rollouts=1)}
    reports=[{'id':'a','stock_solved':True,'mcts':{'completed_rollouts':2,'positive_backups':1}},
             {'id':'b','stock_solved':False,'mcts':{'completed_rollouts':1,'positive_backups':0}}]
    progress=search_progress(active,reports,denominator=4,attempted=3,tokens=80,calls=10,
                             elapsed=20.,algorithm='mcts')
    assert progress['finished']==2 and progress['finished_percent']==50.
    assert progress['attempted_percent']==75.
    assert progress['stock_solved_targets']==1
    assert progress['pending']==2
    assert progress['completed_mcts_rollouts']==6
    assert progress['positive_mcts_backups']==1
    assert progress['full_result_complete'] is False


def test_resource_censored_targets_are_processed_but_not_search_complete():
    from scripts.infer_dynamic_electron_trees import search_progress
    p=search_progress({},[{'stock_solved':False,'stop_reason':'TASK_HOST_MEMORY_BUDGET'}],
        denominator=1,attempted=1,tokens=4,calls=2,elapsed=1.,algorithm='mcts')
    assert p['resource_limited']==1
    assert p['finished']==0 and p['pending']==1
    assert p['all_targets_processed'] is True
    assert p['full_result_complete'] is False


def test_runner_resume_restores_journal_instead_of_reinitializing(tmp_path):
    from scripts.infer_dynamic_electron_trees import initialize_or_restore_route,make_route_controller
    from scripts.shared_route_mcts import SharedMCTSStore
    from mechet.shared_electron_search import SearchJournal
    s=SharedMCTSStore(tmp_path/'shared',{'stock_sha':'s'})
    row={'id':'a','product_unmapped':'CO'};path=tmp_path/'checkpoint.json'
    c=make_route_controller(row,{'C'},algorithm='mcts',shared_store=None,mcts_store=s,checkpoint_path=path)
    j=SearchJournal(tmp_path/'journals','a',{})
    c=initialize_or_restore_route(j,c,stock={'C'},resume=False)
    c.next_context();c.close()
    fresh=make_route_controller(row,{'C'},algorithm='mcts',shared_store=None,mcts_store=s,checkpoint_path=path)
    recovered=initialize_or_restore_route(j,fresh,stock={'C'},resume=True)
    assert recovered.censored_rollouts==1
    assert j.sequence==1
    assert recovered.next_context() is not None
    recovered.close()


def test_partial_materializer_reads_shared_mcts_graph_without_restarting_rollouts(tmp_path,monkeypatch):
    import hashlib,json,sys
    from scripts.shared_route_mcts import MCTSRouteNetwork,SharedMCTSStore
    from mechet.shared_electron_search import SearchJournal
    from scripts.materialize_electron_search import main
    stock_file=tmp_path/'stock.txt';stock_file.write_text('C\n')
    sha=hashlib.sha256(stock_file.read_bytes()).hexdigest()
    rank=tmp_path/'rank';s=SharedMCTSStore(tmp_path/'shared',{'stock_sha':sha})
    c=MCTSRouteNetwork('a','CO',{'C'},mcts_store=s,
        checkpoint_path=rank/'mcts'/'checkpoint.json')
    j=SearchJournal(rank/'journals','a',{'case_stock_sha':sha,'search_algorithm':'mcts'})
    j.initialize(c);c.next_context();c.close()
    output=tmp_path/'graph.json'
    monkeypatch.setattr(sys,'argv',['materialize','--journal',str(j.root),'--stock',str(stock_file),
        '--output',str(output),'--mcts-checkpoint',str(c.checkpoint_path)])
    assert main()==0
    graph=json.loads(output.read_text())
    assert c.mcts_root in graph['mcts_graph']
    assert graph['mcts']['root_visits']==0
    assert graph['persistence']['search_complete'] is False
