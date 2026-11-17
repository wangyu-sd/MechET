from concurrent.futures import ThreadPoolExecutor
import json
import threading


def test_target_workers_keep_full_g4_ownership_and_result_normalization(tmp_path):
    from mechet.route_distributed_collection import collect_owned_targets
    from mechet.route_work_broker import WorkBroker
    rows=[{'task_id':str(i),'target':'C'} for i in range(9)]
    b=WorkBroker(rows,actor_version='v',contract_sha='c',journal_path=tmp_path/'claims')
    def collect(row,directory):
        directory.mkdir(parents=True,exist_ok=True)
        return [[dict(task_id=row['task_id'],reward=float(k)) for k in range(4)]]
    with ThreadPoolExecutor(3) as pool:
        outputs=list(pool.map(lambda rank:collect_owned_targets(b,rank,tmp_path,collect,workers=2),range(3)))
    groups=[g for owned in outputs for g in owned]
    assert sorted(g[0]['task_id'] for g in groups)==list(map(str,range(9)))
    assert all(len(g)==4 for g in groups) and len(b.results())==9
    # Same declared source rows; load balance never changes the group denominator.
    assert sum(len(x) for x in outputs)==9


def test_recovered_target_uses_verified_result_without_running_actor(tmp_path):
    from mechet.route_distributed_collection import collect_owned_targets
    from mechet.route_work_broker import WorkBroker
    rows=[{'task_id':'one','target':'C'}]
    def make():return WorkBroker(rows,actor_version='v',contract_sha='c',journal_path=tmp_path/'claims')
    groups=collect_owned_targets(make(),0,tmp_path,lambda row,path:[[{'task_id':'one'}]*4],workers=1)
    actual=collect_owned_targets(make(),0,tmp_path,lambda *a:(_ for _ in ()).throw(AssertionError('replayed model')),workers=1)
    assert actual==groups


def test_physical_memory_admission_counts_all_target_workers_not_local_pool():
    from mechet.route_distributed_collection import owner_collection_config
    from scripts.train_external_route_rl import make_episodes
    cfg=dict(world_size=4,executor_workers_per_rank=16,max_action_tokens=2048,candidates=4,seed=17)
    local=owner_collection_config(cfg,object(),object())
    episodes=make_episodes([{'task_id':'t','target':'CO'}],set(),set(),local,'version')
    assert local['executor_workers_per_rank']==1
    assert all(ep.chemistry.memory_divisor==64 for ep in episodes)
