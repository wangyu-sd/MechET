"""Continuous native-policy requests with independent CPU successor execution."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


def collect_async_route_groups(episodes,sampler,*,workers,journal,snapshot=None,recovered=None,
                               on_completed_action=None,before_dispatch=None):
    all_episodes=list(episodes)
    ready=deque(e for e in all_episodes if e.next_decision() is not None)
    generating={};executing={}
    limit=sampler.cfg['generation_batch_size']
    sampler.scheduler_metrics=dict(dispatch_requests=0,cpu_completed_calls=0,
                                    max_generation_inflight=0)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while ready or generating or executing:
            if ready and before_dispatch:
                # Derived fork IDs must not become durable model requests
                # before their deterministic execution prefix is durable.
                before_dispatch()
            while ready and len(generating)<limit:
                ep=ready.popleft();request=ep.next_decision()
                value=(recovered or {}).get(request.request_id)
                if value is not None:
                    executing[pool.submit(ep.accept,value)]=ep
                else:
                    generating[sampler.submit(request)]=ep
                    sampler.scheduler_metrics['dispatch_requests']+=1
                    sampler.scheduler_metrics['max_generation_inflight']=max(
                        sampler.scheduler_metrics['max_generation_inflight'],len(generating))
            pending=set(generating)|set(executing)
            if not pending:continue
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                if future in generating:
                    ep=generating.pop(future)
                    # submit() resolves ONLY after the original probability receipt
                    # is durable. Transport failures propagate, never chemical rejects.
                    value=future.result()
                    executing[pool.submit(ep.accept,value)]=ep
                else:
                    ep=executing.pop(future);future.result()
                    journal(ep.records[-1])
                    if on_completed_action:
                        added=on_completed_action(ep)
                        all_episodes.extend(added)
                        ready.extend(e for e in added if e.next_decision() is not None)
                    sampler.scheduler_metrics['cpu_completed_calls']+=1
                    if snapshot:snapshot(ep)
                    if ep.next_decision() is not None:ready.append(ep)
    return [ep.finish_report() for ep in all_episodes]
