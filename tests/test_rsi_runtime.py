import threading
import queue


def test_slow_cpu_tool_does_not_barrier_unrelated_ready_episode():
    from scripts.rsi_runtime import rolling_agent_batch
    blocked=threading.Event();unblock=threading.Event();observed=[]
    class Engine:
        def __init__(self):self.q=queue.Queue();self.i=0
        def submit(self,request):
            self.i+=1;observed.append(request);self.q.put((str(self.i),request));return str(self.i)
        def poll(self,timeout=0.):
            try:return self.q.get(timeout=timeout)
            except queue.Empty:return None
    def request(c):return {'id':c['id'],'step':c['step']}
    def consume(c,response):
        if c['id']=='slow':blocked.set();assert unblock.wait(3)
        if c['id']=='fast' and c['step']==1:
            assert blocked.wait(1);unblock.set()
        c['step']+=1;c['done']=c['step']==2;return c
    result=rolling_agent_batch([{'id':'slow','step':0},{'id':'fast','step':0}],request,consume,Engine(),cpu_workers=2)
    assert len(result)==2 and {'id':'fast','step':1} in observed


def test_partial_episode_restores_verified_import_without_resampling():
    from scripts.rsi_runtime import restore_recorded_steps
    from scripts.run_natural_language_value_search import Node
    c={'node':Node(target='CO',state='[CH3:1][OH:2]',next_map=3,visited={'CO'}),
       'state':'[CH3:1][OH:2]','steps':[],'history':[],'flow_count':0,'terminal':False}
    step={'accepted':True,'name':'import_fragments','arguments':{'fragments':[{'smiles':'[Cl-]','count':1,'purpose':'electron_participant'}]},
          'state_before':'[CH3:1][OH:2]','state_after':'[CH3:1][OH:2].[Cl-:3]','raw':'original_sample','completion_ids':[1,2]}
    restore_recorded_steps(c,[step],'reverse')
    assert c['node'].next_map==4 and len(c['steps'])==1
    assert c['state']=='[CH3:1][OH:2].[Cl-:3]'
