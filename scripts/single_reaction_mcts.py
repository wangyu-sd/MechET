"""Single-reaction PUCT: formal terminal evidence, NEVER gold endpoint reward.

Run ten complete trajectory attempts, rank distinct executable precursor sets
afterward using the existing mean-NLL evaluator. Formal validity is not a claim
of chemical correctness. No purchasable stock or recursive reactions are used.
"""
import hashlib,math
from pathlib import Path
from scripts.shared_route_mcts import MCTSRouteNetwork,SharedMCTSStore
from mechet.dynamic_electron_tree import chemical_key
from mechet.endpoints import split_precursor_endpoints
from mechet.shared_electron_search import durable_json


class SingleReactionMCTSStore(SharedMCTSStore):
    def __init__(self,root,contract):
        super().__init__(root,{**contract,'evaluation_unit':'single_reaction_endpoint',
            'reward':'productive_strict_finish_v1',
            'single_planner_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def needs_expansion(self,sid):
        v=self.snapshot(sid)
        return not v['edges'] or (1+math.isqrt(v['selections'])>len(v['edges'])
            and v['selections']>v.get('last_expansion_selection',-1))


class SingleReactionMCTS(MCTSRouteNetwork):
    def __init__(self,case_id,product,*,mcts_store,shared_store=None,checkpoint_path=None,program_goal=10):
        if program_goal<1:raise ValueError('positive terminal-program goal required')
        self.program_goal=program_goal;self.completed_programs=set()
        super().__init__(case_id,product,set(),mcts_store=mcts_store,
                         shared_store=shared_store,checkpoint_path=checkpoint_path)

    @property
    def root_key(self):return self.trees[self.target].root_key

    def terminals(self):return self.trees[self.target].terminals()

    def _register_terminal(self,molecule,context):
        pass  # Single-reaction protocol MUST NOT introduce recursive obligations.

    def _transition(self,state,edge):
        key=edge.get('child_context')
        if not key:return None,edge.get('error','invalid action')
        ctx=self.contexts[key]
        if not ctx.node.terminal:return self._state([],self.target,key,()),''
        parts=split_precursor_endpoints(ctx.node.state,self.contexts[self.root_key].node.state)
        precursor=chemical_key(parts.structural) if parts.structural else ''
        if not precursor or precursor==self.target or not any(a['name']=='apply_electron_flow' for a in ctx.node.actions):
            return None,'NONPRODUCTIVE_REACTION'
        sid=self.mcts_store.intern({'pending':[],'molecule':None,'active':key,'ancestors':[],
            'solved':True,'terminal_context':key,'structural_precursor':precursor})
        self.seen_states.add(sid)
        return sid,''

    def _complete(self,rollout,reward):
        if reward:
            state=self.mcts_store.snapshot(rollout['sid'])['state']
            key=state['terminal_context']
            if not self.contexts[key].node.terminal:raise ValueError('nonterminal success')
        self.mcts_store.backup(rollout['lease'].uid,rollout['path'],reward)
        rollout['lease'].close();self.completed_rollouts+=1
        if reward:
            self.positive_backups+=1;self.completed_programs.add(key)
            self.blocked.add(rollout['sid'])  # do not deliberately reselect an already collected terminal
        self.has_pending=self.completed_rollouts+self.censored_rollouts<self.program_goal
        if not self.has_pending:self.stop_reason='MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED'

    def next_context(self):
        if self.completed_rollouts+self.censored_rollouts>=self.program_goal:
            self.has_pending=False;self.stop_reason='MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED'
            self.checkpoint();return None
        self.rollout_concurrency=min(8,max(0,self.program_goal-self.completed_rollouts-self.censored_rollouts))
        result=super().next_context()
        if result is None and self.completed_rollouts+self.censored_rollouts>=self.program_goal:
            self.has_pending=False;self.stop_reason='MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED';self.checkpoint()
        return result

    def checkpoint(self):
        if self.checkpoint_path is None:return
        rollouts=list(self.rollouts)+([self.selected['rollout']] if self.selected else [])
        durable_json(self.checkpoint_path,{'case_id':self.case_id,'root':self.mcts_root,
            'planning_contract':self.mcts_store.contract,'metrics':self.mcts_metrics(),
            'seen_states':sorted(self.seen_states),'blocked':sorted(self.blocked),
            'completed_programs':sorted(self.completed_programs),'program_goal':self.program_goal,
            'pending_rollouts':[{'uid':r['lease'].uid,'sid':r['sid'],'path':r['path']} for r in rollouts],
            'stop_reason':self.stop_reason,
            'search_complete':self.completed_rollouts+self.censored_rollouts>=self.program_goal})

    def report(self):
        return {'id':self.case_id,'target':self.target,'evaluation_unit':'single_reaction_endpoint',
            'collected_programs':len(self.completed_programs),'program_goal':self.program_goal,
            'attempt_goal_counts_invalid_and_censored_trajectories':True,
            'reward_contract':'productive_strict_finish_v1','gold_used_by_search':False,
            'work':self.work,'stop_reason':self.stop_reason,'mcts':self.mcts_metrics()}
