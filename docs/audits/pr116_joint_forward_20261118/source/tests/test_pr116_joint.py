import math
import unittest
import torch
from mechet.route_joint_forward import forward_events, dpo_loss, ranked_pairs

class JointTest(unittest.TestCase):
    def test_bounded_progress_and_rollback(self):
        edges=[(0,'edge1',1.),(1,'edge1',1.),(2,'edge2',None),(3,'edge3',.75)]
        self.assertEqual(forward_events(5,edges,1,'stock_solved',.1),[.1,0.,0.,0.,0.])
        events=forward_events(5,edges,2,'rejected_action',.1)
        self.assertAlmostEqual(math.fsum(events),0.)
        self.assertGreater(events[0],0.)
        self.assertEqual(forward_events(5,edges,2,'resource_censored',.1),None)
    def test_observed_zero_beats_missing(self):
        common=dict(node='parent',version='v',protocol='t',prompt=(1,),eos=9)
        def row(action,score,i):
            return dict(common,action=action,score=score,receipt=str(i),tokens=(i,9))
        items=[row('A',.75,2),row('B',0.,3),row('C',None,4),dict(row('D',1.,5),node='other')]
        self.assertEqual(ranked_pairs(items,.25),[('2','3')])
    def test_pair_loss_gradient(self):
        w=torch.tensor(-2.,requires_grad=True); l=torch.tensor(-3.,requires_grad=True)
        loss=dpo_loss(w,l,w.detach(),l.detach(),.1)
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)
        loss.backward()
        self.assertLess(w.grad.item(),0);self.assertGreater(l.grad.item(),0)

class CacheTest(unittest.TestCase):
    def test_zero_is_durable_transient_unavailable_is_retried(self):
        import tempfile
        from pathlib import Path
        from mechet.route_joint_cache import get_scores
        with tempfile.TemporaryDirectory() as folder:
            protocol={'teacher_sha':'a','budget':4}
            q={'teacher_reactants_mapped':'one','target_product':'two','target_heavy_teacher_maps':[1,2]}
            calls=[]
            def run(rows):
                calls.append(len(rows));return [None if len(calls)==1 else 0. for r in rows]
            for expected in [None,0.,0.]:
                rows=get_scores([q,q],protocol,Path(folder),run)
                self.assertEqual([r['score'] for r in rows],[expected,expected])
            self.assertEqual(calls,[1,1])

class JointOptimizerTest(unittest.TestCase):
    def test_shared_learner_step_and_reward_only_control(self):
        from unittest.mock import patch
        from mechet.external_route_rl import optimize_routes, PROCESS_GRPO_ALGORITHM
        from types import SimpleNamespace
        import sys
        groups=[[dict(reward=0.),dict(reward=1.)]]
        def score(module,pairs,pad):
            return [torch.log_softmax(module.weight[0],dim=0)[list(output)] for _,output in pairs]
        native=[dict(request_id=str(i),prompt_token_ids=[0],generated_token_ids=[i],
                     behavior_logprobs=[-math.log(2)]) for i in (0,1)]
        stats=dict(original_g4_eligible=True,effective=1,conditional_groups=1,masked_conditional_groups=0,
            conditional_nonzero_advantage_groups=1,conditional_zero_advantage_groups=0,
            conditional_all_failed_groups=0,conditional_m1_completed_used=2)
        def run(weight):
            actor=torch.nn.Linear(2,1,bias=False)
            torch.nn.init.zeros_(actor.weight)
            ref=torch.nn.Linear(2,1,bias=False)
            ref.load_state_dict(actor.state_dict());ref.requires_grad_(False)
            optim=torch.optim.SGD(actor.parameters(),lr=.1)
            cfg=dict(algorithm=PROCESS_GRPO_ALGORITHM,refit={},loss_batch_size=2,loss_token_normalizer=1.,
                     pad_token_id=0,clip_epsilon=.2,kl_beta=.01,max_grad_norm=10.,
                     behavior_logprob_tolerance=.05)
            if weight is not None:cfg['joint_forward']=dict(margin=.25,beta_d=.1,lambda_d=weight)
            pairs=[(((0,),(1,)),((0,),(0,)))]
            fake=SimpleNamespace(token_logps_batch=score,synchronize_gradients=None)
            with patch.dict(sys.modules,{'scripts.pr96_earho_contract':fake}), \
                    patch('mechet.route_refit_credit.training_records',return_value=(list(zip(native,[-.5,.5])),stats)), \
                    patch('mechet.route_joint_forward.likelihood_pairs',return_value=pairs), \
                    patch.object(optim,'step',wraps=optim.step) as step:
                result=optimize_routes(actor,ref,optim,groups,cfg)
                self.assertEqual(step.call_count,1)
                self.assertTrue(result['optimizer_updated'])
            return actor.weight[0].detach(),result
        legacy,_=run(None);control,_=run(0.)
        self.assertTrue(torch.equal(legacy,control))
        joint,metrics=run(.1)
        self.assertGreater(joint[1],control[1])
        self.assertEqual(metrics['global_dpo_pairs'],1)
        self.assertAlmostEqual(metrics['dpo_loss'],math.log(2),places=6)

class PhaseTest(unittest.TestCase):
    def test_two_reports_share_one_unavailable_teacher_input(self):
        from unittest.mock import patch
        from mechet.route_joint_phase import attach_joint_evidence
        q=dict(status='eligible',teacher_reactants_mapped='[OH2:1]',target_product='O',
               target_heavy_teacher_maps=[1],finish_index=0,finish_request_id='r')
        reaction=dict(product='O',precursors=['O'],occurrence='root')
        report=dict(reactions=[reaction],reward=.1,reward_evidence={},refit=dict(suffix_reports=[]))
        root=[report,dict(report)]
        def no_score(rows,*other):
            self.assertEqual(len(rows),1)
            return [dict(score=None,key='observed-unavailable')]
        with patch('mechet.route_joint_phase.extract_forward_input',return_value=q), \
                patch('mechet.route_joint_phase.get_scores',side_effect=no_score):
            result=attach_joint_evidence([root],dict(joint_forward=dict(lambda_f=.1,lambda_d=.1,
                teacher_protocol={'file':55904912},cache_dir='unused')))
        self.assertEqual(result['global_distinct'],1)
        self.assertEqual(result['eligible_finish_appearances'],2)
        self.assertEqual(result['scored_finish_appearances'],0)
