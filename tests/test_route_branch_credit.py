"""CPU-only contract tests for optional conditional electronic-route forks."""
from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from mechet.route_branch_credit import (
    ForkAnchor, PolicyAction, ConditionalGroup, SuffixRollout,
    OBSERVED, PENDING, RESOURCE_CENSORED, PROTECTED_CENSORED,
    digest, choose_anchor, conditional_advantages, conditional_drgrpo_loss,
    collect_conditional_group, summarize_groups,
)


class ToyEpisode:
    """The reference is private; only finished attempts can be evaluated."""
    def __init__(self):
        self.snapshot = {}
        self.status = PENDING
        self.calls = 0
        self.progress = None

    def restore(self, snapshot):
        self.snapshot = copy.deepcopy(snapshot)
        self.status = PENDING
        self.calls = 0
        self.progress = None

    def public_observation(self):
        return {"target": self.snapshot["target"], "chemical_state": self.snapshot["state"],
                "pending": self.snapshot["pending"], "accepted_history": self.snapshot["history"]}

    def step(self, action):
        self.calls += 1
        if action.action_name == "FLOW":
            self.progress = action.arguments["which"]
            self.status = OBSERVED
        elif action.action_name == "CENSOR":
            self.status = PROTECTED_CENSORED
        elif action.action_name == "INVALID":
            # rejected tool call still consumes policy tokens
            self.status = PENDING
        else:
            raise AssertionError(action.action_name)

    def outcome_status(self):
        return self.status


def fixture_anchor():
    snapshot = {"target":"T", "state":"S1", "pending":["Y1", "Y2"],
                "history":[{"action":"FLOW", "accepted":True}],
                "private_reference":"only in environment restore data"}
    # This public projection is constructed by the adapter, not by the policy.
    env = ToyEpisode()
    env.restore(snapshot)
    return ForkAnchor(target_id="T", source_rollout_id="root-trace-42",
                      decision_index=3, behavior_version="sft-sha:abc",
                      snapshot=snapshot,
                      public_observation_hash=digest(env.public_observation()))


def action(which="good", *, behavior="sft-sha:abc", name="FLOW", ids=(13,14)):
    return PolicyAction(action_name=name, arguments={"which":which},
                        prompt_token_ids=(1,2,3), generated_token_ids=tuple(ids),
                        behavior_logprobs=(-0.7,)*len(ids),behavior_version=behavior)


def observed_group(rewards=(1.0,0.0), nstep=1):
    anchor=fixture_anchor()
    siblings=tuple(SuffixRollout(i, OBSERVED, r, tuple(action(ids=(13,14)) for _ in range(nstep)), 100+i)
                   for i,r in enumerate(rewards))
    return ConditionalGroup(anchor,siblings)


def test_fork_from_same_state_and_independent_replay():
    anchor=fixture_anchor()
    observed=[]
    def policy(obs,seed):
        assert "private_reference" not in obs
        observed.append((obs,seed))
        return action("good" if len(observed)%2 else "bad")
    score_calls=[]
    def score(env):
        assert env.status==OBSERVED
        score_calls.append(env.progress)
        return 1.0 if env.progress=="good" else 0.0
    group=collect_conditional_group(anchor,env_factory=ToyEpisode,sample_action=policy,
                                    score_after_termination=score,group_size=2)
    assert [s.reward for s in group.siblings]==[1.0,0.0]
    assert [s.status for s in group.siblings]==[OBSERVED,OBSERVED]
    assert group.n_calls==2 and group.n_tokens==4
    assert len({s.seed for s in group.siblings})==2
    assert [x[0] for x in observed]==[observed[0][0],observed[0][0]]
    assert score_calls==["good","bad"]
    assert group.anchor.snapshot["history"]==anchor.snapshot["history"]
    assert conditional_advantages(group)==(0.5,-0.5)


def test_anchor_selects_accepted_flow_without_hindsight():
    a=fixture_anchor()
    b=ForkAnchor(target_id=a.target_id,source_rollout_id="root-trace-77",decision_index=5,
                 behavior_version=a.behavior_version,snapshot=a.snapshot,
                 public_observation_hash=a.public_observation_hash)
    assert choose_anchor([a,b],17)==choose_anchor([b,a],17)
    with pytest.raises(ValueError,match="FINISH"):
        ForkAnchor(target_id=a.target_id,source_rollout_id=a.source_rollout_id,
                   decision_index=2,behavior_version=a.behavior_version,snapshot=a.snapshot,
                   public_observation_hash=a.public_observation_hash,next_action_kind="FINISH")
    with pytest.raises(ValueError,match="versions"):
        choose_anchor([a,ForkAnchor(target_id="T",source_rollout_id="a",decision_index=2,
                                   behavior_version="other",snapshot=a.snapshot,
                                   public_observation_hash=a.public_observation_hash)],17)


def test_observation_replay_mismatch_is_fatal():
    class CorruptEnvironment(ToyEpisode):
        def public_observation(self):
            return {"state":"CORRUPTED"}
    with pytest.raises(ValueError, match="restored public observation"):
        collect_conditional_group(fixture_anchor(),env_factory=CorruptEnvironment,
                                  sample_action=lambda obs,seed:action(),
                                  score_after_termination=lambda env:1.0)


def test_behavior_policy_identity_is_frozen():
    with pytest.raises(ValueError,match="frozen behavior"):
        collect_conditional_group(fixture_anchor(),env_factory=ToyEpisode,
                                  sample_action=lambda obs,seed:action(behavior="different"),
                                  score_after_termination=lambda env:1.0)
    with pytest.raises(ValueError,match="behavior policy changed"):
        ConditionalGroup(fixture_anchor(),(
            SuffixRollout(0,OBSERVED,1.0,(action(),),1),
            SuffixRollout(1,OBSERVED,0.0,(action(behavior="wrong"),),2)))


def test_all_identical_rewards_have_zero_conditional_advantage():
    g=observed_group((0.0,0.0))
    assert conditional_advantages(g)==(0.0,0.0)
    assert summarize_groups([g])["variable_reward_fraction"]==0.0


def test_unfinished_is_censored_not_falsely_zero():
    scored=[]
    g=collect_conditional_group(fixture_anchor(),env_factory=ToyEpisode,
                                sample_action=lambda obs,seed: action(name="INVALID"),
                                score_after_termination=lambda env: scored.append(1),
                                group_size=2,max_decisions=2)
    assert scored==[]
    assert all(s.status==RESOURCE_CENSORED and s.reward is None for s in g.siblings)
    assert g.n_calls==4 and g.n_tokens==8
    assert conditional_advantages(g) is None
    assert summarize_groups([g])["censored"]==1


def test_partial_group_censor_excludes_whole_group():
    anchor=fixture_anchor()
    g=ConditionalGroup(anchor,(
        SuffixRollout(0,OBSERVED,1.0,(action(),),1),
        SuffixRollout(1,PROTECTED_CENSORED,None,(action(),),2)))
    assert conditional_advantages(g) is None
    with pytest.raises(ValueError,match="all conditional groups censored"):
        conditional_drgrpo_loss([g],{},beta=0.0)
    with pytest.raises(ValueError,match="censored route"):
        SuffixRollout(1,PROTECTED_CENSORED,0.0,(action(),),2)
        ConditionalGroup(anchor,(
            SuffixRollout(0,OBSERVED,1.0,(action(),),1),
            SuffixRollout(1,PROTECTED_CENSORED,0.0,(action(),),2)))


def _learner(group, *, values=(-0.7,-0.7), ref_values=None):
    cur={}
    ref={}
    for j,s in enumerate(group.siblings):
        for k,a in enumerate(s.actions):
            cur[(0,j,k)]=torch.tensor(values,dtype=torch.double,requires_grad=True)
            ref[(0,j,k)]=torch.tensor(ref_values if ref_values is not None else values,dtype=torch.double)
    return cur,ref


def test_clipped_conditional_drgrpo_gradient_sign():
    g=observed_group()
    cur,_=_learner(g)
    loss,audit=conditional_drgrpo_loss([g],cur,beta=0.0,fixed_scale=2.0)
    loss.backward()
    assert abs(float(loss.item()))<1e-10  # centered on-policy group
    assert all(float(x)<0 for x in cur[(0,0,0)].grad)  # increase good suffix likelihood
    assert all(float(x)>0 for x in cur[(0,1,0)].grad)  # decrease bad suffix likelihood
    assert audit["nonzero_advantage_siblings"]==2
    assert audit["objective"]=="conditional_dr_grpo_suffix_only_v1"


def test_fixed_scale_not_length_normalized():
    g=observed_group(nstep=2)
    cur={}
    for j,s in enumerate(g.siblings):
        for k,a in enumerate(s.actions):
            cur[(0,j,k)]=torch.tensor((-0.7,-0.7),requires_grad=True)
    loss,audit=conditional_drgrpo_loss([g],cur,beta=0.0,fixed_scale=1024)
    assert abs(float(loss.item()))<1e-8
    assert audit["policy_tokens"]==8
    loss.backward()
    assert float(cur[(0,0,0)].grad[0])<0
    assert float(cur[(0,0,1)].grad[0])==pytest.approx(float(cur[(0,0,0)].grad[0]))


def test_sampled_behaviour_logprobs_not_overwritten_by_learner():
    g=observed_group()
    cur,ref=_learner(g,values=(-0.9,-0.9))
    loss,audit=conditional_drgrpo_loss([g],cur,ref,beta=0.02)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in cur.values())
    assert all(float(lp)==-0.7 for lp in g.siblings[0].actions[0].behavior_logprobs)


def test_token_shape_and_logprob_guards():
    g=observed_group()
    cur,_=_learner(g)
    cur[(0,0,0)]=torch.tensor((-0.7,),requires_grad=True)
    with pytest.raises(ValueError,match="tensor shape"):
        conditional_drgrpo_loss([g],cur,beta=0)
    with pytest.raises(ValueError,match="count"):
        PolicyAction("FLOW",{},(1,),(2,3),(-0.1,),"ckpt")
    with pytest.raises(ValueError,match="positive"):
        PolicyAction("FLOW",{},(1,),(2,),(0.3,),"ckpt")


def test_gradient_clipping_negative_advantage_and_reference_kl():
    g=observed_group()
    cur,ref=_learner(g,values=(-1.5,-1.5),ref_values=(-0.7,-0.7))
    loss,audit=conditional_drgrpo_loss([g],cur,ref,beta=0.02)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in cur.values())


def test_group_mixing_and_accounting_are_explicit():
    g=observed_group()
    short=observed_group((1.0,0.0,0.0))
    with pytest.raises(ValueError,match="mixed sibling counts"):
        conditional_drgrpo_loss([g,short],{},beta=0)
    audit=summarize_groups([g])
    assert audit["calls"]==2 and audit["generated_tokens"]==4
    assert audit["variable_reward_fraction"]==1.0
    assert "not_independent_target_rollouts" in audit["branch_policy"]