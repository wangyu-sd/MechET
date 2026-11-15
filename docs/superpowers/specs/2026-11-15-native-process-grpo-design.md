# SFT-aligned multistep process-GRPO

The user authorized researching, designing, implementing and submitting the next full multistep RL condition. This supersedes the earlier prohibition on changing the reward for v8: the new observation/reward are an explicitly separate condition. Keep v8 receipts/checkpoints immutable. Use corrected Qwen3.5-0.8B History89442 (2c98…f453) as fresh actor and fixed reference, external TRAIN23730 / VALID2986, G4, seed17, one epoch, four H20 within the eight-H20 total ceiling. No smoke/pilot or hidden mechanism-action supervision.

## SFT interface

SYSTEM, native current-state USER, compact accepted-action capsule, static three-tool schema and Qwen3.5 serialization come from the same functions as the corrected SFT builder. No full-route ledger, changed system instructions, source-bond appendix or FINISH schema pruning enters the new prompt. The environment retains the complete occurrence-preserving route, ledger and raw receipts and still enforces productive FINISH, cycles and stock AND closure.

Selected retry extension (recommended option after the optional async preference received no reply): keep the original SYSTEM and final native USER byte-exact; between them insert only the last same-state/same-reaction rejected assistant call and its short actual tool error receipt. Accepted-prefix inputs are exactly native SFT; rejected contexts are a declared extension, not falsely advertised as existing SFT examples. If strict full-input identity is selected, omit this extension. All raw failure evidence remains archived. Replay/restore must rerender and verify every historical request.

## Research and alternatives

- DeepSeekMath §4.1.3 (https://arxiv.org/html/2402.03300v3) supports process-supervision GRPO with downstream step-reward sums. We use verified tool transitions as steps rather than a learned mathematics reward model.
- Dr.GRPO (https://arxiv.org/abs/2503.20783) motivates retaining fixed token-sum normalization and no reward-standard-deviation scaling.
- GiGPO (https://arxiv.org/abs/2505.10978) provides conditional repeated-state credit. It is a useful later alternative, but exact chemically complete anchor identity and sibling coverage must be established before enabling it; this implementation is not labelled GiGPO.
- Rewarding Progress (https://arxiv.org/abs/2410.08146) distinguishes progress from merely high state value. We therefore give no positive reward just for parse success, accepted IMPORT or arbitrary legal FLOW.
- Potential shaping (https://aima.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) has boundary conditions. The explicit process costs below change the objective; do not claim policy invariance.

Choose deterministic verifiable process rewards plus return-to-go group baselines. Outcome-only reward retains the known sparse signal; a new learned critic/PRM requires unprovided labels and extra compute. This is a process-supervision GRPO variant with Dr-style normalization, not a claim of bit-identical original Dr.GRPO.

## Reward definition

After rollout only, use the existing hidden split-specific reference graph and trusted-ancestor traversal. At each committed reference-supported UNIQUE reaction edge define S_t; L is the reference step count. Define q_t=0.2*min(S_t/max(L,1),1), retaining the existing partial-progress ceiling. Imported participants, legal elementary actions and duplicate edges earn no progress alone. An edge under an unsupported ancestor earns none.

For rejected non-resource actions, accumulate e_t: ordinary invalid/schema/early-FINISH cost1; cycle rejection cost2; add1 if the same semantic rejection already occurred in the same mapped state and occurrence. Define C_t=1-exp(-sum(e_u)/4), eta=0.05/max(L,1). Accepted actions do not dilute this monotonically increasing cost. Failed-then-recovered paths cannot exceed the same clean path by manufacturing a recovery bonus.

Set b=1 for independently reference-supported stock closure, b=0 for invalid route graph, otherwise b=q_T. Each observed decision receives

`r_t=(q_t-q_{t-1}) - eta*(C_t-C_{t-1}) + 1[t=T]*(b-q_T)`.

Therefore sum(r_t)=b-eta*C_T exactly within numerical tolerance. Terminal invalid-graph correction cancels previously accrued progress; complete success remains in[0.95,1], trusted partial progress at least0.15/max(L,1) and at most0.2, no-progress outcomes in[-0.05/max(L,1),0]. Novel unsupported routes are unverified, not automatically chemical negatives; only independently rejected actions incur validator cost. Resource/protected censor stays reward=None and excludes its whole G4 group without partial process optimization.

Persist original request IDs, event rewards, reverse cumulative returns, progress/cost components and a versioned provenance digest. Oracle progress is never model-visible. Reference scoring happens only after the same native terminal/censor rules.

## Per-action credit and loss

Let G_it=sum_{u>=t} r_iu, gamma=1. For baseline only, trajectories terminated before position t contribute zero future return. Define A_it=G_it-(1/4)*sum_j G_jt. This tool-position alignment is an independent-sibling baseline, not an assertion of equal chemical state. No step-count/output-length or standard-deviation normalization is introduced. All generated tokens of action t use A_it; prompt/tool-result tokens remain excluded. Retain actual sampled behavior denominators, token clipping0.2, reference KL0.02 and fixed C1024/global usable-group/G4 normalization. Censored groups remain wholly absent. Retain explicit episode-level and process-level nonzero-group metrics, without calling every nonzero process coefficient a chemistry success.

## Acceptance and submission

1. Unit regressions for native prefix/schema identity, failure feedback scope, fresh reaction boundaries, complete ledger and forged restore rejection.
2. Real corrected SFT validation-file schema/prefix audit over all21355 decisions, with manifests bound; source-defined gold replay where needed to substantiate parity. No new model evaluation.
3. Reward conservation, terminal tiers, duplicate/unsupported/cyclic graph cases, no import reward, bounded errors and whole-group censor tests. Legacy scorer default output must remain unchanged.
4. Independent numeric/gradient checks for variable-length action return-to-go advantages, token masking and normalization. No old scalar advantage accidentally broadcast into the new path.
5. Read-only accounting over all saved v8 complete rounds; this reports rescorer/credit behavior only, not fresh on-policy improvement.
6. Freeze source/runtime/actor/data/reward contract and full protocol; independently review and submit one new four-H20 task. Preserve the old v8 latest complete checkpoint before replacing its allocation if quota requires it. Do not modify other projects.
7. Confirm actual actor-ready, original parent SHA and fresh collection. No performance/learning gain until actual same-protocol full VALID evidence.

The current single-action physical output capacity remains explicitly recorded; this change removes route-history prompt inflation but does not claim every possible capacity censor is solved. Backend replacement/vLLM and optional memory frameworks remain independent follow-on work rather than additional changes bundled into reward validation.

Design review: including the sibling itself in the G4 mean scales the unclipped pure-PG expectation by3/4, as in the old mean-centered baseline; this is not leave-one-out or a strict policy-invariance claim. Retain the old partial ceiling rather than increasing it to discourage a trusted-prefix-then-abandon shortcut. Positive partial returns still make abandonment a real residual risk; full closure and matched VALID, not nonzero process coefficients, decide success.
