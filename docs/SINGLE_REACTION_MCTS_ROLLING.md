# Full3120 single-reaction MCTS and rolling inference

This replaces ONLY the stopped31k adaptive tree. The live R1/R2 stock-closure
task remains at its immutable runtime. Source IDs, frozen min-proxy product
fields, Qwen3-0.6B Stage-II weights and endpoint evaluator are unchanged. This
is a diagnostic min-proxy protocol, not the corrected principal-product claim.

Each full test product starts its own reverse-electron tree. Shared sampled
PUCT uses actual path N/W backups: a productive strictly executable finish
returns1; an illegal sampled action returns0; context/RAM interruptions are
censored. This reward is NOT chemical correctness or gold endpoint reward.
There is no external stock, learned value, recursive next reaction or new RL.

The default stopping goal is10 completed/censored trajectory attempts per
target, INCLUDING failed attempts, not10 unique successes. Admission respects
the remaining attempt budget. A physically unrepresentable root is explicitly
resource-limited and retains its denominator. No fixed chemical depth, import
count or total branch cap. This is a bounded K10 search condition, not the old
unlimited-coverage condition; do not call their compute budgets matched.
Distinct executable precursor sets are ranked by the existing assistant mean
unwarped NLL after search; GT is read only in endpoint scoring. Ten trials do
not guarantee ten distinct endpoints. Every target/failed trial is retained.

One GPU-owner thread uses the pinned official vLLM0.8.5 V0 public add_request
and step APIs. Up to128 independent state requests per GPU are admitted;
vLLM schedules at most256 active model sequences, each expansion requests16
completions. These are concurrency settings, not chemical branching caps.
Sixteen CPU workers per GPU plan, execute and persist independently serialized
cases. Completed requests return immediately, allowing refill while a longer
request continues. There is no batch-wide GPU/CPU barrier or unsafe concurrent
generate calls. Pinned API source was checked in the existing runtime overlay.

Full3120 frozen greedy is validated read-only and never regenerated. The old
global policy/executor cache namespace is reused; single-reaction MCTS values
are separately keyed by reward, actor/executor, exact state/history and goal.
Raw prompts, token IDs, scores, parser errors, accepted/rejected native edges,
MCTS graphs/checkpoints and per-rank progress are saved incrementally. Progress
includes ready/planning/GPU/executor counts and cache reuse every5seconds.
Default Taiji stdout has a45second heartbeat. No model smoke is authorized:
unit tests verify software behavior; measured speed/accuracy await the full job.

No in-place restart of a partially written output is claimed by this runner.
Use a new output directory and shared caches; old evidence is never overwritten.

Verification before deployment:850 tests passed,11 skipped, one existing PEFT
warning. Targeted RED-to-GREEN regressions cover attempt-goal stopping including
invalid trajectories, no uncounted speculative rollouts, final nonroot context
censoring, true versus short completion truncation, partial-search seals, raw
NLL/token retention, no gold-to-controller projection and continuous short-request
refill during a long request. Independent review's Important findings were
reproduced and repaired. No model accuracy or speedup is inferred from unit tests.
