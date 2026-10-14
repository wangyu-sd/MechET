# Shared electronic-route MCTS for full R1/R2

Status: design for user review; NOT implemented or submitted.

## Scope and invariant

Replace the R1/R2 novelty/NLL frontier scheduler with genuine sampled-action
MCTS. Keep the frozen FlowER Stage-II Qwen3-0.6B actor, existing event prompt,
executor, full460 targets and original cohort stocks. No learned value model,
world model, new SFT, RL, reference-guided search or filtering. Existing31k
task is outside this change. The stopped `_06` artifacts remain historical
adaptive-search outputs, never relabeled as MCTS.

## Planning state and reaction boundary

A planning state contains the multiset of unresolved structural precursor
obligations, one active reaction's exact native electronic state, import
inventory, compressed accepted-action history, and synthesis ancestry needed
to reject cycles. These are not reducible to a canonical SMILES alone.

An import or electron-flow action changes only the active reaction. An
accepted `finish_trace` yields a completed reaction hyperedge and replaces its
selected obligation with ALL extracted structural precursors. Stock members
are closed leaves; other precursors remain obligations. The next active
obligation is chosen deterministically from the canonical sorted unresolved
multiset. Actual action feedback and product-only addressing remain unchanged.

Model EOS, a generated stop string, one purchasable fragment, or an arbitrary
intermediate state is NOT a solved route. Completion requires an executable,
nonproductive/cycle-free reaction sequence with ALL structural leaves in the
original cohort stock. Formal execution and stock closure do not establish
experimental chemical feasibility.

## Actual MCTS loop

1. **Selection:** traverse the planning graph using edge statistics
   `Q + c * P * sqrt(parent_visits) / (1 + edge_visits)`, default `c=1.5`.
   Statistics include completed visit count N and return sum W; Q=W/N.
   Separate in-flight reservations avoid duplicated concurrent traversal.
2. **Expansion:** sample Qwen actions lazily. Progressive widening admits
   more distinct actions as a node receives visits: `1 + floor(sqrt(visits))`.
   Proposal requests use at most16 completions per microbatch; this is NOT a
   maximum total branch count. Save every proposal and rejection; distinct
   valid transitions remain recoverable even when not yet admitted.
3. **Simulation:** continue real executor transitions, including further
   reactions after `finish_trace`, until full stock closure or a failed
   sampled continuation. GPU work is interleaved across targets/rollouts;
   a scheduler yield does not terminate a chemical path or invent a reward.
4. **Backup:** completed stock-connected rollouts return1. Failed sampled
   continuations return0, without labeling all alternatives unsolvable.
   Update N/W along the EXACT traversed edges once per rollout ID. Pending or
   interrupted rollouts are censored, not counted as failures. No terminal
   stock reward is attached to a partial reaction or one successful AND leaf.

Policy priors are estimated from unique recorded proposal batches and action
frequencies; repeated reads of one cached batch do not create additional
independent observations. Raw model likelihoods remain available for route
ranking, not substituted for MCTS rollout values.

## Shared graph, not just shared cache

Share accepted deterministic transitions and MCTS N/W statistics for EXACTLY
matching planning states, actor/executor identity and stock closure criterion.
The key binds unresolved obligations, active native context and ancestry,
plus stock SHA. Source case IDs and GT labels do not enter the key. Different
stocks or different remaining obligations cannot share success values.

Use the existing checksum-protected, atomic Ceph record store and POSIX locks;
no SQLite WAL database on Ceph. Node topology is shared, while visits and
backups are edge-based in the transposition graph. Detect cycles during
selection and backup only the simulated path, not every graph ancestor.
Persist node/edge statistics, active rollout paths, reservation leases and
completed-backup IDs. Release/recover crashed-worker leases without counting
a second backup. Record actual model calls versus reused cached observations.

## Completion, work and outputs

For the first full-result condition, stop a target after its FIRST complete
nonzero-step stock-connected route; this is not stopping on the first
reaction `finish_trace`. Preserve all alternatives already discovered. This
condition reports first-solution coverage/cost and observed route diversity,
not exhaustive route counts. R2 uses the same160 RetroCast targets and fixed
reference-complexity strata. Compare to the sealed `_05` full460 greedy.

No fixed chemical depth, maximum total breadth or import count. Renewable
work checkpoints and actual context/RAM safeguards remain. Open unsolved
searches are not declared complete merely because sampling repeats; report
pending/resource-limited roots separately, retaining all460 in the denominator.
MCTS does not guarantee finite completion on every target.

Save default-POD progress and atomic per-rank progress.json every wave:
initialized/attempted/solved/finished/pending targets, output tokens, actual
model calls, cache reuse, rollout counts, N/W backups and elapsed time. Save
raw prompts/completions, intermediate graphs and first solved certificates
immediately. Separate percent initialized from percent search-finished.

## Verification and submission gate

CPU unit tests must demonstrate exploration of alternatives, real N/W backup,
AND closure, stock-specific values, history-sensitive transpositions, invalid
actions versus pending rollouts, cycle rejection, exactly-once concurrent
backup, crash-lease recovery and partial-graph materialization. Include an
executor-backed tiny deterministic test WITHOUT new model sampling; these
are software tests, not chemical/model benchmark results.

Run the complete repository tests and independent review before ONE ordinary
8A100 full460 MCTS submission. Use a new `meteor` task/output, successful
Tencenttaiji6/private-Ceph donor and45s default stdout heartbeat. Verify actual
POD, eight engines, MCTS mode/statistics, nonempty saved generations and logs.
Do not restart old adaptive search or touch other live jobs. No unsolicited
smoke/subset experiment is authorized.
