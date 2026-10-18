# Independent complete-route control: prepared, not executed

The controller and full-population preparation code are in
`scripts/independent_route_control.py`. `independent_plan.json` is the exact
CPU-prepared manifest; SHA256:
`49f0889eb3b630ee30184fdfe33050954ec5d0b6c3406c1cb2fcc461b8ed73d9`.

It binds the original460 targets/four cohorts, the fixed200 observed-report
pairing, all original eight runtime headers, whole-salt stock files and actual
frozen actor weights. Original checkpoint metadata contains a stale hash;
the plan records that discrepancy without changing the checkpoint or replacing
the verified physical weight hash. Full target/stock files remain outside Git.

Each episode has a private executor/network and no shared policy samples,
frontier, N/W search statistics or backtracking. It takes one stochastic
decision at a time with the original Stage-II inventory/compressed-history
prompt and recursively closes **all** structural stock obligations. Invalid
actions terminate the attempt. Imports alone cannot create a successful route;
already-stock targets are not newly synthesized routes. Truncated generations
remain charged; actual context/memory exhaustion is censored, not bad chemistry.

The predeclared1/4/10 grid counts independent complete-route episodes, not
chemical steps, depth or import caps. Every episode is retained even if an
earlier episode succeeds. This does not establish a matched-compute advantage.

Thirteen unit tests passed, including real executor import/finish behavior,
recursive stock-AND semantics, seed/state independence, truncation accounting,
source-header/weight corruption and full-population preservation. These are
software tests, not model results.

**Still missing:** the batched actor transport/CLI, full-run result aggregation
and conventional complete-reaction planner control. No inference/training/Taiji
task was launched. A prepared controller/manifest is not a finished baseline
experiment and must not be substituted for one in the paper.
