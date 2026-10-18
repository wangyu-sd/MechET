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

## Rolling batch runtime, prepared and software-tested

`scripts/independent_route_runtime.py` now supplies the rolling scheduler,
lazy vLLM0.8.5 V0/TP1 transport, exact Stage-II tool-prefix construction,
durable per-decision input/output recording and episode-prefix aggregation.
This adds code, **not new actor predictions or a completed baseline**.

Each in-flight episode owns a separate controller and incremental journal.
CPU tool/persistence workers overlap with the existing single-owner continuous
GPU-decoding adapter. Inputs are committed before generation; raw token IDs,
text and finish reasons are committed before probability validation and tool
execution. Even a missing probability preserves the original output, charges
observed generation tokens and censors the affected attempt. No erroneous
backend receipt is silently labelled a chemical failure or discarded.

Every episode directory contains `requests/`, `raw_outputs/`, `generations/`,
`journal/`, `tree.json` and `report.json`. Files use the existing checksummed,
atomic/fsynced record format. Requests include complete messages, rendered
prompt, input token IDs, actual context room and sampling seed; generations
retain raw text, output token IDs, log probability, parsed action and errors.
The continuously emitted default-stdout progress report separates finished
episodes, censored episodes, pending work and generated tokens. Interrupted
unreturned requests have explicitly unknown output work, not zero-cost work.

No chemical depth/import cap was added. A truncated action uses the existing
384→768→… adaptation until actual model-context exhaustion. Independent draws
are n=1/temperature1/top-p1 with no probability warping; samples are never
reused across episodes. The aggregation function takes the original episode
index prefixes, **not an NLL-sorted subset**; already-stock targets do not
count as newly synthesized routes. Paired IDs are an explicit supplied set,
not a subset selected by success.

Model loading refuses absent explicit authorization and incorrect physical
actor weights before importing vLLM. The low-level scheduler can accept CPU
software fixtures for verification; a formal entrypoint must authenticate the
prepared full460/fixed200 manifest before any model run. The prepared manifest
and original controller remain unchanged.

Eleven new CPU runtime tests cover real executor import/finish, asynchronous
out-of-order scripted completions, private journals, raw I/O preservation,
unavailable-NLL errors, context exhaustion, interruption, prefix denominators
and authorization/weight checks. A foreign engine response is preserved but
cannot execute against another episode's pending state. Together with the
existing13 controller tests and two continuous-adapter tests,26 focused tests
pass. Scripted completions
are software fixtures, not model results or a smoke experiment.
The final whole-repository CPU suite passed **1,240 tests/14 skipped**, with
one existing PEFT warning,142.96s. No production code changed after that run.

**Still missing:** the formal manifest-authenticated CLI and cross-shard result
collection, actual independently generated control predictions, conventional
complete-reaction planner control and final whole-goal review. No inference,
training or Taiji task was launched. Do not substitute a prepared runtime for
a completed experiment in the paper.
