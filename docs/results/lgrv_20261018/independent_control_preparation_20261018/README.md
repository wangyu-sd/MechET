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
one existing PEFT warning,142.96s. The runtime/controller are unchanged since
that gate; the subsequently added CLI has its separate final verification below.

## Formal CLI and cross-shard collection (19 October)

`scripts/independent_route_cli.py` now provides `validate`, `run` and `collect`.
The formal CLI pins the approved plan/actor hashes; it does not expose a
subset, changed seed, episode-budget or expected-hash override. Validation
reruns the original CPU preparation into a temporary directory, compares the
whole regenerated plan and both product-only/paired-ID files, and rechecks
all source bytes, physical weights, stocks and eight original actor headers.
It does not merely accept hashes asserted by a new self-sealed input.

The real prepared full460/fixed200 plan passed this command with no model
load; [the saved actual stdout](CLI_VALIDATION_20261019.json) retains the
validation-only schema and eight shard sizes58/58/58/58/57/57/57/57. CLI source
SHA256 at that execution:
`12eae729d81c65739009201e8b9f64b43376df59d8b918a271e56cd28f0dee32`.

```bash
PYTHONPATH=.:src python -m scripts.independent_route_cli validate \
  --plan /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/paper/nmi_literature_evidence/prepared_independent_controls_20261018_v1/independent_plan.json \
  --route-assets /aaa/fionafyang/buddy1/whaleywang/MechET/data/route_benchmarks/frozen_r1_r2_20261013/manifest.json
```

`run` additionally requires `--output`, one `--rank` in0..7 and a separately
authorized `--authorize-model-execution` flag. Each worker owns its disjoint
original target slice and all10 independent episodes per target. Default
stdout logs continue during backend initialization and policy execution.
The flag is not permission to launch a task under this PR's no-new-GPU boundary;
no `run` command or platform submission has been executed here.

`collect` takes the same source arguments plus `--output <run-root>` and an
optional fresh `--receipt <path>`. It rechecks eight source/runtime contracts,
episode ownership and unmodified full populations; verifies original inputs,
raw/parsed outputs, tree records and every contiguous native journal commit;
and recomputes stock-AND closure from the saved reaction graph. Counts are
first1/4/10 episode-index unions, never ranked candidates. Missing ranks,
unreported episodes, censoring and already-stock targets remain explicit.
One worker's100% does not establish full460 completion. Raw returned-token
work from unreported episodes is retained; progress-only extra work is a
separate lower bound, not silently added twice. Collection is not a liveness
claim about an active writer, and existing receipts cannot be overwritten.

Thirteen CLI tests add full460 software fixtures, source-rederivation and
corruption cases, all-eight partition coverage, partial file collection,
missing raw/native transitions, foreign ownership and authorization checks.
The worker's zero-step test uses synthetic already-stock fixtures and no
policy requests, not model inference. All39 focused CLI/runtime/controller/
continuous-adapter tests passed. The final full CPU suite passed **1,253 tests/
14 skipped**, one existing PEFT warning,182.18s. No production code changed
after that run; the earlier1,240-test gate above is its historical runtime-only
verification stage.

The independent-control **prepare-manifest/runner-only** requirement is now
implemented; this is not a new empirical baseline. Actual control predictions
remain unrun, as required by the no-new-model boundary. Conventional complete-
reaction planner preparation, broader primary/analogue review and final
whole-goal review remain unfinished. No training or Taiji task was launched.
