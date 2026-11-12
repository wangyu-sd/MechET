# Checkpoint-native versus global route history: paired GPU diagnosis

The original corrected History-89442 checkpoint can expand the diagnostic
states under its native input. Wrapping the same states in the new global
history representation substantially degrades the observed continuations.
Historical-handle namespacing alone does not repair that degradation.

## Preserved attempt02 result: FAIL

One H20, identical original checkpoint, four fixed saved-state contexts, two
seeds and at most eight new decisions per continuation. These are training
trajectory diagnostics with fixed predicted prefixes, not benchmark results or
fresh complete routes. No optimizer, reference action labels or FINISH mask.

| Input arm | Target-changing continuations | Productive current-reaction commits | Missing-atom errors | Rejected FINISH requests |
|---|---:|---:|---:|---:|
| Checkpoint-native | 8/8 | 7/8 | 0 | 0 |
| Existing v3 history | 3/8 | 1/8 | 18 | 21 |
| Scoped/structured v4 history | 6/8 | 1/8 | 6 | 24 |

The error columns count requests, including repeats; they are not fractions of
routes. A productive commit passes the existing structural-change/cycle guard;
it does not establish agreement with a reference route or experimental validity.
The native arm has one cap-censored continuation, v3 has four, v4 has five.

All372 actor and reference FP32 CUDA adapter tensors exactly matched the
original checkpoint before and after. All186 actor LoRA modules were active;
an enabled-versus-disabled forward changed logits by a maximum17.40625.
The base/adapter lineage is therefore supported by actual GPU checks, rather
than checkpoint filenames alone. Original-base plus History adapter is the
historical composition: History training continued the same Stage-I adapter,
without merging it and training a new delta.

Software status also FAILED the unchanged sampled-versus-teacher-forced
absolute logprob tolerance0.25: observed maximum0.3572392464. Per-arm main B1
maxima were0.1396018863(native),0.3572392464(v3),0.3410599232(v4), with0/3/2
main requests over tolerance. Therefore the discrepancy is already present in
long-history B1 continuation, rather than arising only in padded generation.
All three extra probes really executed one physical B4 with four nonempty
outputs. Total179 completed policy requests,0 optimizer updates,1323.28s.
Do not treat the software failure as a pass by widening the tolerance.

Canonical private report SHA:
`d63e6ecb38a4a0c6feac1e34ee349653b3045ae74767394a4ba57395fc704dde`.
A structure-free aggregate copy is in
[route_history_smoke02_results_20261112.json](../../configs/ops/route_history_smoke02_results_20261112.json).
Attempt01's pre-model bootstrap failure and attempt02's complete failures are
preserved. The model's native behavior is evidence against attributing the
observed non-expansion solely to an incapable or wrongly named checkpoint;
these four contexts do not establish its general accuracy.

## Repairs and exact reproduction artifacts

V4 gives historical atoms and bonds observation-local names and displays parsed
past tool calls as structured actions. It keeps raw archives unchanged, handles
malformed names/keys and protects real boron/ring SMILES from alias substitution.
The smoke establishes that this repair alone is insufficient.

V5 additionally keeps the current native user message byte for byte. Complete
scoped route history and current failure feedback move to the system appendix.
No fake assistant output is added. Mixed historical v3/v5 receipts restore with
the same count, state, raw-output, feedback, commit and native-observation
validation; empty original system messages are preserved too. Preparation still
defaults to v3; v5 is a separately selected diagnostic condition.

- [Frozen attempt02 protocol](ROUTE_SCOPED_HISTORY_SMOKE_20261112.md)
- [Separate v5 follow-up protocol](ROUTE_NATIVE_HISTORY_SMOKE_20261112_03.md)
- [Runtime16-to-v4 patch](../../patches/route_scoped_history_v4_20261112.patch)
- [Attempt18-to-v5 patch](../../patches/route_native_history_v5_20261112.patch)
- [V4 before/after bindings](../../configs/ops/route_scoped_history_v4_integration_20261112.json)
- [V5 before/after bindings](../../configs/ops/route_native_history_v5_integration_20261112.json)

Both patches were dry-run, applied to independent Ceph copies and checked against
all four after-file hashes. Final affected regression suite:67 passed,6.72s;
this overlaps earlier v3/v4 coverage. The suite caught the mixed-version restore
gap before submission. Independent read-only review found no Critical/Important
issue after repair. These are software checks, not experimental success counts.

Attempt03 is declared separately: freshly generate native and v5 under the same
four contexts/seeds,16 continuations and eight extra physical-B4 probe requests.
Keep the same probability gate and both behavioral/software criteria. Its verified
FAIL result is recorded below; preparation was not evidence of a fix.
The formal full v3 task is neither replaced nor promoted by this diagnostic.


## Attempt03 result: FAIL; preserving native user alone is insufficient

The fresh native arm reproduced8/8 target-changing continuations and7/8
productive current-reaction commits, with zero missing-atom or rejected FINISH
requests. V5 produced6/8 target-changing continuations but only1/8 commits,
three missing-atom errors, zero rejected FINISH, three resource-censored and
four diagnostic-cap continuations. Its root continuation generated multiple
tool calls and reached the output boundary. The system appendix follows the
native one-tool instruction; this motivates testing their order, rather than
adding a FINISH mask.

All weight/active-adapter and actual B4 coverage checks passed. Software status
FAILED: maximum sampled/teacher-forced drift0.4759037495>.25. Native main44
receipts had maximum0.1396018863 and zero violations; v5 main42 had six
violations, with its maximum on a2048-token root output. Total94 sampled
requests,0 optimizer updates,1308.01s. Source report SHA
`2d863896cb61a433075875f52c91dc59aca493c0eb5e1ca682a428dcf432b184`;
[aggregate report](../../configs/ops/route_history_smoke03_results_20261112.json).
No actor or reference weight update occurred.

## V6: complete memory before original decision instructions

V6 prefixes complete scoped memory and active feedback before a boundary and
the original native system text. The current native user remains unchanged.
This keeps the original one-tool instructions closest to the current decision;
its effectiveness remains a GPU-test hypothesis. Restore still validates every
original receipt across mixed v3/v5/v6 histories, including empty original
system messages. No kernel, numerical precision, reward or sampling change.

The [separate04 protocol](ROUTE_NATIVE_LAST_HISTORY_SMOKE_20261112_04.md) retains
fresh native/v6, the same four contexts/two seeds,16 continuations and eight
extra B4 requests. Before launch its behavioral gate is strengthened to require
at least as many productive current-reaction commits as native, in addition to
the existing target-change and alias criteria. Temporary movement alone is
insufficient. Earlier v4/v5 criteria and outcomes are preserved unchanged.
Final affected regression suite:74 passed,5.26s, overlapping earlier coverage;
independent review found no Critical/Important issue. Resource censoring is now
an explicit arm-summary count. The
[v6 patch](../../patches/route_native_last_history_v6_20261112.patch) against
immutable19 was applied and all four after-file hashes verified against
[its bindings](../../configs/ops/route_native_last_history_v6_integration_20261112.json).
No v6 behavioral success or formal training promotion is established here.

## Attempt04 result: FAIL; V7 removes remaining historical handle/policy cues

V6 produced8/8 target-changing continuations and5/8 productive commits, versus
native8/8 and7/8. It had zero immediate/rejected FINISH, two missing-atom errors,
three diagnostic caps and no resource censoring. The two invalid atoms A19/A15
were absent from current inventory while their numeric identifiers still
occurred in past labels. This is an observed ambiguity, not proof of copying.
Its maximum main probability drift was0.3600578308>.25, on a392-token coupled
flow; native remained0.1396018863. All GPU adapter/source and actual B4 coverage
checks passed.102 policy requests,0 optimizer updates,878.15s. Keep FAIL.
[Preserved aggregate report](../../configs/ops/route_history_smoke04_results_20261112.json),
SHA `4d111859bdefa8668bd28d3070783f2620cda3b5b581c72cf41d074673d28db0`.

V7 keeps complete chemical history but removes old policy scaffolding from its
model view. Historical handles use independent alphabetic symbols, including
previously scoped and leading-zero spellings. Every chemical inventory, state,
action, event and error remains; raw archives remain verbatim. A historical
SYSTEM is omitted only when identical to current native SYSTEM. A derived
accepted-action capsule is omitted only on an exact recognized native-format
match; all its facts remain reconstructible from complete events. Novel fields,
result codes, extra tails, nonidentical system text and unknown feedback text
remain visible. Standard redundant past feedback instructions are omitted;
outcomes and errors are kept. The active block is factual, without extra global
route/stock imperatives; environment closure/cycle checks are unchanged. Native
SYSTEM and USER remain last. No reward, mask, precision or kernel change.

Final affected suite:85 passed,5.13s, overlapping earlier tests. The REAL native
formatter test caught a two-versus-one-newline suffix mismatch before freezing;
the fix now recognizes the actual PROMPT_SUFFIX. An unknown-capsule-tail case
also failed before repair. Independent final review found no remaining
Critical/Important issue. The
[v7 patch](../../patches/route_passive_history_v7_20261112.patch) against frozen20
was applied and checked against all four
[file bindings](../../configs/ops/route_passive_history_v7_integration_20261112.json).

The [05 protocol](ROUTE_PASSIVE_HISTORY_SMOKE_20261112_05.md) retains fresh
native/v7, the same4contexts/seeds17/18,16 continuations plus8 extra B4 requests,
unchanged0.25 probability tolerance and the stronger commit comparison. No v7
GPU success, benchmark improvement or formal training promotion is claimed here.
