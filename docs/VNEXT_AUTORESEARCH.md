# MechET vNext AutoResearch

> **Status:** runnable P0 campaign with frozen scientific gates.  
> **Scope:** automates experiment execution, infrastructure retries, metric collection, and preregistered promotion/stopping decisions. It does **not** autonomously rewrite hypotheses, thresholds, datasets, seeds, rewards, model architecture, or the test set.

## Purpose

The vNext roadmap in `MECHET_VNEXT_EFFICIENT_LLM_ROADMAP.md` introduces several possible improvements—pointer-grounded actions, structured decoding, faster Qwen runtime, packing, chemical-state tree credit, successor value learning, and search-to-policy distillation.

The AutoResearch controller turns the **first evidence gate** into a restartable campaign:

```text
pointer product-start:
16-reaction smoke
    -> 256-reaction screen
    -> full 1,319-reaction trace-view validation

in parallel:
vLLM eager/CUDA-graph/prefix/structured benchmark
packing forward+backward benchmark
```

P1 tree credit, successor value, and search distillation are intentionally **not auto-launched by this first campaign**. They consume substantially more compute and should be unlocked only after the P0 evidence is interpreted.

## Scientific boundary

The controller is a finite-state experiment runner, not a self-modifying research agent.

It may automatically:

- materialize the frozen campaign commit into a Taiji-visible code mirror;
- submit declared Taiji jobs;
- poll task state;
- retry a declared number of infrastructure failures;
- verify required artifacts;
- calculate metrics from fixed artifact paths;
- apply frozen numerical gates;
- stop dependent experiments after a scientific failure;
- persist an append-only-style event history in the campaign ledger;
- resume after controller interruption.

It may **not** automatically:

- change the campaign YAML after the ledger is created;
- move to a different git revision;
- change seeds, split membership, denominators, or test manifests;
- change a metric or threshold because a result is unfavorable;
- modify the executor, prompt contract, model architecture, reward, or optimizer;
- reinterpret an off-reference output as correct;
- unlock the held-out test set;
- launch a P1/P2 method simply because P0 is negative.

The controller fails closed if either the campaign file hash or git commit changes after the run begins.

## Files

- `configs/autoresearch/vnext_p0_20261001.yaml` — frozen P0 campaign DAG and scientific gates.
- `scripts/run_vnext_autoresearch.py` — finite-state controller and ledger.
- `scripts/run_taiji_vnext_pointer_search.sh` — matched product-start pointer runner.
- `scripts/summarize_vnext_pointer_search.py` — paired Top-1/Pass@beam, bootstrap CI, and exact McNemar summary.
- `scripts/run_taiji_vnext_runtime_benchmark.sh` — vLLM eager/CUDA-graph/prefix/structured runtime benchmark.
- `scripts/summarize_vnext_runtime_benchmark.py` — distributed runtime aggregation.
- `scripts/run_taiji_vnext_packing_benchmark.sh` — real Qwen3-8B packing training microbenchmark.
- `tests/test_vnext_autoresearch.py` — freeze/gate/dependency regression tests.

## P0 campaign

### 2026-10-01 pointer smoke recovery

The first `pointer_smoke16` POD stopped at its data gate before sampling:
the job reads the frozen action-delta reaction file, but its gate had been
changed to expect history-v2 decision-manifest fields. The original ledger and
failed task remain intact. `configs/autoresearch/vnext_p0_pointer_recovery_20261001.yaml`
pins a corrected commit, new pointer task flags and output directories, and
the **unchanged** 16 → 256 → 1,319 scientific thresholds. It imports the
original runtime/packing reports as artifact stages instead of rerunning those
GPU jobs. The packing report failed its throughput gate; that result remains
visible as `SCIENTIFIC_STOP` in the recovery ledger.

Use a distinct `MECHET_AUTORESEARCH_CODE_MIRROR` ending in
`MechET-autoresearch-vnext-pointer-recovery-20261001` for this recovery. Do not
resume the original pointer stage against its frozen, incorrect code commit.

The original runtime task was later found to place all eight vLLM worker
processes on GPU 0 under `torchrun`; the other seven GPUs were idle and no
benchmark report was produced. That instance was stopped and its empty output
directory was retained with a `.gpu0-collision-attempt1` suffix. The isolated
`configs/autoresearch/vnext_runtime_rankfix_20261001.yaml` reruns only this arm:
eight independent workers each see one GPU through `CUDA_VISIBLE_DEVICES`, and
the summarizer now requires all eight ranks, all 128 states and all five modes.
It retains the original runtime metric definitions and gates. The pointer
recovery ledger imports the corrected runtime report as an artifact stage.

### 2026-10-01 runtime compatibility recoveries

The rank-fixed attempt proved the eight GPU bindings, but its first CUDA-graph
warmup created roughly 124 host CPU threads per worker while vLLM constructed
dummy LoRA weights. It was stopped after producing only the eager reports; its
partial output is preserved under the
`vnext_runtime_a100_20261001.cpufix-predecessor-rankfix1` suffix. PR #75 capped
OpenMP, MKL, and OpenBLAS at two threads per independent worker. The CPU-fixed
attempt completed all three unconstrained modes, then the pinned vLLM 0.8.5
rejected request-level `xgrammar:no-fallback` backend selection before the
structured modes could generate. Its partial output is preserved under the
`vnext_runtime_a100_20261001.backendfix-predecessor-cpufix1` suffix.

PR #76 sets the same strict XGrammar backend at **engine initialization** for
the two structured modes, as required by vLLM V1. The recovery campaign is
`configs/autoresearch/vnext_runtime_backendfix_20261001.yaml`; use a code mirror
ending in `MechET-autoresearch-vnext-runtime-backendfix-20261001` and a separate
ledger. It retains the same frozen 128 oracle states, Stage-II adapter, five
modes, metric definitions, and numerical gates. Neither previous incomplete
attempt is a benchmark result, and neither should be resumed against its old
code commit. The pinned XGrammar package successfully compiled all 128 frozen
inventory schemas offline (maximum 74 atoms); this checks schema compilation,
not vLLM generation or endpoint quality.

On 2026-10-02 both backend-fixed retries reached the structured mode but ended
inside vLLM 0.8.5's *additional* V1 schema validator: it rejects `minItems`
and `maxItems` even when XGrammar itself compiles the schema. These are user
program compatibility failures, not evidence of a platform allocation failure
or a completed five-mode benchmark. The recovery removes only those unsupported
array keywords from the generation schema and checks the same cardinality
requirements after JSON decoding. All 128 frozen schema-only and 128 inventory
schemas pass both pinned XGrammar compilation and the pinned vLLM validator.
This is a separate code-compatibility recovery; the states, adapter, five modes,
metric definitions, and numerical gates remain unchanged.

### Pointer arm

The pointer arm uses the same Stage-II policy, current deterministic executor, product-only model-visible information boundary, and matched K=4 proposal budget.

The three stages are:

1. `pointer_smoke16` — cheap product-start sanity check.
2. `pointer_valid256` — medium validation screen.
3. `pointer_valid1319` — full trace-view validation denominator.

The 16/256 stages are **screening gates**, not publication claims. The full 1,319-stage report persists:

- baseline Top-1;
- pointer Top-1;
- paired Top-1 delta;
- bootstrap 95% CI;
- baseline and pointer Pass@beam;
- paired Pass@beam delta and CI;
- exact McNemar discordance;
- pointer invalid-handle count.

A positive full-validation direction is required for the P0 pointer stage to finish as `PASSED`; otherwise it becomes `SCIENTIFIC_STOP`. This outcome does not change historical MechET evidence.

### Runtime arm

The runtime job compares fixed oracle states under:

- vLLM eager + prefix cache;
- CUDA-graph without prefix cache;
- CUDA-graph + prefix cache;
- CUDA-graph + schema decoding;
- CUDA-graph + current-inventory structured decoding.

The gate requires:

- CUDA-graph + prefix cache not slower than the eager reference;
- at least 99% exact action agreement for the deterministic unconstrained comparison;
- at least 99% parse success and structural-handle validity for inventory-constrained decoding.

A failed runtime gate means the optimization is not adopted; it does not block the chemistry arm.

### Packing arm

The packing job runs real Qwen3-8B Stage-II forward+backward passes. It first checks packed-vs-independent assistant-only loss parity, then measures supervised-token throughput and peak allocated memory.

Packing is promoted only if:

- absolute loss difference is at most 0.03; and
- block packing improves supervised-token throughput.

The frozen paper trainer remains unchanged until this benchmark passes.

## Infrastructure retry policy

Each Taiji P0 stage allows one automatic infrastructure retry. A retry changes only the task flag suffix; it does not change the scientific config.

A platform task that ends successfully but lacks its declared result artifact is treated as an infrastructure failure, not a negative scientific result.

A metric gate failure is never retried as infrastructure.

## Starting the campaign

The controller needs three environment values on the submit host:

```bash
export MECHET_TAIJI_CLIENT=/path/to/taiji_client
export MECHET_TAIJI_DONOR_TASK=<successful-task-with-private-ceph-init>
export MECHET_AUTORESEARCH_CODE_MIRROR=/aaa/fionafyang/buddy1/whaleywang/MechET-autoresearch-vnext-20261001
```

Then run from the repository checkout:

```bash
python scripts/run_vnext_autoresearch.py \
  --campaign configs/autoresearch/vnext_p0_20261001.yaml \
  --ledger outputs/autoresearch/vnext_p0_20261001/ledger.json \
  --workdir outputs/autoresearch/vnext_p0_20261001 \
  --mode run
```

For a no-submit contract check:

```bash
python scripts/run_vnext_autoresearch.py \
  --campaign configs/autoresearch/vnext_p0_20261001.yaml \
  --ledger /tmp/mechet-vnext-autoresearch-ledger.json \
  --workdir /tmp/mechet-vnext-autoresearch \
  --mode step \
  --dry-run
```

`--mode status` reads the frozen ledger without advancing the campaign.

## Ledger states

Each stage is in exactly one of:

- `PENDING`
- `RUNNING`
- `PASSED`
- `SCIENTIFIC_STOP`
- `INFRA_FAILED`
- `BLOCKED`
- `SKIPPED`

A dependent stage is automatically marked `BLOCKED` when an upstream scientific/infrastructure terminal state prevents execution.

## Transition to P1

P0 does not automatically mutate itself into a Tree-RL campaign.

After P0 completes, the next campaign should be frozen from the observed evidence and should compare, under a matched state/candidate budget:

1. token-ratio GRPO;
2. GSPO;
3. same-state tree/sibling credit;
4. tree credit plus successor reachability;
5. search-to-policy distillation;
6. greedy distilled-policy deployment.

That P1 campaign must have a new YAML hash and a new ledger. This separation is intentional: it prevents the controller from using P0 results to silently change the scientific question or optimization protocol.
