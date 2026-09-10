# Grounded electron-flow: stage results and next research gate

Last updated: 2026-09-11.

This note consolidates the current evidence around electron-flow localization,
the conclusions that can already be drawn, and the ideas discussed for the next
iteration. It is an engineering and research decision record, not a frozen
paper-results table. Final paper claims remain governed by
[`PAPER_EXPERIMENT_PROTOCOL.md`](PAPER_EXPERIMENT_PROTOCOL.md).

## Scope and denominators

The unqualified **FlowER full** split is the frozen official reaction-level
split: **257,171 train / 2,890 valid / 28,971 test**. The current strictly
executable condition contains **257,167 / 2,890 / 28,967** rows. Neither the
historical 32k proof subset nor its 28k/27k executable views may be called
FlowER full.

The experiments in this note are development diagnostics on fixed validation
subsets. They do not change the official split, filter a benchmark denominator,
or establish full-test endpoint accuracy.

## Stage evidence

### First-use state SFT validation

The corrected first-use checkpoint is
`outputs/agent/first_use_state_qwen3_8b_v100_seed17_20260909/checkpoint-4019`
after one epoch.
It was evaluated greedily (`K=1`) on the fixed seed-17 product-only validation
set. The run was stopped by user request after 225 of 256 examples, so every
number below is explicitly partial.

| Metric | Partial result | Interpretation |
|---|---:|---|
| Syntactically complete generations | 225 / 225 | The model learned the program surface |
| `finish()` followed by EOS | 225 / 225 | The former non-termination pathology was removed |
| Maximum imports in one sample | 11 | No sample exceeded the diagnostic threshold of 20 |
| Strict formal execution | 19 / 225 (8.44%) | Site grounding, not surface syntax, is now the main failure |
| Structural endpoint exact match | 0 / 225 | No positive endpoint conclusion is supported |

The dominant recorded failure labels were 76 `MISSING_SITE`, 47 non-equivalent
SMARTS ambiguities, 69 `STATE_ASSERTION_MISMATCH`, and 14 `INVALID_SMILES`.
Only five failures were fragment-import failures. Counts can overlap across
diagnostic stages and therefore must not be summed as a partition.

**Conclusion.** First-use fragment scheduling fixed repeated import and missing
termination behavior, but it did not solve atom/site localization. More changes
to import syntax are therefore unlikely to address the dominant error.

### Gold-derived grounded-event smoke

PR #51 added a controlled diagnostic in which each example contains one gold
electron-flow event and up to seven formally executable, gold-derived hard
negatives. A candidate is made by replacing a source or sink container with
another container of the same semantic signature, executing it, deduplicating
by successor state, and deterministically shuffling the labels.

The smoke used the unadapted Qwen3-8B base model, gold state, and gold-derived
candidate construction. It is a representation diagnostic—not deployable
product-only inference and not endpoint evaluation.

| Candidate slice | n | Top-1 | Random Top-1 | Recall@2 | Recall@4 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|
| All eligible events | 982 | 24.03% | 23.47% | 50.20% | 82.69% | 2.93 |
| At least four candidates | 742 | 20.22% | 17.98% | 40.43% | 77.09% | 3.28 |
| Exactly eight candidates | 210 | 15.71% | 12.50% | 32.86% | 56.19% | 4.16 |

Candidate construction attempted 1,019 events and retained 982. The 37
ineligible examples comprise 34 without enough executable negatives and three
`BE_DELTA` cases. Candidate counts were: 102 with two choices, 138 with three,
181 with four, 151 with five, 133 with six, 67 with seven, and 210 with eight.

The apparent advantage is weak and confounded by a severe label prior: Qwen
selected label `A` for 773/982 examples (78.7%). Label-permutation ensembling is
therefore required before interpreting this representation diagnostic. The
smoke does **not** show that base Qwen has learned reliable electron chemistry.

The run is reproducible through `scripts/build_qwen_grounded_event_smoke.py`,
`scripts/eval_qwen_grounded_event_smoke.py`, and
`scripts/run_taiji_qwen_grounded_event_smoke.sh`. The completed default artifact
is `outputs/eval/qwen3_8b_grounded_event_smoke_20260910/`; the frozen summary is
also recorded in the [PR #51 result comment](https://github.com/wangyu-sd/MechET/pull/51#issuecomment-5621021619).

### Action-space audit

On the same 1,019 selected validation events, the molecular and primitive-space
statistics are:

| Quantity | Median | P90 | Maximum | Mean |
|---|---:|---:|---:|---:|
| Atoms | 54 | 100 | 215 | 61.53 |
| Bonds | 53 | 104 | 211 | 60.82 |
| Lone-pair donor handles | 15 | 25 | 50 | 16.33 |
| `FORM` primitive upper bound | 825 | 2,304 | 8,132 | 1,157.71 |
| `CLEAVE` primitive upper bound | 106 | 208 | 422 | 121.64 |
| `SHIFT` primitive upper bound | 5,500 | 20,384 | 89,886 | 9,494.41 |
| Total primitive upper bound | 6,496 | 22,671 | 98,440 | 10,773.76 |

The diagnostic upper bounds use `FORM = L(n-1)`, `CLEAVE = 2E`, and
`SHIFT = 2E(n-2)`. They count raw addressable primitives before executor
filtering and are not the number of complete coupled-event candidates.

The actual reference programs are locally short: 936/1,019 events (91.85%)
contain at most two moves. Across 2,078 moves, the dominant types are
`BOND->ATOM` (890), `LP->BOND` (882), and `BOND->BOND` (277). The most common
whole-event skeleton is `CLEAVE+FORM` (739/1,019, 72.52%); the top seven
skeletons cover 91.66% of events.

**Conclusion.** The chemistry is usually a short composition, but choosing
atom-level arguments directly creates a large structured space. Enumerating it
as flat text is the wrong interface even when the final program is short.

## Ideas considered

| Idea | Current judgment | Status |
|---|---|---|
| A--H grounded-event ranking | Useful controlled probe of representation and priors; gold-derived options prohibit headline inference claims | Smoke implemented in PR #51 |
| Local marked SMILES with ephemeral `:1/:2` tags | More readable than internal `q,aro,deg,r2` descriptors; useful as a representation diagnostic | Discussed, not implemented |
| Gold-independent atom/bond/lone-pair handles | Formally clean, but flat composition remains large and turns Qwen into an inefficient pointer network | Not recommended as the main design |
| GNN atom/bond scoring | Natural for local vectors, but already overlaps the repository's `ForwardElectronExpert`, A3 and graph/edit baselines | Existing component/baseline direction |
| Lazy tree search over executable moves | Can postpone enumeration, but a weak policy/value function amplifies branching cost | Defer until the event policy is informative |
| Whole-event counterfactual ranking | Removes token-syntax loss and can test whether the model distinguishes executable wrong mechanisms | Retain as a diagnostic/subobjective |
| Lifted electron-flow programs | Separates global mechanistic intent from exact graph grounding; potentially compositional rather than template classification | Preferred hypothesis; audit before training |

## Preferred hypothesis: lifted electron-flow programs

The next representation should ask the language model for an address-free
mechanistic program, while a deterministic graph constraint solver binds its
variables to the current molecular state. For example:

```text
exists D, C, L:
  lone_pair(D)
  electrophilic(C)
  leaving_group(L)
  bonded(C, L)
  LP(D) -> BOND(D, C)
  BOND(C, L) -> ATOM(L)
```

The model decides the global chemical relation and electron-flow composition;
the solver performs exact atom grounding, merges symmetry-equivalent bindings,
and passes only grounded actions to the existing executor. This directly targets
the observed failure mode: Qwen need not memorize arbitrary map numbers, but the
final answer is still caused by a real, replayable electron-transfer program.

This is only a promising hypothesis. It degenerates into ordinary template
classification if the lifted vocabulary becomes a closed list of reaction
templates, and it fails if most programs admit many chemically non-equivalent
groundings. Both risks can be measured without training a model.

## Required model-free gate

Before another full SFT or RL run, compile the strict executable FlowER condition
without changing its **257,167 / 2,890 / 28,967** denominators and report:

1. abstraction coverage: fraction representable by the lifted grammar;
2. gold grounding recall under execution-equivalent matching;
3. number of non-equivalent groundings per program;
4. symmetry-only multiplicity separately from true ambiguity;
5. lifted operator and predicate vocabulary coverage;
6. source/target token lengths versus the current action program;
7. solver runtime and failure taxonomy by split;
8. all missing or unresolved rows retained as denominator failures.

Proceed to a Qwen SFT/ranking pilot only if coverage is high and non-equivalent
grounding ambiguity is manageable. Otherwise reject or revise the abstraction
before consuming accelerator time.

## Research boundary and literature context

Search methods such as [Tree of Thoughts](https://arxiv.org/abs/2305.10601),
[RAP](https://arxiv.org/abs/2305.14992),
[LATS](https://arxiv.org/abs/2310.04406),
[TS-LLM](https://arxiv.org/abs/2309.17179), and
[ReST-MCTS*](https://arxiv.org/abs/2406.03816) show how language-model reasoning
can be organized as a search tree. Retrosynthetic planning already has explicit
tree-search precedents such as
[Retro*](https://proceedings.mlr.press/v119/chen20k.html). These works motivate a
future search layer, but do not solve the present grounding problem. Empirical
analysis of language-model search also finds that advanced search depends on a
strong discriminator and may cost substantially more than sampling
([When is Tree Search Useful for LLM Planning?](https://arxiv.org/abs/2402.10890)).

Accordingly, tree search is not the immediate rescue. The next scientific
question is narrower and testable: **can a lifted electron-flow program preserve
the causal mechanism while removing arbitrary atom-address prediction?**

## Decisions recorded

- Do not start another SMARTS/address-format branch before the model-free gate.
- Do not use the gold-derived A--H candidate smoke as deployable inference or an
  endpoint metric.
- Do not claim that unadapted Qwen already shows reliable chemical preference.
- Do not present atom-level Qwen vectors as a new "chemical potential" model;
  graph models are the appropriate baseline for that formulation.
- Keep first-use and grounded-event runs as explicit diagnostics/ablations.
- Preserve the paper's central claim: the precursor must be produced by an
  executable electron-flow computation, not followed by a post-hoc rationale.
