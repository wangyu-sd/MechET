# MechET vNext: efficient LLM training, inference, and chemical-state policy roadmap

> **Status:** forward-looking design / implementation plan.  
> **Date:** 2026-10-01.  
> **Scope:** this document does **not** modify the current frozen MechET paper protocol, benchmark denominators, executor semantics, or the Stage-II parity repair in PR #70. It defines a gated vNext research path after current protocol-correct evaluation is complete.

## 1. Motivation

Current MechET evidence shows that tool syntax and formal interaction are no longer the primary bottlenecks. Existing local diagnostics already reach approximately 99% tool-choice accuracy, 98% argument compilation, and 96% formal execution, while strict electron-event and multi-flow accuracy degrade substantially as coordination complexity increases.

The vNext objective is therefore not to make the model better at generic tool calling. It is to improve three specific capabilities:

1. **Local chemical grounding** — identify the chemically relevant atoms/bonds before composing an electron-flow event.
2. **Long-horizon credit assignment** — compare alternative chemical decisions from the same molecular state without turning an all-negative branch group into a spurious positive update.
3. **Efficient search and deployment** — exploit the unusually high prefix sharing of molecular-state branching so that training-time search does not make inference prohibitively expensive.

The design principle is:

```text
localize -> compose -> transition -> branch -> credit -> distill
```

with the existing deterministic executor remaining authoritative for molecular-state transitions.

---

## 2. Non-goals

This roadmap deliberately does **not** propose:

- replacing Qwen3-8B with a larger generic model as the first intervention;
- learned world-state prediction in place of the deterministic executor;
- a JEPA or latent world model for transitions already exactly computable by the executor;
- multi-agent debate;
- a large LLM chemistry judge as the source of formal validity;
- a new reaction-condition, yield, or selectivity model;
- rebuilding a multistep planner from scratch;
- making latent chemistry actions opaque.

Those directions may be separate projects, but they do not target the current MechET bottleneck.

---

## 3. Target architecture

The preferred vNext architecture is a hierarchical chemical-action policy around the existing Qwen actor and executor.

```text
current state S_t
      |
      v
chemical-site grounding
(pointer head first; graph proposer only if needed)
      |
      | top-K source/sink candidates
      v
Qwen3-8B event composition
      |
      v
structured / grammar-constrained action
      |
      v
deterministic MechET executor
      |
      v
successor state S_{t+1}
      |
   +--+------------------+
   |                     |
   v                     v
successor value       terminal?
   |
   v
bounded state tree
```

The scientific division of labor is:

- **pointer / graph head:** where can the chemistry occur?
- **Qwen:** which local candidates should be combined into the next electron-flow event?
- **executor:** what molecular state actually follows?
- **value model:** which executed successor is promising for endpoint recovery?
- **tree policy optimization:** which sibling chemical decision deserves credit?

---

## 4. P0 architecture change: pointer-grounded chemical actions

### 4.1 Why pointer grounding

Current aliases such as `A01`, `A17`, and bond handles are not open-vocabulary text. They are references into the current molecular inventory. Generating them as ordinary language tokens forces the LLM to solve a pointer-selection problem indirectly.

Factorize the action distribution as

[
P(a mid S)
=
P(	ext{type}mid S)
P(	ext{source}mid S,	ext{type})
P(	ext{sink}mid S,	ext{type},	ext{source})
P(	ext{event extras}mid cdots).
]

The first implementation should reuse Qwen hidden states for visible atom/bond handles rather than adding a second graph model immediately.

For atom handle representation (h_i),

[
p_{mathrm{src}}(imid S)
=
mathrm{softmax}(q_s^	op h_i),
]

and similarly for sink selection.

Training objective:

[
mathcal L =
mathcal L_{mathrm{LM}}
+
lambda_{mathrm{src}}mathcal L_{mathrm{src}}
+
lambda_{mathrm{sink}}mathcal L_{mathrm{sink}}.
]

### 4.2 Promotion gate

Before changing the paper model, measure on the frozen validation decision set:

- Source Recall@1/@4/@8
- Sink Recall@1/@4/@8
- Source-sink pair Recall@K
- Strict event accuracy
- Successor-state agreement
- Breakdown by 1 / 2 / 3 / >=4 coupled flows

Promote only if localization gains propagate to strict-event or successor accuracy, especially on multi-flow cases.

---

## 5. P0 decoding change: dynamic structured actions

The runtime already knows the legal inventory at each state. The decoder should therefore not spend samples on malformed JSON, nonexistent handles, or invalid tool fields.

Use per-state structured decoding:

[
a in mathcal A_{mathrm{structural}}(S_t).
]

Candidate implementations:

- vLLM structured outputs with XGrammar;
- SGLang constrained JSON / grammar decoding.

This is a structural validity constraint, **not** a chemical truth constraint. The executor remains responsible for chemistry.

Required ablation:

1. unconstrained current decoding;
2. schema-only constrained decoding;
3. schema + current-inventory handles.

Report compilation failure, invalid-handle rate, wall time, and chemistry metrics. If chemistry is unchanged but runtime improves, keep it as an engineering optimization rather than a scientific claim.

---

## 6. P0/P1 training change: chemical-state tree credit assignment

Independent full-trajectory GRPO is poorly matched to MechET because the meaningful comparison occurs among alternative actions from the **same** molecular state.

Preferred training unit:

```text
S_t
|- a1 -> S1  productive / exact successor
|- a2 -> S2  productive alternative
|- a3 -> S3  dead end
|- a4 -> invalid
```

The key credit question is:

> Given one current chemical state, which sibling action leads to a better executed successor?

This aligns naturally with Tree-GRPO style rollout trees and with MechET's existing same-state branching, successor deduplication, and bounded continuation.

### 6.1 Required safeguards

- all-negative sibling groups receive zero relative positive advantage;
- chemically equivalent successors are pooled before relative credit;
- invalid/no-op/cycle branches are rejected before ranking;
- gold precursor or reference successor never enters inference prompts;
- training-private reference information, if used, is an explicitly named supervision condition;
- terminal endpoint equality remains a separate outcome signal.

### 6.2 Algorithms to benchmark

Do not invent another optimization acronym before matched comparison.

Run:

- GRPO baseline;
- GSPO baseline;
- Tree-GRPO style sibling credit;
- Tree credit + dynamic sampling.

Dynamic sampling should skip or resample groups with no useful reward/value contrast instead of promoting the least-bad member of an all-negative group.

Reference code:
- Tree-GRPO: https://github.com/AMAP-ML/Tree-GRPO
- slime: https://github.com/THUDM/slime
- veRL: https://github.com/volcengine/verl

---

## 7. P1 successor value / process reward

Formal executability is only a hard feasibility filter; it must not be treated as evidence that a branch is useful.

Learn

[
V_phi(P,S_t)
=
P(	ext{successful endpoint}mid P,S_t)
]

or transition value

[
Q_phi(P,S_t,a_t,S_{t+1}).
]

The preferred target is **future endpoint reachability**, not generic molecular plausibility.

Potential supervision:

- successful vs failed on-policy trajectories;
- executor-valid off-policy hard negatives;
- first-divergence outcomes;
- search survival / terminal endpoint labels.

A PRIME-style implicit process-reward experiment is worth testing because it can learn dense transition values from outcome labels without requiring manual per-step labels. However, it must be compared against the existing simple successor classifier before being promoted.

Do not use a large language-model judge as the primary successor-value oracle.

---

## 8. P1 search-to-policy distillation

Search may improve accuracy but should not become a permanent inference tax.

Use tree search / value-guided branching as a training-time teacher, then distill the selected action distribution back into the policy:

[
q_{mathrm{search}}(amid S)
ightarrow
pi_	heta(amid S).
]

Candidate losses:

[
mathcal L_{mathrm{distill}}
=
D_{mathrm{KL}}
(q_{mathrm{search}}|pi_	heta),
]

or selected-action NLL.

This is related to on-policy distillation / self-evolving distillation, but the teacher here is chemistry-specific:

```text
actor + executor + state tree + value
```

rather than an external language teacher.

Useful references:
- SEED self-evolving on-policy distillation: https://github.com/jinyangwu/SEED
- EasyOPD: https://github.com/k-irona/EasyOPD

Required deployment comparison:

1. base policy, greedy;
2. policy + tree search;
3. distilled policy, greedy;
4. distilled policy + small tree search.

The desired result is that most search gain is internalized into a cheaper policy.

---

## 9. Graph model only after pointer-head gate

If pointer grounding remains the dominant bottleneck, add a dedicated graph candidate proposer.

The first reference implementation should follow the **structure**, not the edit vocabulary, of Graph2Edits:

```text
molecular graph
   -> message passing
   -> atom logits
   -> bond logits
   -> top-K local candidates
```

Reference:
- Graph2Edits: https://github.com/Jamson-Zhong/Graph2Edits

Possible second-stage encoder:
- RXNGraphormer: https://github.com/licheng-xu-echo/RXNGraphormer

Do not adopt a graph encoder merely because it is chemistry-native. Promote only if it improves source/sink recall or successor accuracy beyond the Qwen pointer-head baseline under matched parameter and compute accounting.

A more invasive graph-soft-token / cross-attention adapter is P2 and should only be considered after the local candidate experiment establishes a real graph-information bottleneck.

---

## 10. Efficient SFT stack

The existing Qwen3-8B SFT line should stay on the mature training stack rather than being rewritten.

Recommended stack:

```text
Qwen3-8B
  -> Megatron-SWIFT / ms-swift
  -> sequence packing
  -> FlashAttention
  -> Liger kernels where stable
  -> data parallel replicas first
```

### 10.1 Immediate training benchmarks

Benchmark, do not assume:

- packing on/off;
- Liger on/off;
- gradient checkpointing on/off;
- bf16 full / LoRA mode used by current Stage-SFT;
- per-device token throughput;
- peak memory;
- end-to-end epoch time.

For an 8B model that fits on one A100/H20, prefer one model replica per GPU with data parallelism before adding tensor parallelism. TP should be reserved for cases where the model or context does not fit.

Unsloth can be used for small single-GPU smoke experiments, but it should not replace the main distributed training/runtime stack until custom trajectory/tool behavior is proven compatible.

---

## 11. Efficient inference: two-runtime strategy

One runtime should optimize reproducibility; another can optimize tree search / RL throughput.

### 11.1 Runtime A: paper evaluation

Use vLLM as the stable reference runtime:

```text
vLLM
+ BF16
+ automatic prefix caching
+ CUDA Graph
+ structured output
+ no speculative decoding by default
```

This path should remain the matched evaluation reference.

### 11.2 Runtime B: tree search and RL

Use SGLang as the optimized branch runtime:

```text
SGLang
+ RadixAttention
+ session-aware cache
+ branch batching
+ structured decoding
+ optional NGRAM/EAGLE speculative decoding
```

SGLang is particularly attractive because MechET's chemical state tree and its KV-prefix tree have the same structural sharing pattern.

References:
- SGLang: https://github.com/sgl-project/sglang
- vLLM: https://github.com/vllm-project/vllm

---

## 12. Immediate vLLM cleanup

Current repository code already enables prefix caching in several inference / branch runners, which is correct.

However, historical branch runners also use `enforce_eager=True`. This should become a compatibility/debug flag rather than the default fast path, because eager mode disables CUDA Graph and compiler-based execution paths.

Matched runtime benchmark:

1. current eager vLLM;
2. non-eager vLLM;
3. non-eager + prefix cache;
4. non-eager + prefix cache + structured output;
5. optional suffix speculative decoding.

Do not change the scientific protocol during this benchmark.

Primary engineering metrics:

- reactions/hour;
- trajectory turns/s;
- branch expansions/s;
- time to first token;
- GPU utilization;
- KV-cache hit rate;
- peak KV memory;
- executor idle fraction;
- exact output-equivalence rate.

---

## 13. SGLang branch benchmark

For a fixed set of molecular states and identical sampling parameters, compare:

1. vLLM reference;
2. SGLang RadixAttention;
3. SGLang + session-aware radix cache;
4. SGLang + NGRAM speculative decoding;
5. H20 only: SGLang + TRTLLM attention backend;
6. H20 only: FP8 KV cache.

The workload must reflect actual MechET use:

- same-state K=8 branches;
- repeated multi-turn continuation;
- short structured actions;
- executor round trips.

Do not judge by generic tokens/s alone.

---

## 14. Speculative decoding priority

For MechET, short structured outputs make standard draft-model speculative decoding lower priority than prefix reuse.

Order of experimentation:

1. no speculative decoding;
2. vLLM suffix decoding or SGLang NGRAM;
3. EAGLE3 only if generation remains a material bottleneck;
4. domain-specific draft-model training only after the policy stabilizes.

The rationale is that current MechET spends substantial compute in repeated prefills and branch scheduling; speculative decoding only attacks decode time.

---

## 15. KV cache and quantization

For an 8B model, weight memory is not the main scale problem on A100/H20. Branch count, active trajectories, and context length drive KV-cache pressure.

Optimization order:

```text
prefix reuse
-> session reuse
-> branch batching
-> KV cache sizing
-> FP8 KV
-> weight quantization
```

### A100

Default scientific reference:

- BF16 weights;
- BF16 KV;
- no aggressive weight quantization.

### H20

Benchmark:

- BF16 weights + BF16 KV;
- BF16 weights + FP8 KV;
- FP8 model only if calibration and output-equivalence gates pass.

Every low-precision configuration must pass a chemistry-equivalence audit because small logit perturbations can change an early site choice and cascade through a trajectory.

---

## 16. Agent-RL infrastructure choices

### slime

Best when MechET needs direct control over:

- custom branching;
- tree rollout;
- partial trajectories;
- custom sampling;
- on-policy search.

A `generate_with_mechet.py` runner can follow slime's tool-agent examples while replacing the external tool with the MechET executor.

### Agent Lightning

Best when preserving the existing MechET runtime is more important than owning the rollout loop. It can observe an existing agent harness and extract trainable trajectories outside the runtime.

Reference:
- https://github.com/microsoft/agent-lightning

### AReaL

Best candidate only after tree rollout becomes large enough that synchronous long-tail trajectories dominate cost. Its asynchronous service architecture is attractive for highly variable trajectory lengths.

Reference:
- https://github.com/areal-project/AReaL

### Decision gate

Implement the smallest smoke for slime and Agent Lightning against the current executor. Choose one based on:

- amount of MechET runtime rewrite;
- token-ID/logprob fidelity;
- branch-tree support;
- wall-clock throughput;
- reproducibility;
- operational complexity.

Do not maintain two full RL stacks.

---

## 17. Experiment ladder

### P0 — no new scientific claim required

1. complexity-stratified audit of existing predictions;
2. vLLM non-eager/CUDA-graph benchmark;
3. structured action decoding;
4. Qwen pointer source/sink heads;
5. training packing benchmark.

### P1 — method candidate

6. chemical-state tree rollout;
7. GRPO vs GSPO vs tree sibling credit;
8. successor value;
9. dynamic all-negative filtering/resampling;
10. search-to-policy distillation.

### P2 — only if P1 exposes remaining bottlenecks

11. dedicated GNN candidate proposer;
12. graph soft-token adapter;
13. EAGLE3/domain-specific draft model;
14. fully asynchronous AReaL-scale training.

---

## 18. Required experimental controls

Every promoted method change must retain:

- identical train/valid/test split;
- same product-only inference information boundary;
- same deterministic executor;
- same terminal endpoint contract;
- identical candidate/search budgets for matched comparisons;
- explicit generated-token and executor-call accounting;
- wall-clock reporting;
- reaction-complexity stratification;
- at least one pure greedy K=1 evaluation.

For architecture changes, report performance by:

- reacting-atom count;
- changed-bond count;
- ring change;
- number of coupled electron flows;
- fragment-import count;
- trajectory length.

The new method is not successful if it only improves formal execution or tool syntax while multi-event chemistry remains unchanged.

---

## 19. Recommended near-term PR sequence

### PR-A — structured fast inference

- remove eager-mode default where safe;
- add matched vLLM CUDA-graph benchmark;
- add per-state structured output support;
- add SGLang benchmark harness;
- log reaction/hour, turns/s, expansions/s, KV-cache statistics.

### PR-B — pointer-grounded action policy

- expose atom/bond hidden states;
- source pointer head;
- sink pointer head;
- auxiliary losses;
- local and multi-flow Recall@K diagnostics;
- unchanged executor/action semantics.

### PR-C — chemical-state tree rollout

- sibling branches from one executor-owned state;
- successor equivalence pooling;
- all-negative zero-positive rule;
- GRPO / GSPO / tree-credit matched configs.

### PR-D — successor value and distillation

- endpoint-reachability value;
- tree-guided selection;
- search teacher artifact;
- search-to-policy distillation;
- greedy distilled-policy evaluation.

Only after these gates should a graph proposer be considered.

---

## 20. Success criterion

The vNext line should be promoted only if it demonstrates at least one of the following under matched compute:

1. **better complex chemistry:** gains grow with multi-centre / multi-flow transformation complexity;
2. **better sample efficiency:** same endpoint performance with fewer branch expansions;
3. **better deployment efficiency:** search gains can be distilled back into a cheaper policy;
4. **better wall-clock efficiency:** substantially more complete reaction trajectories per GPU-hour without changing chemistry outcomes.

The project should not claim novelty from adopting vLLM, SGLang, GSPO, Tree-GRPO, or any other framework in isolation. The method claim, if validated, is the integration of explicit chemical localization, electron-flow composition, deterministic molecular transitions, and state-tree credit assignment for difficult retrosynthetic transformations.
