# MechET external baseline execution plan

## Purpose

This document defines the **published external retrosynthesis baselines** to be reproduced for the MechET paper, separates them from MechET-specific ablations, and divides the workload between two collaborators.

The goal is not to reproduce every historical retrosynthesis model. The goal is to cover the main inductive biases that a reviewer could reasonably argue already solve the same one-step inverse-prediction problem:

1. local reaction templates;
2. aligned sequence translation;
3. iterative molecular editing;
4. graph-generative modeling;
5. reaction-center/leaving-group factorization;
6. explicit reaction-operation languages;
7. flow-based diverse/feasibility-guided generation;
8. reasoning-oriented reaction LLMs.

**Important:** any method invented inside MechET (direct-Qwen control, generic edit control, one-shot electron-flow generation, no-enumeration, stale feedback, no-feedback, independent answer channel, etc.) is an **internal ablation/control**, not an external baseline.

---

## 1. Frozen evaluation contract

All external baselines must ultimately be evaluated by the **same MechET-side evaluator**, not by each external repository's own top-k script alone.

### Datasets

Primary datasets:

- FlowER-derived inverse benchmark;
- mech-USPTO-31k inverse benchmark.

Use the frozen train/validation/test reaction IDs already maintained by MechET. Do not re-split reactions independently inside each baseline repository.

### Common input/output contract

Create one adapter per baseline that exports predictions to:

```text
outputs/external_baselines/<method>/<dataset>/predictions.jsonl
```

Each record should contain at least:

```json
{
  "stable_id": "...",
  "product": "...",
  "reference_precursors": "...",
  "candidates": [
    {
      "rank": 1,
      "precursors": "...",
      "score": 0.0
    }
  ],
  "runtime_ms": 0.0,
  "source_method": "...",
  "checkpoint": "..."
}
```

Preserve atom maps if the model emits them, but do not require all external models to output mapped precursors. The centralized evaluator will separately compute structural exact match, mapped exact where applicable, full-precursor exact match where meaningful, round-trip plausibility, missing-prediction rate and invalid-SMILES rate.

### Required top-k

At minimum export top-10 candidates per target whenever the method supports ranked sampling/beam search. The paper can report P@1/3/5/10 from the same artifacts.

### Fairness rules

- Same train/valid/test reaction IDs for all retrained methods.
- No reaction class label unless MechET also receives it in that experiment.
- Do not use a method's USPTO-50K test checkpoint as a matched result on our datasets.
- Official checkpoints may be used only in a clearly labeled **external-checkpoint reference** block.
- Do not manually improve a baseline architecture. Only adapt I/O and data loading as needed.
- If a published implementation fundamentally cannot support the dataset after reasonable engineering effort, record the blocker rather than silently replacing the method with a home-made approximation.

---

## 2. Baseline audit

### 2.1 LocalRetro — P0

**Paper:** Chen & Jung, JACS Au 2021  
**Official code:** https://github.com/kaist-amsg/LocalRetro  
**Paradigm:** local reaction templates + global graph context.

Why it matters:

- Directly tests whether MechET's advantage is already explained by the locality of reaction changes.
- A strong and mature template-based reference.

Implementation notes:

- Official code supports USPTO-50K and USPTO-MIT.
- Requires atom-mapped reactions for template extraction.
- Training pipeline is explicit: template extraction -> preprocessing -> training -> decoding.
- Official README reports roughly 100 min training on a 3090 for USPTO-50K, so our smaller mechanistic datasets should be manageable.

Adaptation needed:

1. export MechET splits to LocalRetro-style atom-mapped `reactants>>products` files;
2. derive templates **only from the training split**;
3. run official preprocessing/training/decoding;
4. convert decoded predictions into the common JSONL format.

Risk: **low-medium**. Main issue is template coverage on the smaller mechanistic corpora.

---

### 2.2 R-SMILES — P0

**Paper:** Root-aligned SMILES: A Tight Representation for Chemical Reaction Prediction  
**Official code:** https://github.com/otori-bird/retrosynthesis  
**Paradigm:** sequence translation with root-aligned product/reactant SMILES.

Why it matters:

- Tests a strong alternative explanation: perhaps the main difficulty is only sequence misalignment, not reaction-program representation.
- Provides the natural sequence baseline between direct SMILES generation and MechET.

Adaptation needed:

1. convert frozen train/valid/test reactions to the repository's raw format;
2. generate root-aligned source/target pairs using the official preprocessing;
3. train using the provided retrosynthesis recipe;
4. export beam predictions to the common JSONL format.

Risk: **medium**. The codebase is older and data preprocessing is representation-specific, but the task itself is directly compatible.

---

### 2.3 EditRetro — P0

**Paper:** Nature Communications 2024  
**Official code:** https://github.com/yuqianghan/editretro  
**Paradigm:** iterative string editing.

Why it matters:

- Probably the most important non-mechanistic comparator for MechET's sparse-residual claim.
- Tests whether explicit local editing alone is sufficient without electron-flow semantics/execution.

Adaptation needed:

1. reuse official preprocessing while replacing the dataset split with MechET frozen IDs;
2. train the official architecture without architecture changes;
3. run iterative generation and preserve the method's native ranking score;
4. export top-k precursor predictions.

Risk: **medium**. The repository is complete and has public checkpoints/scripts, but preprocessing and iterative decoding need careful adaptation.

---

### 2.4 RetroBridge — P0

**Paper:** ICLR 2024  
**Official code:** https://github.com/igashov/RetroBridge  
**Paradigm:** graph-generative Markov bridge.

Why it matters:

- Represents a modern non-autoregressive/template-free graph-generative baseline.
- Tests whether MechET gains are specific to an LLM/autoregressive decoder.

Implementation notes:

- Official training, sampling and evaluation code are public.
- Model checkpoints are available, but matched experiments require retraining on our splits.
- Sampling is relatively expensive because the published recipe uses many bridge steps.

Adaptation needed:

1. create a dataset object from frozen product/reactant pairs;
2. retrain with the official architecture;
3. sample top-k candidates using a fixed sampling budget;
4. export candidates before any method-specific round-trip reranking so the MechET-side evaluator remains authoritative.

Risk: **medium-high** due to sampling cost, not conceptual incompatibility.

---

### 2.5 Retro-MTGR — P1

**Paper:** Nature Communications 2025  
**Official code:** https://github.com/zpczaizheli/Retro-MTGR  
**Paradigm:** multi-task graph representation learning for reaction-center deduction + reactant/leaving-group recommendation.

Why it matters:

- Gives a strong explicit decomposition baseline for reaction center and precursor realization.
- Useful for the paper's failure-mode analysis separating disconnection vs leaving-group errors.

Implementation notes:

- Public code supports USPTO-50K and USPTO-MIT style datasets.
- Code is more legacy/hard-coded than the other repositories (Python 3.7 / file-path edits / index-range split specification).

Risk: **medium-high engineering risk**, but training itself is not expected to be the bottleneck.

Recommendation: run after P0 baselines are stable.

---

### 2.6 ReactSeq — P0

**Paper:** Nature Machine Intelligence 2025  
**Official code:** https://github.com/jiachengxiong/ReactSeq  
**Paradigm:** explicit reaction-description language built from stepwise molecular editing operations.

Why it matters:

- The closest published competitor to MechET at the level of an explicit transformation language.
- Essential for showing that MechET is not simply another reaction tokenization scheme.

Implementation notes:

- Full data generation, training, inference and ReactSeq-to-reactant transformation code are public.
- Requires **mapped and kekulized reaction SMILES** for ReactSeq generation.
- Uses two environments in the official recipe: OpenNMT/PyTorch for training and older RDKit/Indigo for preprocessing/transformation.
- Official recipe uses heavy data augmentation; for fairness, record exact augmentation and keep the same augmentation policy across both mechanistic corpora.

Adaptation needed:

1. export mapped reactions and kekulize using the official transformation logic;
2. generate ReactSeq targets;
3. train with official OpenNMT configuration;
4. decode ReactSeq and transform predictions back to reactant SMILES;
5. export to common JSONL.

Risk: **medium-high**, mostly environment/preprocessing complexity.

---

### 2.7 RETRO SYNFLOW — P0 (heavy)

**Paper:** NeurIPS 2025  
**Official code:** https://github.com/DSL-Lab/RetroSynFlow  
**Paradigm:** discrete flow matching for accurate/diverse retrosynthesis, with optional feasibility steering.

Why it matters:

- Strong modern comparison for accuracy + diversity + learned feasibility guidance.
- Particularly important for the claim that MechET provides an internal transformation-consistency signal rather than relying only on a learned forward oracle.

Implementation notes:

- Clean public experiment API.
- Official example trains the product->reactant flow model for hundreds of epochs and uses multi-GPU jobs.
- Synthon-completion variants additionally require a reaction-center model.

Recommended matched condition:

- first reproduce the direct `product -> reactants` GraphDiscreteFM setting;
- treat reward-steered inference as a secondary condition if the required forward model is available and domain-compatible.

Risk: **high compute**, low conceptual risk.

---

### 2.8 RxnNano — P0/P1 runnable LLM comparator

**Paper/preprint:** 2026  
**Official code:** https://github.com/rlisml/RxnNano  
**Paradigm:** compact reaction LLM with hierarchical curriculum, latent chemical consistency and atom-map permutation invariance.

Why it matters:

- It is a much more practical runnable LLM comparator than a paper with no released training code.
- Directly supports mapped retrosynthesis JSONL and LoRA training.
- Its atom-map permutation invariance makes it especially relevant to MechET's map-nuisance controls.

Implementation notes:

- Public training command supports mapped retrosynthesis directly.
- Default example uses Qwen2.5-7B-Instruct + LoRA.
- Data JSONL is simple: `product`, `reactants`, optional `rxn_Class`.

Risk: **medium compute**, low engineering risk.

Recommendation: include as the primary **runnable external LLM** if resources permit.

---

### 2.9 RetroDFM-R — external-checkpoint reference first

**Paper:** 2025  
**Official code:** https://github.com/OpenDFM/RetroDFM-R  
**Paradigm:** retrosynthesis reasoning LLM with continual pretraining, cold-start distillation and RL.

Why it matters:

- Strong reasoning-based retrosynthesis reference.
- Official 8B checkpoint and inference code are public.

Why not make matched retraining P0:

- The published pipeline includes multiple large stages (continual pretraining, distillation, RL/OpenRLHF).
- Re-running the full recipe on our 11k-scale mechanistic dataset would be expensive and would not be a clean architecture-only comparison.

Recommended use:

1. use the official checkpoint as an **external-checkpoint reference** on an exact-clean compatible target set;
2. do not use it to identify the causal effect of MechET's representation;
3. only attempt matched retraining if later reviewer pressure makes it necessary.

Risk: **high compute for retraining; low risk for inference-only reference**.

---

### 2.10 RetroReasoner — contextual until official code is verified

**Paper/preprint:** 2026, arXiv:2603.12666  
**Paradigm:** structured disconnection reasoning + SFT + round-trip RL.

It is highly relevant scientifically, but at the time of this audit I did **not verify a public official training repository** from the paper/search results.

Recommendation:

- keep in Related Work;
- do not assign a collaborator to reimplement it from the paper;
- add it to the runnable benchmark only if official code/checkpoints become available.

Do **not** create a home-made 'RetroReasoner-like' baseline and label it RetroReasoner.

---

## 3. Recommended final comparator tiers

### P0 — should be completed for the main empirical comparison

1. LocalRetro
2. R-SMILES
3. EditRetro
4. RetroBridge
5. ReactSeq
6. RETRO SYNFLOW
7. RxnNano (runnable LLM comparator)

### P1 — useful strengthening

8. Retro-MTGR
9. RetroDFM-R official checkpoint on an external/compatible evaluation setting

### Context only unless implementation becomes available

10. RetroReasoner
11. FlowER native forward model
12. MechSMILES native mechanism model

FlowER and MechSMILES are neighboring mechanistic works, not native product-only retrosynthesis baselines.

---

## 4. Assignment to two collaborators

The split below is designed to balance **engineering complexity + GPU cost**, not simply the number of models.

### Collaborator A

#### A1. LocalRetro — P0
- expected engineering: low-medium
- expected compute: low
- first target: mech-USPTO
- second target: FlowER-derived

#### A2. ReactSeq — P0
- expected engineering: medium-high due to dual environments + reaction transformation
- expected compute: medium
- special responsibility: validate mapped/kekulized conversion on 100 reactions before training

#### A3. RetroBridge — P0
- expected engineering: medium
- expected compute: medium-high at inference
- special responsibility: ensure sampling budget is fixed and recorded

#### A4. RxnNano — P0/P1
- expected engineering: low-medium
- expected compute: medium (7B LoRA)
- special responsibility: report mapped vs map-permuted evaluation using the same checkpoint

**Collaborator A deliverable:** template/edit-language/graph-generation/LLM breadth.

---

### Collaborator B

#### B1. R-SMILES — P0
- expected engineering: medium
- expected compute: medium
- special responsibility: verify root-alignment preprocessing preserves the frozen split and reaction identity

#### B2. EditRetro — P0
- expected engineering: medium
- expected compute: medium
- special responsibility: preserve native iterative decoding and ranking

#### B3. RETRO SYNFLOW — P0 heavy
- expected engineering: medium
- expected compute: high
- special responsibility: first run direct product->reactant flow matching; add reward steering only as a secondary condition

#### B4. Retro-MTGR — P1
- expected engineering: medium-high because of legacy/hard-coded data loading
- expected compute: low-medium
- special responsibility: use frozen IDs rather than repository index-based train/test ranges

#### B5. RetroDFM-R — reference only
- no matched retraining initially
- run official 8B checkpoint inference on the agreed exact-clean external set
- export outputs to the common JSONL format

**Collaborator B deliverable:** sequence/edit/flow/factorized/reasoning breadth.

---

## 5. Shared milestones

### Milestone 0 — common data adapter

Before either collaborator trains a model, freeze shared exports:

```text
data/external_baselines/
  flower_inverse/{train,valid,test}.csv
  mech_uspto_inverse/{train,valid,test}.csv
```

Minimum columns:

```text
stable_id,split,rxn_smiles,product,reactants
```

`rxn_smiles` should remain atom-mapped. Also provide unmapped `product` and `reactants` if a baseline expects unmapped strings.

### Milestone 1 — 100-reaction preprocessing audit

For every method:

- preprocess 100 reactions;
- verify zero cross-split leakage;
- verify product/reactant identity after any canonicalization;
- verify the method can decode at least one syntactically valid prediction;
- save the exact external repository commit SHA and environment lockfile.

No full training should start before this passes.

### Milestone 2 — overfit/smoke test

- train on a tiny subset (32-128 reactions);
- confirm training loss decreases;
- confirm inference pipeline reaches final precursor SMILES;
- confirm predictions can be converted to common JSONL.

### Milestone 3 — mech-USPTO full run

Run the full frozen mech-USPTO split first. It is smaller and is the cleanest place to catch task adaptation bugs.

### Milestone 4 — FlowER-derived full run

Repeat with the frozen FlowER-derived split using exactly the same method configuration unless a dataset-size hyperparameter must change.

### Milestone 5 — centralized evaluation

Do **not** manually paste numbers from external repositories into the paper. All final tables should be generated from common prediction JSONLs by one MechET-side evaluation script.

---

## 6. Required artifact checklist per baseline

Each completed method must provide:

```text
external_baselines/<method>/
  README_RUN.md
  upstream.json
  environment.yml or requirements.lock
  data_adapter.py
  train_command.sh
  infer_command.sh
  flower_inverse/predictions.jsonl
  mech_uspto_inverse/predictions.jsonl
  logs/
```

`upstream.json` must include:

```json
{
  "repository": "...",
  "commit": "...",
  "license": "...",
  "paper": "...",
  "modified_files": ["..."],
  "notes": "only data/I-O adaptation; no architecture changes"
}
```

---

## 7. Stop / escalation rules

A collaborator should stop and document the blocker instead of rewriting the published method if any of the following happens:

1. official preprocessing assumes unavailable labels that would leak target information;
2. the method fundamentally requires a reaction class not supplied to MechET;
3. adapting the dataset would require replacing the published model architecture;
4. decoding cannot be mapped back to precursor SMILES without an undocumented component;
5. full reproduction would require an unreasonable training pipeline relative to its role as a baseline (RetroDFM-R is the clearest example).

In those cases, move the method to the external-checkpoint/context tier and replace it with another **published runnable method**, not a home-made surrogate.

---

## 8. Recommended paper-facing baseline set after this audit

For the main table, the most defensible executable set is:

```text
LocalRetro
R-SMILES
EditRetro
RetroBridge
ReactSeq
RETRO SYNFLOW
RxnNano
MechET
```

Add Retro-MTGR if the legacy implementation adapts cleanly.

Use RetroDFM-R in a separated external-checkpoint block unless matched retraining is later justified.

Keep RetroReasoner in Related Work until official code/checkpoints are verified.

This list gives substantially better experimental coverage than adding multiple home-made Qwen baselines. Internal MechET variants should appear only in the ablation table, where they answer representation/execution questions rather than SOTA-comparison questions.
