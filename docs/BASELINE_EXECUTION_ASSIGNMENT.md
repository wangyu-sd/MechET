# MechET external baseline execution plan

## 0. Non-negotiable dataset definition

Headline external-baseline experiments use the **complete reaction-level datasets**, not the executable-trace subsets.

| Dataset | Train | Valid | Test | Total | Headline use |
|---|---:|---:|---:|---:|---|
| FlowER full reaction-level | 257,171 | 2,890 | 28,971 | 289,032 | full matched retrosynthesis benchmark |
| mech-USPTO-31k full reaction-level | 24,959 | 3,120 | 3,120 | 31,199 | full matched retrosynthesis benchmark |

The smaller executable subsets have a different role:

- FlowER executable-trace test subset: 3,080 / 28,971 test reactions;
- mech-USPTO current executable inverse Tool-SFT subset: 10,152 / 1,319 / 1,253 = 12,724 reactions. The old 9,118 / 1,187 / 1,124 artifact is a deprecated pilot.

Those subsets are used only for **electron-flow program supervision and program-level analysis**. They are not the main benchmark denominator and external baselines must not be trained only on them.

Authoritative processing details: [`DATASET_PROCESSING_PROTOCOL.md`](DATASET_PROCESSING_PROTOCOL.md).

---

## 1. Purpose

This document defines the published external retrosynthesis baselines to reproduce for the MechET paper and divides the work between two collaborators.

External methods establish field-level competitiveness. Any method created inside MechET---direct Qwen, generic sparse edit, one-shot electron flow, no enumeration, stale feedback, no feedback, independent answer channel---is an **internal control/ablation**, not an external baseline.

All matched external methods receive the same complete reaction IDs and the same product-to-precursor task. Each method retains its **published native training target**; no external method receives MechET electron-flow traces or executor states.

---

## 2. Frozen common reaction exports

All baseline preprocessing must start from shared exports:

```text
data/external_baselines/
  flower_full/
    train.jsonl
    valid.jsonl
    test.jsonl
    manifest.json
  mech_uspto_31k_full/
    train.jsonl
    valid.jsonl
    test.jsonl
    manifest.json
```

FlowER full reaction-level data are built with:

```bash
python scripts/build_flower_full_endpoint_sft.py \
  --data-root /path/to/flower/data \
  --output-dir data/flower_full_endpoint_sft \
  --splits train valid test
```

Freeze the shared, method-agnostic FlowER handoff with:

```bash
python scripts/export_full_baseline_pairs.py \
  --datasets flower_full \
  --flower-dir data/flower_full_endpoint_sft \
  --output-root data/external_baselines
```

Every external repository must start from the resulting
`data/external_baselines/flower_full/{train,valid,test}.jsonl` and preserve its
`stable_id`. It may then derive its published native representation. Do not
start from `textbook_tool_sft`, `flower_inverse_tool_sft`, or the 3,080-row
trace test view.

mech-USPTO full reaction-level data are built with:

```bash
python scripts/build_mech_uspto_full_endpoint_sft.py \
  --data-root data/raw/mech_uspto_31k/data \
  --output-dir data/mech_uspto_31k_full_endpoint_sft
```

Do not re-split reactions inside an external repository.

---

## 3. What each baseline is actually trained to predict

The reaction universe is shared; the supervision representation is method-native.

| Method | Training input | Native training target derived from the full train reactions | MechET trace allowed? | Priority |
|---|---|---|:---:|---:|
| LocalRetro | product graph | local atom/bond reaction-template labels extracted from train reactions only | No | P0 |
| R-SMILES | root-aligned product SMILES | root-aligned precursor SMILES | No | P0 |
| EditRetro | product string | official oracle edit sequence / iterative edit supervision | No | P0 |
| RetroBridge | product graph | paired precursor graph under the published Markov-bridge objective | No | P0 |
| ReactSeq | mapped+kekulized product/reaction representation | official ReactSeq operation sequence derived from product/precursor pair | No | P0 |
| RETRO SYNFLOW | product graph / synthons | published reaction-center + synthon + flow-matching supervision | No | P0-heavy |
| RxnNano | product prompt | published mapped-retrosynthesis target under its native curriculum/consistency recipe | No | P0/P1 |
| Retro-MTGR | product graph | reaction-center + leaving-group/reactant multi-task labels | No | P1 |
| RetroDFM-R | official model input | official pretrained model only at first; no home-made reduced recipe | No | contextual |
| RetroReasoner | official model input | use only when official training/checkpoint path is verifiable | No | contextual |

### Important methodological boundary

Do **not** convert every baseline into a generic `product -> reactants` Transformer and keep its paper name. For example:

- LocalRetro must still learn local templates;
- EditRetro must still learn its iterative edit policy;
- ReactSeq must still learn ReactSeq operations;
- RETRO SYNFLOW must include its published reaction-center/synthon flow formulation if it is labeled RETRO SYNFLOW.

If only a reduced component is runnable, report the component by its actual name rather than attributing it to the full published method.

---

## 4. P0 external baseline set

### LocalRetro

Role: tests whether local reaction templates already explain MechET's gains.

Required procedure:
1. convert the **full** FlowER or mech-USPTO training split to the official atom-mapped reaction format;
2. extract templates from the training split only;
3. train official LocalRetro;
4. decode the corresponding full test split;
5. export candidates to the shared MechET evaluator.

### R-SMILES

Role: tests whether product/reactant sequence alignment alone is sufficient.

Required procedure:
1. use the complete train reaction pairs;
2. apply official root alignment;
3. train the published retrosynthesis configuration;
4. beam-decode the complete test split.

### EditRetro

Role: strongest sparse-edit comparison.

Required procedure:
1. derive official oracle edits from every full-train product/precursor pair;
2. preserve the original staged/iterative training procedure;
3. preserve native iterative decoding and ranking;
4. evaluate on every full-test reaction.

### RetroBridge

Role: modern graph-generative comparison independent of an LLM decoder.

Required procedure:
1. construct product/reactant graph pairs for all full-train reactions;
2. train the official bridge model;
3. use a frozen sampling budget for the complete test set;
4. export candidates before any gold-dependent processing.

### ReactSeq

Role: closest published reaction-operation language baseline.

Required procedure:
1. convert every full-train reaction to the mapped/kekulized form expected by the official code;
2. derive ReactSeq targets using the official transformer;
3. train the official sequence model;
4. transform decoded ReactSeq predictions back to precursor SMILES;
5. evaluate all full-test IDs.

### RETRO SYNFLOW

Role: strong flow-based accuracy/diversity comparison.

Required matched condition:
1. train the published reaction-center component on the full train split;
2. construct synthons using that published pipeline;
3. train the synthon-to-reactant flow model on the full train split;
4. evaluate the complete test split under a fixed candidate budget.

Forward-oracle/Feynman--Kac steering is a secondary condition if its reward model can be reproduced without cross-dataset leakage.

A plain direct `GraphDiscreteFM(product -> reactants)` run is not labeled full RETRO SYNFLOW.

### RxnNano

Role: runnable reaction-LLM comparison, especially relevant to atom-map nuisance controls.

Use the published mapped-retrosynthesis recipe and its native curriculum/consistency setup. A generic Qwen LoRA trained only on our pairs is an internal backbone control, not RxnNano.

---

## 5. P1 / contextual methods

### Retro-MTGR — P1

Train its native reaction-center and leaving-group/reactant multi-task objectives on the full train split. Do not use repository index ranges that create a new split.

### RetroDFM-R — contextual first

Use the official checkpoint for an explicitly separated external-pretrained reference. Do not place it in the matched block unless the full published training recipe is reproduced on our full reaction split.

### RetroReasoner — contextual

Keep in Related Work unless an official checkpoint/training path is available. Do not implement a home-made `RetroReasoner-like` system and label it RetroReasoner.

### FlowER and MechSMILES

Their native tasks are mechanistic/forward or mechanism-reconstruction tasks, not the matched product-only retrosynthesis task. They remain mechanistic neighboring work. Any reverse adaptation is an internal task adaptation, not an existing external baseline.

---

## 6. Shared output contract

Every baseline exports:

```text
outputs/external_baselines/<method>/<dataset>/predictions.jsonl
```

Each row contains at least:

```json
{
  "stable_id": "...",
  "product": "...",
  "reference_precursors": "...",
  "candidates": [
    {"rank": 1, "precursors": "...", "score": 0.0}
  ],
  "runtime_ms": 0.0,
  "source_method": "...",
  "checkpoint": "..."
}
```

At least top-10 candidates are exported when the method supports them. Missing predictions remain failures in the full test denominator.

All final endpoint scores are recomputed with one MechET-side evaluator.

Cross-method metrics:
- Success@1/3/5/10;
- invalid-output rate;
- missing-output rate;
- full precursor / structural precursor views where source annotations permit;
- forward round-trip consistency as a separate model-based diagnostic.

Do not rename independent stochastic Pass@K as beam Top-K.

---

## 7. Assignment to two collaborators

The split balances engineering effort and GPU cost.

### Collaborator A

1. **LocalRetro — P0**
   - full mech-USPTO first, then full FlowER;
   - verify train-only template extraction.

2. **ReactSeq — P0**
   - full mech-USPTO + full FlowER;
   - first validate mapped/kekulized conversion on 100 reactions.

3. **RetroBridge — P0**
   - full mech-USPTO + full FlowER;
   - record sampling steps and candidate budget.

4. **RxnNano — P0/P1**
   - full datasets under the official mapped-retro recipe;
   - report canonical-map and random-map test views.

### Collaborator B

1. **R-SMILES — P0**
   - full mech-USPTO + full FlowER;
   - verify root alignment preserves stable IDs.

2. **EditRetro — P0**
   - full mech-USPTO + full FlowER;
   - preserve official edit targets and iterative decoding.

3. **RETRO SYNFLOW — P0-heavy**
   - full mech-USPTO + full FlowER;
   - reaction-center + synthon flow first; steering secondary.

4. **Retro-MTGR — P1**
   - full reaction universes only.

5. **RetroDFM-R — contextual**
   - official checkpoint inference only until a matched full-data reproduction is justified.

---

## 8. Milestones before full training

### Milestone 0 — freeze full dataset manifests

Required counts:

```text
FlowER:       257171 / 2890 / 28971
mech-USPTO:    24959 / 3120 / 3120
```

No baseline training begins until these counts and stable-ID hashes are frozen.

### Milestone 1 — 100-reaction preprocessing audit

For each method:
- run official preprocessing on 100 frozen reactions;
- verify stable IDs survive preprocessing;
- verify product and precursor identity after representation conversion;
- verify no validation/test rows enter train-derived vocabularies/templates;
- save external repository commit SHA and environment lockfile.

### Milestone 2 — 32--128 reaction overfit test

Confirm:
- training loss decreases;
- inference returns syntactically valid candidates;
- prediction candidates map back to the original stable IDs;
- common JSONL export works.

### Milestone 3 — full mech-USPTO

Run all assigned P0 methods on 24,959 / 3,120 / 3,120.

### Milestone 4 — full FlowER

Run all assigned P0 methods on 257,171 / 2,890 / 28,971.

---

## 9. Where executable subsets are used

The 3,080 FlowER trace test cases and 12,724 current-compiler mech-USPTO inverse traces are reserved for questions that require a gold executable program:

- one-shot electron-flow vs closed-loop MechET;
- no-enumeration / stale-feedback controls;
- program execution and endpoint--program consistency;
- C1/C2/C3 primitive/motif composition splits;
- detailed electron-flow trajectory analysis.

External field-level baseline performance is **not** restricted to those subsets.
