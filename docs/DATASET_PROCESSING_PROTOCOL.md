# MechET dataset processing protocol

## Benchmark universes

The paper uses the **complete reaction-level splits** of two mechanism-annotated reaction corpora for headline one-step retrosynthesis evaluation.

### FlowER full reaction-level benchmark

- train: **257,171** reactions
- validation: **2,890** reactions
- test: **28,971** reactions
- total: **289,032** reactions

The complete endpoint export is produced by `scripts/build_flower_full_endpoint_sft.py`. No proof compilation or executor replay filtering is allowed when defining this benchmark universe.

### mech-USPTO-31k full reaction-level benchmark

- train: **24,959** reactions
- validation: **3,120** reactions
- test: **3,120** reactions
- total: **31,199** reactions

The upstream files contain elementary-step rows keyed by `rxn_idx`. The
reaction-level precursor is the complete species state at the first forward
step (`step_idx_forward = 0`, `elem_reac_spe`); `elem_reac_min` is not used
because it omits substrates introduced in later elementary steps. The desired
product proxy is the deterministic largest organic fragment of the invariant
reaction-level final mixture (`rxn_prod_min`). Each pair is mapped once with
RXNMapper 0.4.2 and then product-only canonically reindexed. Each `rxn_idx`
contributes exactly one benchmark reaction, and all external methods share the
same mapping.

## Trace-supervision subsets are not benchmark denominators

Executable inverse traces are derived from the full sources only when the current symbolic executor can replay and stitch the complete mechanism.

- FlowER executable-trace test subset: **3,080** reactions from the 28,971-reaction full test split.
- mech-USPTO current-compiler executable inverse Tool-SFT subset: **12,724** reactions total = 10,152 / 1,319 / 1,253 train/validation/test. The older 9,118 / 1,187 / 1,124 artifact is a deprecated pilot.

These subsets are used for electron-flow program supervision, program-level ablations, C1/C2/C3 composition analysis, and trace-specific diagnostics. They must **never** replace the full FlowER or full mech-USPTO reaction-level universes in the headline endpoint benchmark.

## External baseline contract

All matched external baselines train and evaluate on the **full reaction-level split** of each corpus. They receive the same reaction IDs and the same product-to-precursor task. Each method may derive only the supervision required by its published method from those full reaction pairs.

- LocalRetro: derive train-only local reaction templates from full train reactions.
- R-SMILES: derive root-aligned product/reactant strings from full train reactions.
- EditRetro: derive oracle edit actions from full train product/reactant pairs.
- RetroBridge: train the published product-to-reactant graph bridge on full train pairs.
- ReactSeq: derive mapped/kekulized ReactSeq targets from full train reaction pairs.
- RETRO SYNFLOW: train the published reaction-center/synthon/flow components on full train pairs.
- RxnNano: use the published retrosynthesis training recipe on full train pairs; do not add MechET trace supervision.
- Retro-MTGR: derive its native reaction-center/leaving-group targets from full train pairs.

No external baseline may read MechET electron-flow traces, executor states, legal-action lists, or C1/C2/C3 labels during ordinary headline training.

## MechET training views

MechET may use two linked views of the same source corpus:

1. **full endpoint view** — all reaction-level rows, used to preserve full-dataset coverage and endpoint supervision;
2. **executable trace view** — the subset for which an inverse electron-flow program can be replayed under the current executor, used for program supervision.

Headline endpoint evaluation always uses the full test denominator. Program-execution and gold-program metrics must report both the full denominator and the trace-covered subset denominator where applicable.

## Split integrity

The upstream train/validation/test assignments are frozen. Do not create new random splits for external baselines. Every exported row must preserve:

- `stable_id`
- source dataset
- source split
- reaction-level source ID (`trajectory_id` for FlowER; `rxn_idx` for mech-USPTO)
- product
- precursor/reference reactants

All downstream method-specific preprocessing must start from these frozen exports.

## Canonicalization and atom addressing

Cross-method endpoint evaluation removes atom maps before structural comparison, canonicalizes with RDKit, preserves stereochemistry, and treats disconnected fragment order as invariant.

MechET may assign deterministic atom addresses internally because its executor needs local references. Those addresses are generated from the product alone and are nuisance coordinates, not reaction-derived predictive features. Original reaction-map integers are not used as model evidence in headline inference.

## Metrics

The common cross-method metric is **Success@K**: whether the reference precursor appears among at most `K` candidates. Method-native beam search, ranked sampling, or stochastic generation may be used, but candidate semantics must be recorded.

MechET-specific independent-sampling `Pass@K` and program-execution metrics are reported separately and are not relabeled as ranked Top-K.

## Required artifacts

For each corpus freeze:

```text
data/external_baselines/<dataset>/
  train.jsonl
  valid.jsonl
  test.jsonl
  manifest.json
```

The manifest records row counts, stable-ID hashes, source hashes, preprocessing revision, and any excluded/quarantined rows. For the full benchmark exports, mechanism replay incompatibility is **not** an exclusion criterion.

The FlowER handoff is generated independently of mech-USPTO readiness:

```bash
python scripts/export_full_baseline_pairs.py --datasets flower_full
```

The shared JSONL contains mapped and unmapped product/precursor views plus the
complete mapped reaction string. Method-specific code may derive templates,
root alignment, edit operations, or graph pairs from these rows, but may not
change the frozen split or silently drop test IDs.
