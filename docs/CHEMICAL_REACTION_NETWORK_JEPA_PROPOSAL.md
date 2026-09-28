# Mechanism-conditioned reaction-state JEPA: research proposal

**Status:** proposal for discussion, not an implemented method, completed experiment, or change to the current MechET/ICLR protocol. This document does not authorize a training job. It extends the experimental graph-policy direction in [PR #65](https://github.com/wangyu-sd/MechET/pull/65); it does not replace the trace-owned main method or the separate EARHO work.

## Scientific question and precise scope

Can a model learn *which chemically legal electron-flow decisions lead toward useful reaction outcomes*, while sharing a representation across retrosynthesis, forward reaction prediction, reaction-condition selection, transition-state proposal, and gas-phase fragmentation?

The proposed object is a **mechanism-conditioned reaction-state network**. A node is an entire chemical state (a multiset of molecular graphs, charges, electron containers, provenance of imported species, and optional context). An edge is a typed, executor-checkable event. Branches represent alternative mechanisms and products. This is not yet a quantitative chemical reaction network with experimentally determined concentrations, rate constants, or microkinetics; those require additional data and models.

The current paper's [scientific thesis](SCIENTIFIC_THESIS.md) is narrower: computational, executor-owned inverse electron-flow reasoning for a declared organic-chemistry scope. Gas-phase ions, radicals, condition effects, and transition-state energetics below are **new tasks with separate data and validation contracts**, not capabilities inherited from that thesis.

## Model and inference contract

Let `x_t` contain the full current state and its domain/context `c`: solution versus gas phase, reaction conditions if known, ion/adduct and collision energy for MS, and provenance for atoms entering via fragment imports. The shared atom/bond encoder `E` is permutation-equivariant; a graph-level readout is permutation-invariant. Private atom maps may align training states and executor handles but their integer values are not learned features.

```text
z_t = E(x_t, c)
candidate actions a_t ~ policy(a | z_t, compact history, task)
x_(t+1) = exact domain executor(x_t, a_t)
predicted future latent zhat_(t+h) = F_h(z_t, a_t ... a_(t+h-1), c)
```

The action-conditioned JEPA predictor `F_h` matches a stop-gradient/EMA target encoder of actual future states for short and longer horizons (for example 1, 2 and 4 accepted events). The loss includes **atom- and bond-local targets as well as graph-level targets**: a graph-only embedding could ignore the very bond-order, charge, or radical change that defines an electron event. Action prediction, inverse-action consistency, and variance/covariance anti-collapse regularization complement latent prediction. Given an action sequence, future states are less multimodal; where actions are unknown, use multiple hypotheses rather than averaging mutually exclusive products.

An exact executor, not the latent predictor, defines committed chemical states. The useful role of JEPA is to rank actions by predicted *downstream consequence* and to provide a long-horizon representation/value signal. If it merely learns to imitate the already-available deterministic one-step executor, it adds cost without a new capability. At inference, a small policy-guided search uses the predictor for proposal/ranking and checks selected actions with the executor; every reported structure comes from execution. This is a testable design hypothesis, not a performance claim. Action-conditioned latent planning has precedents in [V-JEPA 2](https://arxiv.org/abs/2506.09985), but its transfer to chemistry is unproven.

The core graph policy can reuse the factorized `FLOW`, sparse `BE_DELTA`, `IMPORT_REACTIVE`, `IMPORT_ENV`, and `FINISH` action families proposed in [PR #65](https://github.com/wangyu-sd/MechET/pull/65). Open-vocabulary reactive imports need an explicit graph generator and independent validity/coverage metrics; a small closed fragment catalog must not silently become the main model. The shared encoder can be large while each domain has a small, typed dynamics/action head. One undifferentiated transition head for solution reactions and collision-induced gas-phase fragmentation is not chemically justified.

## What “chemically credible” means

Confidence must be attached to a **specific claim and evidence level**, not to one opaque model probability. The proposed output is an auditable record: structures, conditions or ionization context, proposed events, executor outcomes, provenance, evidence, calibrated uncertainty, and explicit abstention reason where necessary.

| Level | Evidence and check | What it supports | What it does **not** prove |
|---|---|---|---|
| 0. Formal accounting | Typed action replay; atom/isotope provenance; bond-electron, formal-charge and spin bookkeeping; explicit reservoirs for imports, protons, electrons and neutral losses; graph sanitization and cycle checks | Internally consistent state transformation | That the reaction occurs under real conditions |
| 1. Mechanistic consistency | Reverse-path replay plus a separately trained forward predictor or other independent check; known reaction class and competing branches; sensitivity to required reagents/conditions | A coherent computational mechanism hypothesis | A unique physical mechanism or selectivity |
| 2. Computational physics | Appropriate conformer/solvation/ionization model; lower-cost electronic-structure screening, then targeted higher-level calculations; barrier and energy uncertainty | Evidence about particular steps under a declared approximation | Universal feasibility across solvents, temperatures or instruments |
| 3. External empirical agreement | Independent literature reactions, held-out condition/yield measurements, known products, reference spectra and expert adjudication, with provenance and non-overlap audit | Out-of-sample predictive reliability in a stated domain | Guarantee for every new molecule |
| 4. Prospective confirmation | Carefully selected new synthesis, spectroscopy or mechanistic experiments | Stronger evidence for the tested cases | A universal certificate for all model outputs |

Wet-lab testing is therefore **not a prerequisite for every prediction or for an initial research result**. Credible pre-experimental claims can rest on Levels 0–3, but must name the level reached. A forward-model score, RetroChimera rank, or JEPA distance is soft evidence, not a law. A round trip through the *same* executor is an integrity check, not independent confirmation. Even an algebraically reversible inverse program does not mean the forward reaction is kinetically accessible under the same conditions. FlowER's BE-matrix work motivates electron accounting, not a blanket claim of experimental mechanism identification ([FlowER](https://www.nature.com/articles/s41586-025-09426-9)).

For a transition-state *claim*, the bar is more specific: a proposed 3D geometry is a candidate until an electronic-structure calculation verifies a first-order saddle point and an intrinsic reaction coordinate (or comparably justified path check) connects the intended endpoints. Recent automated TS-search work uses this two-endpoint criterion ([Meissner et al., 2026](https://www.nature.com/articles/s41524-026-02301-9)); graph-level electron edits alone cannot meet it.

## Task adapters: shared representation, distinct evidence

| Task | Input and predicted object | Additional model/data requirement | Minimum convincing evaluation |
|---|---|---|---|
| Retrosynthesis | Product-only graph → executable inverse-event trajectory → complete precursor mixture | Mechanism trajectories, fragment-import generator, optional endpoint preference teacher | Formal acceptance and full-mixture Top-1/Top-K on frozen denominators; alternative-route review; diversity and cost |
| Forward reaction simulation | Reactant/reagent state plus conditions → branching electron-event trajectories and product distribution | Independently trained forward policy; conditions and side-product data | Product/side-product recovery, selectivity calibration, unseen-class transfer, conservation and nontrivial trajectory checks |
| Condition selection | Specified reactants and desired product → candidate solvent, catalyst, additives, temperature and amounts | Condition/yield/selectivity records; context-dependent outcome model and explicit missingness | Held-out condition retrieval, yield/selectivity error and calibration, condition-class OOD tests; regret only where counterfactual condition outcomes or prospective tests are available |
| Transition-state proposal | Reactant/product or an elementary graph event → 3D conformers, reaction coordinate and candidate TS | 3D equivariant geometry/energy-force head, electronic-structure labels and a QM refinement interface | Saddle-point/endpoint-connection success, barrier error and QM calls per valid TS; never score a graph edit as a found TS |
| MS/MS and gas-phase fragmentation | Precursor ion/adduct, polarity, collision energy and instrument domain → charged-fragment/neutral-loss network and spectrum | Dedicated gas-phase action head, charge/radical/H-transfer handling and spectrum-intensity head; reference spectra | Exact formula and `m/z`, fragment recall, spectrum similarity, energy-conditioned response, isomer retrieval, charge/mass accounting |

Condition recommendation is a separate supervised problem with categorical and continuous targets; reaction records often lack standardized quantities and settings ([condition-recommendation study](https://doi.org/10.1039/D5SC04957A)). A low proposed activation barrier is not by itself a yield prediction. The first practical TS use should be *proposal acceleration*: rank event-conditioned 3D starting guesses, then verify selected candidates with QM, rather than announcing a universal transition-state generator ([reactive ML potentials for TS search](https://www.nature.com/articles/s41467-026-72945-0)).

MS is a meaningful transfer test of electron/fragment representations, **not the same dynamics** as solution-phase synthesis. The domain head must model ionization/adducts, collision energy, competing charged fragments, complementary neutrals, radical pathways and hydrogen transfer. Fragment-network and intensity prediction are distinct heads, as in [ICEBERG](https://pmc.ncbi.nlm.nih.gov/articles/PMC12154671/); a correct fragment formula is not automatically a correct intensity or mechanistic explanation. Published spectra provide empirical validation without new wet-lab measurements, subject to instrument/domain stratification.

## RetroChimera teacher: an endpoint preference, not a physics oracle

[RetroChimera](https://www.nature.com/articles/s41586-026-11160-9) predicts and ranks precursor sets from a product. Freeze a versioned checkpoint and cache its product-only Top-K outputs on **training products only**. Match canonical complete precursor multisets; record teacher coverage and provenance. No teacher prediction, GT precursor, or reference suffix enters a student test prompt.

On fresh student rollouts, compare **executor-accepted terminal endpoints** using calibrated rank/support where the teacher actually provides it. A teacher Top-K omission means *unknown*, not chemically impossible. Prefer a verifiable endpoint or independent evidence over the teacher when they disagree. Retain mechanism-supervised anchors and reward non-GT alternatives only when the evidence contract supports them. This is **on-policy endpoint-preference distillation**, not token-level OPD or electron-action distillation: RetroChimera does not expose an expert distribution over student-visited electron states. Its paper and [released interface](https://github.com/microsoft/retrochimera) support endpoint ranking, not mechanism labels.

Because the teacher may have been trained on patent-derived or commercial data, audit exact and near-duplicate train/test overlap before making a clean-transfer claim. Pistachio, USPTO-50K and USPTO-FULL checkpoints are separate teacher conditions, not interchangeable baselines. Report teacher-only performance and student/teacher complementarity before claiming an improvement from distillation.

## Data, scale and falsification gates

1. **Start with the current mechanistic domain.** Use frozen strict-executable trajectories for inverse state/action supervision; keep the official reaction-level FlowER split and the strict-executable subset named separately per `PROJECT_MEMORY.md`. Electron-event decisions from one reaction are correlated, not independent reaction examples. Existing FlowER-derived trajectories do not automatically label condition optima, 3D TS geometries or MS fragmentation.
2. **Pretrain and train with provenance.** Separate domain, phase, source dataset, mechanism provenance (`inferred`, `calculated`, `experimental`), atom mapping, conditions, uncertainty and data license. Use graph/action masking and multi-horizon JEPA targets only where transitions are known. Add independent forward, conditions, QM and spectra datasets per task; do not invent missing labels or merge solution/gas tasks by SMILES alone.
3. **Scale empirically.** Compare approximately 50M, 200M and, only if justified, 500M parameter graph/trajectory models on frozen splits and controlled data/compute budgets. Track unique reactions and families, not just correlated transition counts. Measure throughput and inference-time executor/QM calls. Stop scaling if long-horizon accuracy, OOD reliability or calibration plateaus; parameter count alone is not evidence of chemistry.
4. **Require isolating controls.** Compare graph policy alone, policy+JEPA, policy+teacher, and policy+JEPA+teacher under matched encoder/action/data budgets; include teacher-only endpoint predictions. Test teacher top-k coverage before distillation. Remove action conditioning or shuffle future-state targets to test whether JEPA learns useful dynamics rather than a trivial graph prior.
5. **Assess reliability, not just exact match.** Report formal failure modes, top-k complete endpoints, independent forward and condition metrics, 1/2/4/8-step rollout drift, alternative-product coverage, reaction-class/scaffold/time/source OOD, calibration/reliability plots, risk–coverage under abstention, and paired qualitative chemistry review. A single experimental reference is not the full set of valid precursor routes.
6. **Use a staged go/no-go.** First establish improvement in mechanistic inverse/forward tasks. Add condition prediction only after labeled contexts are identified. Add 3D TS and gas-phase MS as separately gated extensions. A poor cross-domain result is evidence against the proposed shared representation, not a reason to weaken the evaluator.

## Immediate, low-cost decision experiment

Without changing the ICLR main run or launching a large model, train one modest graph encoder/policy with and without a 1/2/4-step action-conditioned JEPA auxiliary objective on the same frozen mechanistic training view. Compare teacher-forced next-event accuracy, closed-loop legality, long-horizon endpoint accuracy, uncertainty calibration and compute. In parallel, freeze RetroChimera outputs on a development-only product set to measure endpoint overlap and potential distillation coverage. Only if these two signals are nontrivial should a 200M-scale student and teacher-guided post-training be proposed.

**Decision rule:** If JEPA improves only latent loss but not executed decisions, it is not a useful world model for this task. If RetroChimera improves only endpoint ranking while valid mechanism paths cannot reach those endpoints, it is not yet a mechanism teacher. If the model cannot separate gas-phase fragmentation from solution reactions under controlled tests, do not claim a unified chemical dynamics model.
