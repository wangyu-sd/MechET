# CRN-JEPA: an executable 2D chemical reaction world

**Status:** spin-off research proposal. This document is not an implemented method, completed experiment, training authorization, or change to the current MechET/ICLR protocol. It records a research direction that originated from MechET's executor and graph-policy work but is now scientifically broader than retrosynthesis and should migrate to an independent repository once the Phase-0 design is frozen.

## Core thesis

The project does **not** train a separate model for each chemical-reaction task. It constructs an executable two-dimensional chemical reaction world and learns its latent dynamics with JEPA.

The chemical world is a **Chemical Reaction Network (CRN)** in the chemistry sense, not a neural network. Molecular species are nodes. Elementary electron-transfer events are directed hyperedges that may consume and produce multiple species. A reaction state identifies which species are currently present together with the relevant domain context. JEPA learns which regions of the CRN remain reachable after candidate chemical events.

The resulting scientific question is:

> Can a model learn the future reachable structure of an executable chemical reaction network well enough that forward reaction simulation, mechanism exploration, retrosynthesis, catalytic-cycle analysis and gas-phase fragmentation become different forms of navigation over the same learned chemical world?

## 1. Chemical world definition

### 1.1 Species

A species is a chemically explicit 2D molecular graph

\[
m_i=(G_i,q_i,r_i,\chi_i,\iota_i,\ldots),
\]

including atom identity, bond order, formal charge, radical/spin bookkeeping where available, stereochemical state and isotope information when relevant.

### 1.2 Elementary events

A CRN edge is not an overall reaction edit. It is an **elementary mechanistic event** with explicit electronic bookkeeping:

\[
e:\{m_1,\ldots,m_k\}\rightarrow\{m'_1,\ldots,m'_l\}.
\]

Because a chemical step can involve multiple reactants and multiple products, the appropriate representation is a directed hyperedge. An event records the affected atoms/bonds/electron containers, charge/radical changes, imported species and provenance.

A formally replayable graph edit is not automatically a physically established mechanism. Every edge therefore carries a provenance/evidence tag.

### 1.3 Reaction state

The CRN is the world map. A **ReactionState** is the current location in that world:

\[
X_t=(\{m_i\}_t,c_t),
\]

where \(c_t\) may include phase, solvent, catalyst, temperature, ion/adduct, collision energy or other domain context when known.

A transition is

\[
X_t\xrightarrow{e_t}X_{t+1}.
\]

The executor is authoritative for committed states:

\[
X_{t+1}=\operatorname{Executor}(X_t,e_t).
\]

There is only one CRN. ReactionState is not a second CRN; it is the currently active multiset of species and context inside the same chemical world.

## 2. What JEPA must learn

The JEPA is not introduced to approximate the already deterministic one-step executor. Its purpose is to model **future chemical reachability**.

Let

\[
z_t=E(X_t,c_t).
\]

For an action sequence \(e_{t:t+h-1}\), the predictor estimates the latent future:

\[
\hat z_{t+h}=F_h(z_t,e_{t:t+h-1},c_t),
\]

and is trained against a stop-gradient/EMA encoding of the actual future state reached by executable rollout.

The useful object is not one-step reconstruction but the future reachable region

\[
\mathcal R_h(X_t,e_t),
\]

including whether a branch remains productive, reaches a target, enters a dead end, produces a competing product, closes a catalytic cycle or explains observed fragments.

### 2.1 Counterfactual learning

At the same state, generate multiple executor-legal events

\[
\{e_t^{(1)},e_t^{(2)},\ldots,e_t^{(K)}\}.
\]

Execute each event and short rollout to create local CRN branches. The model must distinguish futures caused by different events rather than merely imitate the reference trajectory.

This leads to the central capability:

> Given several chemically legal electron-transfer decisions, predict which future region of the reaction network each decision opens or closes.

### 2.2 Multi-scale latent targets

Graph-level targets alone can ignore the local bond-order, charge or radical change defining an elementary event. The JEPA therefore uses:

- atom-local latent targets;
- bond/electron-container-local latent targets;
- graph/state-level targets;
- multi-horizon targets, initially \(h=1,2,4\), later testing \(h=8\).

Anti-collapse regularization and action-conditioned controls are required. A shuffled-action or shuffled-future target should fail if the model is genuinely learning dynamics.

## 3. One world, multiple queries

The same CRN supports several tasks without redefining the underlying world.

| Query | World-model interpretation |
|---|---|
| Forward reaction simulation | Given \(X_0\), expand the reachable forward CRN and rank product branches |
| 2D mechanism exploration | Given endpoints or a current state, search elementary electron-event paths |
| Retrosynthesis | Navigate backward/goal-condition the CRN toward available precursor states |
| Catalytic-cycle analysis | Search cycles that regenerate catalyst while converting substrate to product |
| Gas-phase fragmentation | Expand a gas-phase ion/fragment CRN under ionization and collision context |

The shared object is the chemical state/electron-event representation. Dynamics may be domain-conditioned. In particular,

\[
F_{\mathrm{solution}}\neq F_{\mathrm{gas}}
\]

is allowed and expected. Sharing a representation does not imply identical solution-phase and collision-induced dynamics.

### 3.1 Forward simulation

Forward prediction becomes local reaction-network expansion rather than direct reactant-to-product translation. Evaluation should include product recovery, branch coverage, conservation, pathway recovery and search cost.

### 3.2 2D mechanistic transition search

The core model does **not** claim to locate a conventional 3D quantum-chemical transition state. Its task is to identify elementary mechanistic topology: which electron containers change, which bonds form or break, how charges/radicals move, which intermediate state follows and which competing event sequences exist.

A later external QM module may verify selected paths or barriers, but that is evidence attached to a graph pathway, not the definition of the CRN-JEPA task.

### 3.3 Catalytic cycles

Catalysis is naturally expressed as a cycle in the CRN. A valid candidate cycle must regenerate the catalyst state while converting substrate to product. With topology alone the claim is "possible catalytic-cycle topology"; dominant cycle, turnover frequency or rate-limiting claims require kinetic/barrier evidence.

### 3.4 Retrosynthesis

Retrosynthesis becomes goal-conditioned CRN navigation. Endpoint teachers may suggest desirable precursor states, but the CRN model must still find an executable elementary-event path to those endpoints.

### 3.5 Gas-phase fragmentation

MS/MS is a transfer domain over the same species/electron bookkeeping but a different physical regime. The gas-phase head must account for precursor ion/adduct, charge localization, radicals, hydrogen transfer, collision energy, complementary neutral losses and spectrum intensity. A fragment formula or mass match is not by itself a mechanistic proof.

## 4. Supervision hierarchy

Only data that actually support an elementary mechanism may supervise CRN edges. Overall reaction edits are not treated as mechanism labels.

### 4.1 Strong elementary-event supervision

Candidate sources include:

- **FlowER-derived mechanistic trajectories** for large-scale electron redistribution paths;
- **PMechDB** for curated polar elementary steps and arrow-pushing supervision;
- **RMechDB** for radical elementary steps;
- carefully filtered **proton-transfer elementary-step corpora**;
- selected **RMG** reaction-family/elementary-step data where the semantics and domain are compatible;
- future curated textbook/literature mechanisms with explicit provenance.

Each source must be normalized into the same executable schema and audited independently before mixing.

### 4.2 Endpoint grounding, not mechanism supervision

Large reaction corpora such as USPTO and ORD provide experimentally reported reaction endpoints. They may supervise

\[
X_{\mathrm{start}}\rightsquigarrow X_{\mathrm{end}},
\]

endpoint reachability, terminal constraints and task heads, but they do not define the hidden elementary path.

**ReactSeq-style overall molecular edit sequences are deliberately excluded from CRN-edge supervision.** They may describe net transformations but are not sufficiently reliable evidence of an elementary mechanism for this project.

### 4.3 Endpoint teacher

RetroChimera is treated as an **endpoint-preference teacher only**. It may provide ranked precursor sets for product-only retrosynthesis. It does not provide electron-event labels or physical mechanisms.

The student may ask whether a teacher endpoint is reachable through an executable CRN path. Teacher omission is unknown, not impossible. Teacher-only, student-only and combined results must be reported separately, with source-overlap/leakage audits.

### 4.4 Physical and empirical evidence

Additional datasets constrain branches without pretending to supply elementary mechanism labels:

- QM reaction paths/barriers and kinetic data for selected transition/path plausibility;
- condition, yield and selectivity records for context-conditioned branch utility;
- reference mass spectra for gas-phase fragmentation outcomes;
- curated catalytic-mechanism literature for cycle/path evidence.

These labels should attach to events, paths or terminal outcomes with explicit provenance.

## 5. Unified evidence schema

Every ingested item should distinguish **what was observed** from **what was inferred**.

Suggested provenance classes:

\`\`\`text
mechanism_edge:
  expert_curated
  literature_proposed
  template_imputed
  qm_supported
  executor_counterfactual

endpoint:
  experimental_endpoint
  literature_endpoint
  teacher_endpoint

physical_label:
  experimental
  calculated
  kinetic_fit
  spectral_observation
\`\`\`

No pseudo-mechanism is silently promoted to experimental mechanism ground truth.

## 6. Training program

### Stage A — elementary chemical world pretraining

Train the shared 2D encoder and event representation on mechanistically supported elementary transitions. Objectives include event prediction, state-transition consistency, atom/bond-local targets and source/domain discrimination.

### Stage B — CRN-JEPA world learning

Construct local branching CRNs from observed trajectories plus executor-legal counterfactual actions. Learn action-conditioned future representations at multiple horizons and future reachability/risk.

Matched controls:

1. graph policy only;
2. graph policy + one-step future prediction;
3. graph policy + multi-horizon JEPA;
4. graph policy + counterfactual multi-horizon JEPA.

### Stage C — endpoint-scale grounding

Add experimentally reported reaction endpoints and optional RetroChimera endpoint preference without inventing hidden mechanisms. Use endpoints as terminal/reachability supervision.

### Stage D — domain-specific grounding

Add context/outcome heads only where labels exist: conditions/yield/selectivity, catalytic cycles, gas-phase fragmentation and optional energetic evidence.

Large-scale parameter growth is conditional on Stage-B evidence. A larger model is not justified if multi-horizon reachability and closed-loop decisions do not improve.

## 7. Phase-0 seed from MechET

This spin-off may reuse MechET assets only as an initial controlled seed experiment.

For any MechET-derived Phase-0 comparison, the authority is:

- \`docs/PAPER_EXPERIMENT_PROTOCOL.md\`;
- \`configs/datasets/flower_artifacts.json\`;
- artifact \`flower_action_delta_v1\` at \`data/flower_inverse_tool_sft_action_delta_v1\`.

The frozen strict-executable reaction universe is:

| split | reactions |
|---|---:|
| train | 257,167 |
| valid | 2,890 |
| test | 28,967 |

This is explicitly distinct from the unqualified FlowER endpoint split of 257,171 / 2,890 / 28,971. The four excluded train and four excluded test rows are upstream-corrupt for the strict executable view.

The Phase-0 JEPA comparison must use the exact same reaction IDs, event compiler, executor contract, graph policy capacity and training budget across JEPA/no-JEPA conditions. It is a feasibility experiment, not the final dataset definition of the independent project.

## 8. Result design

The paper should not be organized as unrelated leaderboards. Results should establish a capability ladder.

### R1. An executable chemical reaction world can be assembled

Report:

- unique species and elementary transitions;
- trajectory/path counts;
- branching-factor and path-length distributions;
- connected components and cycles;
- polar/radical/proton-transfer coverage;
- cross-source executor replay/consistency;
- provenance composition.

### R2. CRN-JEPA learns multi-horizon future reachability

Evaluate \(h=1,2,4,8\) with:

- future-state retrieval Recall@K / MRR;
- reachable-endpoint AUROC/AUPRC;
- calibration/Brier/ECE;
- horizon degradation.

### R3. Future prediction improves counterfactual chemical decisions

At a state with multiple executor-legal actions, rank branches by future productivity/reachability.

Headline metrics:

- counterfactual pairwise ranking accuracy / NDCG;
- closed-loop endpoint success;
- search nodes or executor calls per solved reaction;
- success-vs-search-budget curves.

A JEPA that improves latent loss but not executed decisions fails this gate.

### R4. Reaction-level OOD becomes elementary-mechanism composition

Construct splits where complete reaction classes or scaffolds are unseen but elementary motifs/events overlap with training. Measure whether the learned world supports compositional generalization rather than reaction-template memorization.

### R5. One chemical world supports multiple queries

Use one pretrained CRN-JEPA backbone for at least:

1. forward simulation;
2. 2D mechanism search;
3. retrosynthesis.

Task heads/planners may differ, but the underlying world representation is shared. Compare frozen-backbone, lightweight-adapter and from-scratch controls.

### R6. Transfer beyond the initial solution-phase synthesis domain

Treat catalytic-cycle analysis and gas-phase fragmentation as stringent transfer studies, not automatic inherited capabilities. Report transfer against domain-specific from-scratch baselines and maintain separate dynamics heads where required.

## 9. Three primary go/no-go numbers

Before broadening the project, require convincing gains in:

\[
\boxed{\text{Future Reachability@4}}
\]

\[
\boxed{\text{Counterfactual Action Ranking}}
\]

\[
\boxed{\text{Closed-loop Endpoint Success under matched search budget}}
\]

If these do not improve over the graph-policy baseline, the JEPA does not yet justify a chemical world-model claim.

## 10. Repository boundary

This direction should ultimately live outside the MechET repository.

MechET is a retrosynthetic executable electron-flow project with an ICLR-specific protocol, paper contracts and historical artifacts. CRN-JEPA instead targets a task-general 2D chemical reaction world spanning forward dynamics, mechanism exploration, retrosynthesis, catalytic cycles and gas-phase fragmentation. Keeping both in one repository would create misleading authority, dataset and evaluation coupling.

Recommended transition:

1. keep PR #68 as the historical spin-off proposal and Phase-0 bridge;
2. freeze the CRN schema, evidence schema and first Phase-0 benchmark here;
3. create a new independent repository with its own datasets, provenance registry, model code and evaluation contracts;
4. import only reusable MechET executor/graph components with explicit lineage rather than inheriting the full MechET protocol.

Working project name: **CRN-JEPA**. A final public repository name should be chosen before implementation begins.

## Decision rule

This project succeeds only if the learned representation captures actionable future structure of the chemical reaction network.

- If JEPA predicts latent futures but does not improve counterfactual decisions, it is not yet a useful world model.
- If an endpoint teacher improves ranking but the CRN cannot reach those endpoints through executable paths, it remains an endpoint teacher rather than a mechanism teacher.
- If a shared representation fails controlled transfer between domains, retain domain-specific dynamics instead of weakening the evaluation.
- If a task lacks mechanistic supervision, use endpoint/physical supervision honestly rather than fabricating mechanism labels.
