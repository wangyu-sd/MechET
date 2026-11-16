# Retro-E: Standard PaRoutes/RetroCast multistep TRAIN/VALID and matched strong baselines
Research update for PR #109, 2026-11-16. **Normative for benchmark/data protocol**, superseding any earlier wording that treated the custom 460 as the sole/main standardized benchmark.

## 1. Findings from official sources

- Original PaRoutes (https://github.com/MolecularAI/PaRoutes) supplies full patent-derived route data, two evaluated route sets n1/n5 (10,000 each), corresponding stock inventories and one-step reaction data. It does **not** supply one universally canonical multistep TRAIN/VALID split.
- RetroCast (https://retrocast.ischemist.com/guides/training-sets/) provides versioned PaRoutes-derived multistep route TRAIN/VALID releases. It is a **third-party standardized train/validation release**, not the original PaRoutes authors' official TRAIN split.
- Two source-published route protocols:
  (1) route-holdout-n1-n5: excludes exact n1/n5 test route signatures;
  (2) reaction-holdout-n1-n5: excludes exact test routes and test reaction edges from remaining candidate training routes. The latter is recommended as the primary clean route-RL training version, subject to manifest and source verification.
- RetroCast Reference evaluation: ref-lin-600 + ref-cnv-400 + ref-lng-84 = 1,084; stocks n5-stock except ref-lng uses n1-n5-stock. Market evaluation: mkt-lin-500 + mkt-cnv-160 = 660, stock buyables-stock; Market is a different practical track.
- The project's existing 23,730 route TRAIN / 2,986 VALID are a project-specific curated RetroCast-derived training version, not automatically identical to the full source release. The manuscript's 460 (100 expert, 100 drug/clinical, 160 convergent, 100 PaRoutes-derived) is a curated evaluation collection, NOT a community-standard multistep train/validation/test data release.
- RetroChimera (https://github.com/microsoft/retrochimera) is principally a very strong **single-step** reactant predictor. Official released checkpoints: Pistachio (strongest, proprietary-source training), USPTO-FULL, USPTO-50K. To compare complete routes, attach the model to a search system such as Syntheseus Retro* / MCTS (https://microsoft.github.io/syntheseus/stable/cli/search/). Its Nature (2026) expert assessment of 9/10 fully accepted routes is a 10-target route study, **not** a PaRoutes-n5 or RetroCast Reference 1,084 measured result; do not transfer that result.
- RetroCast (https://retrocast.ischemist.com/) has route output adapters for RetroChimera/Syntheseus and other systems, but an adapter does not imply that matched predictions are already supplied or computed.

Sources:
https://retrocast.ischemist.com/guides/training-sets/
https://retrocast.ischemist.com/dev/rationale/training-set-releases/
https://retrocast.ischemist.com/guides/benchmarks/
https://github.com/MolecularAI/PaRoutes
https://github.com/microsoft/retrochimera
https://microsoft.github.io/syntheseus/stable/cli/search/
https://doi.org/10.1038/s41586-026-11160-9

## 2. Train data versions (same FlowER-trained Stage-II)

Common immutable initialization: project FlowER-trained Qwen3.5-0.8B History-SFT checkpoint-89442, exact actor/tokenizer/Executor/stock-material contract SHA. The original SFT was trained using 243,753 replay-approved FlowER examples; **route-RL reaction-holdout does not undo potential pretraining overlap**. Never label that as chemistry-level clean pretraining.

D1 legacy: existing 23,730/2,986 TRAIN/VALID, kept only as a direct PR107 vs PR109 historical continuation.
D2 route-standard: RetroCast PaRoutes route-holdout-n1-n5 TRAIN/VALID.
D3 reaction-standard **primary**: RetroCast PaRoutes reaction-holdout-n1-n5 TRAIN/VALID.
D4 industrial/Open Pro: source/permit-verified additional multi-source routes, external RetroChimera allowed but entirely separate from controlled Standard.

Pin RetroCast release; preserve original manifest.json and SHA256SUMS. Obtain actual counts from release/parsed rows. Apply the *same* translation from nested source routes to hidden-reference route-RL target instances. Count source routes, target identities, reference edges, stock-eligible cases, failures, AND/convergent structures. Preserve occurrences, graph connectivity and source patent IDs. Routes supply target/step/precursor/stock references, **not** gold native electron-arrow action traces; Student still executes IMPORT/FLOW/FINISH autonomously.

CLI inputs documented upstream:

    retrocast get-training-data reaction-holdout-n1-n5 --split training --dir data/datasets/paroutes
    retrocast get-training-data reaction-holdout-n1-n5 --split validation --dir data/datasets/paroutes
    retrocast get-training-data route-holdout-n1-n5 --split training --dir data/datasets/paroutes
    retrocast get-training-data route-holdout-n1-n5 --split validation --dir data/datasets/paroutes
    retrocast get-data ref-lin-600
    retrocast get-data ref-cnv-400
    retrocast get-data ref-lng-84

Use a pinned --release option when creating actual experiment manifests. Check redistribution/license before publishing derived data.

## 3. Standard target ladder

1. Freeze a 100-target **stratified subset of public Reference**, only for pipeline/cost/error analysis; never tune checkpoints or rewards on it.
2. **RetroCast Reference 1,084**: moderate-scale main public benchmark, report 600/400/84 per cohort, source stocks, stratified route length and topology.
3. **PaRoutes n5 10,000**: full-size primary public benchmark; also n1 10,000 if resources permit. Explicitly evaluate solvability, reference-route similarity and efficiency under source stocks.
4. **Retro-E custom 460**: retain as a separate, supplementary chemistry/stress-test panel; report exact overlap with all standardized source sets rather than treating it as independent.
5. **RetroCast Market 660**: operational/commercial inventory track, appropriate for separate Retro-E-Pro/open-enhanced comparison.

Important: Reference and Market are derived from PaRoutes; their results are correlated, not independently additive. Do not aggregate arbitrary total solved counts across different stocks.

## 4. Strong baseline matrix

Controlled claim about the *algorithm* (same Student checkpoint and training split):
- Frozen Stage-II SFT with independent sampling vs ReFIT search.
- Original action-level Process GRPO vs teacher-free failure-aware Process GRPO vs ReFIT-PO.
- Same ReFIT-PO checkpoint with no/low-budget/large-budget ReFIT.
- Mandatory final TRAIN-only search-to-policy distilled Standard student vs its original RL parent under matched evaluation.

External strong *whole-system* comparators (different data/backbone disclosed, same TEST IDs/stocks/budget when possible):
- RetroChimera Pistachio + Syntheseus Retro* and/or MCTS: top-priority modern strong comparator. Separately consider RetroChimera USPTO-FULL checkpoint for a source-constrained comparison. Proprietary Pistachio training is not data-matched with public FlowER initialization.
- AiZynthFinder MCTS and Retro* with pinned expansion model/checkpoint: canonical PaRoutes comparators.
- DirectMultiStep Explorer / Explorer XL: strong direct-route generation comparator if frozen predictions/checkpoints are reproducible.
- Optional SynPlanner or ASKCOS, only when reproducible outputs and comparable search contracts are available.

Syntheseus CLI: search_targets_file, inventory_smiles_file, model_class, model_dir, search_algorithm=retro_star/mcts, explicit time/expansion controls. See official https://microsoft.github.io/syntheseus/stable/cli/search/ . Record all original and normalized outputs. RetroCast adapts/routes/scores but does not automatically run each external model.

Primary common metric: source-stock connected nonzero-step Solved@Budget; separate reference-supported exact-route matching/route TED Top-1/5/10 and route diversity. Report true model requests, generated tokens where observable, Executor calls, model wall-time/hardware, time to first solved route. Do not pretend one model call has identical cost in a token-by-token chemical Agent versus a native one-step predictor.

Do not use RetroChimera as reward/teacher for the *controlled* Retro-E Standard RL, ReFIT-PO or final Standard distillation. RetroChimera is an **external baseline** and optionally a component in separately named Retro-E-Pro. +F may use original forward FlowER (only after valid FINISH), not RetroChimera.

## 5. Required Codex deliverables

- Download source-pinned full TRAIN and VALID files for D2 and D3, manifest and stock; checksum and report counts/identity.
- Convert the actual nested route data to the existing Route Environment/HiddenRouteReward interface without leaking reference graph into Student observation.
- Audit both RL dataset-test identities AND FlowER SFT pretraining-test reaction overlap; distinguish exact target, reaction, route, patent-source and scaffold exposure.
- Run a single correctly frozen matched-target baseline suite including at least RetroChimera+Syntheseus and AiZynthFinder MCTS/Retro* (if accessible); report missing model/results rather than inventing a score.
- Use D3 reaction-holdout as standardized main controlled RL experiment; D2 route-holdout as data-ablation; D1 legacy as continuation.
- Keep the final **search-to-policy distillation mandatory**, from training-only ReFIT exploration. Compare distilled actor under no-search and equal-search cost.
- Deliver measured Standard / +F / Open-Pro tables separately, plus source lineage and model registry; no old single-step Top-k or 10-molecule RetroChimera expert acceptance values inserted into route benchmark comparison cells.
