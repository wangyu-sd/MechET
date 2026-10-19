# Source-checked two-reaction Cabozantinib route

This is a checked **case**, not a completed population-level literature score.
The dataset target ID says `Cabozantinib S-malate`; its frozen molecular input
and the generated route reach **Cabozantinib free base**. Salt formation is not
silently added to the model's route.

| Actual generated complete reaction | Primary evidence | Qualification |
|---|---|---|
| Acid + SOCl2 → corresponding acid chloride | [WO2018104954A1](https://patents.google.com/patent/WO2018104954A1/en), preferred embodiment step(a) and Example1 | Performed in situ, then used for coupling; intermediate not separately isolated or characterized |
| Acid chloride + quinolinyl-oxy aniline → free-base diamide | [US20160022662A1](https://patents.google.com/patent/US20160022662A1/en), named preparation | Performed free-base preparation; actor's auxiliary mixture differs |

Both essential net transformations have source-checked exact structural
precedent under LGRV-v1. Frozen-stock AND rechecking finds all three leaves:
the aniline, the monoamide carboxylic acid, and SOCl2. This establishes one
**all-essential-step precedent-covered stock route**, not laboratory validation
of the generated route or physical validation of its symbolic intermediates.
No isolated yield is assigned to the predicted acid chloride.

The acid-chloride preparation's generated program has five FLOW events and
twelve component electron-pair moves. Its ordered v1 whole-program signature
is absent from this actor's actual training exposure, while each primitive
appears in at least4,677 eligible training reactions. The coupling program is
training-seen. Thus the case links an unseen **electronic composition** to an
experimentally documented **net transformation**; it is not novel chemistry
or a strict-H2/reference-C1 generalization claim.

## Traceability

- Frozen report: `outputs/eval/r1_r2_frozen_06b_shared_mcts_20261014_07/tree/rank00/evaluations/8b4ee51d6807d371a02cee3c8680b290a7b3236ab6e7dea93c928cdf2a3280fc.json`;
  SHA256 `0c3efa3a1ca7868127016f727b09ca0aed622479a8526b085374f4c93f6cb525`.
- Native tree SHA256 `c8c4098d123efe6ead4502daf4275bdfeccf1a1e4343d22195a1a93d9d5d107b`.
- Route key `afb473abe8f362dc20e6dffa50647baa80c44b4304e19299a893841138f18b9c`.
- Coupling reaction key `4e8b1ad7a25f05c2ebdc7642de03c9990617588c2546b5b4f6af78e77387a4e0`.
- Acid-chloride reaction key `c08ce91f53318293c4af8fe8fc0f610fcfee43895186af539053ed8fadca581b`.
- Unseen program signature `f5dfdc1888cfce2ff484c99248c66ae999660b7210f9842fa322ce2690d4d3a2`;
  source `/reaction_trees/O=C(Cl)C1(C(=O)Nc2ccc(F)cc2)CC1/contexts/37/actions`.
- Primary-source bytes, receipt, normalized text and passage hashes:
  `configs/analysis/lgrv_primary_source_seed_20261018.json` (unchanged) plus
  `configs/analysis/lgrv_primary_sources_20261018_v2.json` (new source).
- Exposure, frozen pool and program-prefix hashes: [R1 snapshot](R1_PROGRAM_JOIN_SNAPSHOT.json).

The exact two reaction keys do not establish a matched-compute advantage over
Greedy; the archived runs have unequal work. The complete source-index search,
other reactions, and same-rule baseline primary checks are still unfinished.
