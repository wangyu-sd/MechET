# Seven additional source-checked preparations

Three RDKit reaction plates show all seven preparations in the
[new registry](../../../../configs/analysis/lgrv_primary_sources_20261018_v6.json),
including structural selectivity in multifunctional substrates. Each reaction
is an actual frozen model-generated complete transformation. These are not
simulated predictions or comparisons selected to make another model fail.

| Plate | Performed preparations and structural distinction |
|---|---|
|1|US7842688B2 Example159(c): phenolic O-alkylation while retaining a secondary alcohol/N-Boc group; US8492378B2 Example217: oxadiazole S-benzylation; US8729263B2 Preparation1: exocyclic amine substitution while retaining ring NH and the second pyridazine Cl |
|2|US7435815B2 C.1: quinolin-6-yl sulfonamide, not the adjacent isoquinoline entry; US4829069 Example2(g): quinazoline S-benzylation |
|3|US7538121B2 Example114: quinolin-7-yl biarylamide from the named acid and amine; US5059689 Example17: single triazine substitution with a sterically hindered spiroamine |

The [review specification](../../../../configs/analysis/lgrv_checked_preparations_20261018_v6.json)
contains independently transcribed structures, specific preparation locators,
conditions and yield limitations. The binding utility verifies the pinned
source text, original bytes, normalization, unique short witness and exact
chemical identity before producing the registry. It does not decide that an
experiment occurred; that decision comes from reading the preparation.

All nine exports passed vector/183-mm geometry/embedded-font/300-dpi checks.
All three actual PNGs were opened and inspected for structures, reaction
arrows, source labels, conditions and clipping. The full source documents stay
outside Git; only metadata and short witnesses are distributed.

These plates show **forward net-reaction precedents** for transformations
constructed by inverse search. Their straight arrows are not electron-flow
arrows or experimental verification of every symbolic intermediate. Actual
recorded inverse FLOW/fork figures remain in the earlier figure sets. No
whole-route or population chemical-accuracy claim follows from this gallery.
