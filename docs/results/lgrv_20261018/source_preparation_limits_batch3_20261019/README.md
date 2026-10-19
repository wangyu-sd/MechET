# Two source-side precursor limits

Three original source/query pairs cover the last two cached-query checks.
They retain C: no positive support or chemical-failure label is emitted.

| Original indexed candidate | Actual original passage | Consequence |
|---|---|---|
| C3-ethylpiperidine carboxylate + EtI → ethyl ester | US8188280B2 and US8367698B2,Production12(1),referencing11(1): an **already ethyl ester** undergoes LDA/enolate **C3-ethylation** | The product is prepared,but the cited operation does not establish indexed carboxylate **O-ethylation** |
| Sodium bis-carbamoylphenolate + trifluoronitrobenzene | US7608731B2,Example1/intermediate21: neutral phenol + **KOtBu**,488g,88%; general prose separately mentions sodium phenolate | Actual ether preparation is established,but the literal indexed sodium precursor formulation is not a performed example here |

The two publications of the first preparation are not two independent
experiments. Their recorded product is not denied; its ethyl ester already
exists in the starting material, while EtI creates a different C-C bond.
Neither document is a source experiment for the generated heteroaryl
carboxylate O-ethylation query.

For21, the source's sodium-base examples use other phenols, not the reviewed
bis-amide. The query additionally lacks the source's nitro SNAr activator and
retains benzylammonium protonation. That substantive applicability gap remains
regardless of the source Na/K formulation issue. No potassium-to-sodium swap,
nitro-group insertion, protonation repair or generated failure is claimed.
Standard nitro resonance notation canonicalizes to the same charged nitro
graph under the repository's existing identity function; no policy changed.

[Specification](../../../../configs/analysis/lgrv_source_preparation_limits_20261019_v4.json)
and [authenticated findings](source_limits_review.json) are reproduced with
the unchanged [versioned limit reader](../source_preparation_limits_batch2_20261019/bind_limits.py),
sealed v16 analysis, original capped traceback and cached primary bytes.
No model/GPU job ran; no old result, source recipe or figure was overwritten.
