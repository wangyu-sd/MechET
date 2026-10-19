# Query-scoped primary-source review, 19 October

This increment contains two exact preparations and one explicitly scoped
experimental analogue. The original query pool, radius1/full-stereo retrieval,
cap5 and all six methods are unchanged. These are source reviews, not new
model predictions or a new retrieval calibration.

| Primary preparation | Reviewed transformation | Evidence scope |
|---|---|---|
| US7842688B2, Example101b; General MethodD | Piperidinol O-alkylation with chloromethylquinolone | Exact named substrates/product; no source-assigned piperidine stereo or percentage yield |
| US6969713B2, Example92ii; Example80iii | cis-Dimethylpiperazine N-alkylation by chloroacetamide | Exact structures including cis relative stereochemistry;0.25g isolated, no percentage yield |
| US7572820B2, Example321A; Example310D | 2-Bromothiazole/arylboronic-acid Suzuki coupling | Strong experimental analogue **only for query857b0658…**; not the exact predicted substrate pair |

The Suzuki source has thiazole5-CO2Me and3,5-bisCF3phenyl, while the query
has thiazole4-Me/5-CH2O(N-Boc-piperidinyl) and para-chlorophenyl. The record
states the chemical compatibility inference and its limits: retention of
ether/carbamate/aryl chloride is plausible under the reported coupling class,
but query-specific conversion, yield and selectivity are unmeasured. Its local
edit alone would not qualify. The independently transcribed named source
structures reconcile to an **original frozen capped candidate**, not a new
candidate found after seeing outcomes. No patent-family independence or
physical electron-arrow evidence is claimed.

## Verifier correction

`assess()` now requires `analogue_query_reaction_key` to equal the current
complete query identity before its substrate/functional-group/stereo review
can confer TierB. This field already routes reviews in the analysis CLI;
the checker previously did not enforce it itself. Regression tests reproduce
unbound notes, another query's notes, and remote F/Cl substitutions sharing
one local family. They fail before the fix and pass after it. Exact TierA,
the full-stereo guard and direct-conflict TierD behavior are unchanged.
Published v11 had no TierB records, so this does not show that its44 exact
supports were inflated.

Full CPU software verification: **1,256 passed,14 skipped**,184.90s; one
pre-existing PEFT configuration warning. This is not chemical performance.

## Reproduction and source payloads

Specification:
[`lgrv_checked_preparations_20261019_v6.json`](../../../../configs/analysis/lgrv_checked_preparations_20261019_v6.json).
Bound registry:
[`lgrv_primary_sources_20261019_v6.json`](../../../../configs/analysis/lgrv_primary_sources_20261019_v6.json).
Each record includes the actual cached raw/text hashes, unique short witness,
locator, source structures, method references and limitations. Full patent
payloads remain outside Git. Earlier registries and population versions are
not overwritten.

[`verify_preparations.py`](verify_preparations.py) rehashes raw sources,
reconstructs normalized text, rebinds each reviewed specification, checks
named-source chemistry, original candidate membership/full-stereo local
compatibility and the query-binding guard. Its
[`review_validation.json`](review_validation.json) is a three-source validation
receipt, **not** a replacement for the full-population analysis.

```bash
PYTHONPATH=src python docs/results/lgrv_20261018/primary_review_v12_20261019/verify_preparations.py \
  --analysis-root /path/to/source_checked_20261019_v11 \
  --cache-root /path/to/primary_documents_bulk_20261018_v2 \
  --review-spec configs/analysis/lgrv_checked_preparations_20261019_v6.json \
  --registry configs/analysis/lgrv_primary_sources_20261019_v6.json \
  --output /new/path/review_validation.json
```
