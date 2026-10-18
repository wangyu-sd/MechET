# Product identity does not establish the indexed preparation

Five original capped source/query comparisons remain unresolved for explicit
source-side reasons. They are not five failed chemical reactions.

| Original primary | Queries | What was actually found | What remains unresolved |
|---|---:|---|---|
| US7465795B2,Table2/MethodE |3|Matching named cinnoline product in a conditional compound table;MethodE uses an organotin partner|The indexed cinnolineBr/pyrazolylBpin preparation is not established by this Stille-type method |
| US5068340,Examples3/2/1(a)|2|Actual pyridinium-phosphate inner salt,0.305g isolated;acidic precursor workup|Literal indexed phosphate-anion precursor formulation,and transfer to the queries' competing amino sites |

The first patent's Table2 says the compounds *can also be made*. Its method
uses8-trimethylstannylcinnoline,aryl/heteroarylBr,Pd(PPh3)4 anddryDMF at150C
for4–16h. An actual MethodE Example23 produces a different fluoropyridyl
compound;its45mg/37.7% does not establish the candidate dimethylpyrazolyl
preparation. No claim is made that the alternative Suzuki combination is
chemically impossible.

The second patent does report an actual quaternization. Example3 uses crude
bromoethylphosphate in pyridine at60C11h andisolates the inner salt. Its
referenced phosphorylation workup acidifies topH1 before extraction;the
index instead explicitly encodes an anionic precursor without counterion.
Deprotonation in pyridine is possible,but is not an exact isolated-precursor
identity statement. The queries additionally introduce methylaminopyridine,
bromoethylamine oramide contexts whose N-selectivity is not measured by this
source. Neither counterions nor protons are silently added orremoved.

## Bound evidence and reproducibility

- [Cinnoline specification](../../../../configs/analysis/lgrv_source_preparation_limits_20261019_v1.json)
  and [bound audit](source_limits_review.json).
- [Phosphate specification](../../../../configs/analysis/lgrv_source_preparation_limits_20261019_v2.json)
  and [bound audit](phosphate_source_limits_review.json).
- [Binder](bind_limits.py):rehashes all sealed analysis outputs andoriginal
  source traceback;checks physical raw/normalized text,unique short witnesses,
  independently transcribed indexed-candidate graphs,original capped source
  membership andexistingC status. It does not decide chemistry from the text.

Ten CPU software tests first fail before the binder exists,then pass. They
cover corrupt analysis/source bytes,missing witnesses,wrong candidate/primary,
duplicate pairs,wrong population andattempted positive evidence orprior-support
downgrade. These are not model experiments. All returned findings retainC and
`generated_reaction_proven_wrong=false`;no positive registry is emitted.

```bash
PYTHONPATH=src python docs/results/lgrv_20261018/source_preparation_limits_20261019/bind_limits.py \
  --review-spec configs/analysis/lgrv_source_preparation_limits_20261019_v1.json \
  --analysis-root /path/to/source_checked_20261019_v15 \
  --traceback docs/results/lgrv_20261018/coverage_traceback_20261018/candidate_patent_traceback.jsonl.gz \
  --cache-root /path/to/primary_documents_bulk_20261018_v2 \
  --output /new/path/source_limits_review.json
```

Repeat with thev2 specification for the phosphate finding. Source judgement
is an explicitly authored input,not independent policy calibration. No model
inference,retrieval-policy change or full patent text redistribution occurred.
