# Actual epoch-8 paired smoke: only the main product

User explicitly requested the preceding smoke, then asked to isolate the main
product before submitting Stage-II training. All 32 paired validation reactions
completed. Evaluation and independent audit both exited 0. No new training or
platform job was submitted.

## Result

Same Qwen3-0.6B epoch-8 `checkpoint-4056`, same 32 reaction IDs, tool grammar,
state-only observations, FP16/no quantization, greedy K1, executor, reference
full precursor mixtures, and generation budgets. Only the product input changes.

| Condition | Main-product-derived precursor exact match | Full precursor-mixture exact match | Accepted FINISH |
|---|---:|---:|---:|
| Original two-component product side | 28/32 = 87.50% | 28/32 = 87.50% | 29/32 = 90.625% |
| Main product component only | 0/32 = 0% | 0/32 = 0% | 12/32 = 37.50% |

The 87.5% result is **not** strict main-product-only accuracy. This paired smoke
is not a full-3,120-test result and does not establish chemical impossibility of
every nonmatching endpoint.

## Input definition

For each original trace-view product side, select the largest carbon-containing
connected component by heavy-atom count. There are no size ties in these 32
cases. Remove original atom-map labels and independently map that one component.
Preserve its recorded charge, stereochemistry and explicit-hydrogen semantics;
do not neutralize it or substitute the unrelated `rxn_prod_min` proxy.

Remove the other component from BOTH the target text and the private initial
state. Never inject it through imports, a hidden root state, GT history, or an
environment default. Subsequent imports are exclusively model predictions.
`selected_inputs.jsonl` records each actual input and its removed component.

Example `10643`:

```text
Previous: COC(=O)c1cccc(CBr)c1.[Cl-]
This run: COC(=O)c1cccc(CBr)c1
```

All 32 roots are verified single-component roots. No training data, checkpoint,
reference endpoint or production executor was modified by this input ablation.

## Concrete failure evidence

All **32/32 reference first electron events touch atoms in the removed
component**. To follow those recorded trajectories from the new root, the missing
participant must first be inferred and imported.

Yet the model's first decision is **FLOW in 32/32 cases, IMPORT in 0/32**.
Eight first decisions are rejected immediately. Over the whole rollout:

- 10 reactions stop on a source/destination bond that is absent;
- 5 stop on `CHEMICAL_STATE_INVALID`;
- 5 stop on `STATE_CYCLE`;
- 12 reach FINISH but miss the reference endpoint.

The 18 model IMPORT decisions occur only after one, two, or three earlier
decisions; none is a root IMPORT. These observations locate a root-distribution
problem: the trained trace-view roots already include the partner, whereas the
new main-only roots require a preceding partner-generation decision. They do
not establish that additional epochs or compressed history alone will fix it.

This is not token-budget exhaustion: max completion 171 tokens, max prompt
1,063 tokens, max 6 attempted decisions. Bounds remain 512 completion tokens,
4,096 context tokens, 40 decisions and 32 imported fragments. Total generation
11,219 tokens, 103 calls, 83 accepted actions; runtime 691.71 seconds excluding
model loading on local Tesla T4.

## Endpoint scoring audit

Complete-mixture scoring uses the unchanged original reference, with atom-map
removal, canonical SMILES and component-order-insensitive comparison. No new
neutralization or relaxed success rule was introduced. It independently
recomputes to 28/32 versus 0/32.

For structural scoring, BOTH conditions were additionally rescored against the
same reference fragments containing atoms of the **main product**. This avoids
comparing an all-product-side reference (which may include BBr3 or water) with a
main-product-only predicted subset. Eleven original structural references change
under this consistent definition; final paired hit counts remain 28 versus 0.
The authoritative structural comparison is `endpoint_comparison.json`.
The raw driver's legacy `top1_structural_exact` uses the original all-product-side
reference; do not cite that field as a general main-derived precursor metric.
Its complete-mixture score is unaffected.

All 103 actual prompt token sequences were reconstructed from model-executed
states and verified against the captured inputs. Generated tool calls,
accepted/rejected execution, final states, artifact hashes and model SHA were
independently checked. No model sampling occurred during this audit.

## Artifacts

- `evaluation.json`: completed inference summary and immutable runtime contract.
- `endpoint_comparison.json`: verified, consistently defined paired scoring.
- `per_reaction_endpoint_comparison.jsonl`: actual main inputs, normalized
  predicted/reference core precursors, hits, terminal status and errors.
- `records/raw_generations.jsonl`: every actual prompt, input/output token IDs,
  attention mask and generated text.
- `records/episodes.jsonl`: every attempted action, accepted states, errors and
  endpoints.
- `selected_inputs.jsonl`: original product side, selected main product, removed
  component, reaction ID.
- `run_main_product_only.py`: experiment driver using the unchanged runtime.
- `audit_and_rescore.py`: independent replay and consistent paired scoring.

Adapter SHA-256:
`bf5cacc93d2c5e64b11df585e0dd0febf790773e21d13ec0838fa8dd538cdf51`.
Path: `outputs/agent/reliable_mech_uspto31k_state_qwen3_0_6b_v100_seed17/checkpoint-4056`.
This is the actual 8-epoch **state-only Stage-I** checkpoint, not the root
validation-best epoch-7 adapter or the FlowER compressed-history Stage-II model.

Stage-II submission remains on hold under the user's “等等” instruction.
