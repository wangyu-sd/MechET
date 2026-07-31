# Proof-carrying retrosynthesis (`MECH_PROOF v1`)

`MECH_PROOF v1` is an experimental action-only path that makes the proof the computation used to obtain the precursor.

## Causal output path

```text
mapped product
  -> ROOT imports + sparse EDGE actions
  -> deterministic RDKit executor
  -> reconstructed precursor
```

The assistant output contains no `STATE` lines and no `<answer>` block. Dataset endpoints remain in metadata for reward and evaluation only.

## Program form

```text
<proof>
MECH_PROOF v1
TARGET_SMILES "<mapped product>"
ROOT s0
  IMPORT "<mapped species>"
PRECURSOR_STATE sk
EDGE s0 s1
  BOND i j +1
  LP i -2
  CHARGE i -1 0
...</proof>
```

The executor:

1. constructs each root from `TARGET_SMILES` and declared imports;
2. applies edges in dependency order;
3. checks bond, lone-pair and charge deltas against the executed transition;
4. enforces exact electron conservation;
5. requires equal reconstructed states at DAG joins;
6. returns the reconstructed precursor state.

## Compile cold-start data

```bash
python scripts/build_mechet_proof_sft.py \
  --input-dir data/mechet_sft \
  --output-dir data/mechet_proof_sft \
  --splits train valid test
```

The compiler reads existing `MECH_ET v3` gold trajectories, removes model-authored state strings and the answer channel, and validates that execution reconstructs the original precursor before accepting a sample. The manifest reports accepted and skipped trajectories.

## SFT

```bash
export QWEN_MODEL_PATH=/path/to/Qwen3-8B
python scripts/train_mechet_sft.py --config configs/proof_sft_pilot.yaml
```

The generic assistant-only trainer is reused because proof rows use the same chat-message JSONL envelope.

## Proof RLVR beyond the teacher trace

```bash
python scripts/train_mechet_rlvr.py --config configs/proof_rlvr_pilot.yaml --dry-run
python scripts/train_mechet_rlvr.py --config configs/proof_rlvr_pilot.yaml
```

`train_mechet_rlvr.py` dispatches rewards from `task_type`:

- `mech_et_cot_retro`: strict state-annotated local verification;
- `mech_proof_retro`: deterministic proof execution and endpoint reward.

For proof rows, no intermediate teacher state or exact teacher mechanism is compared during RLVR. An executable proof receives a process reward, and a proof whose executor-derived precursor matches the dataset endpoint receives an additional outcome reward. This allows exploration of valid proofs that differ from the cold-start trajectory.

## Inference and evaluation

```bash
python scripts/infer_mechet_proof.py \
  --data data/mechet_proof_sft/valid.jsonl \
  --adapter outputs/mechet_proof_rlvr_pilot/adapter \
  --out outputs/mechet_proof_eval/generations.jsonl

python scripts/eval_mechet_proof_generations.py \
  --data data/mechet_proof_sft/valid.jsonl \
  --predictions outputs/mechet_proof_eval/generations.jsonl \
  --out outputs/mechet_proof_eval/summary.json
```

Primary metrics are:

- `format_ok_rate`;
- `execute_ok_rate`;
- `endpoint_exact_rate`;
- topology-stratified versions of the same metrics.

## Current scope

This first implementation establishes the causal proof-to-precursor path and supports chain, tree and DAG dependencies. Chemical plausibility beyond formal execution—such as energetic feasibility, reagent compatibility and precedent retrieval—remains a separate ranking layer rather than part of the formal verifier.
