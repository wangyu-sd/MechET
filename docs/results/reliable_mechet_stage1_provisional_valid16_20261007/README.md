# Stage-I small-test artifacts and training snapshot (2026-10-07)

This directory preserves the **complete raw outputs** of the matched 16-reaction
validation smoke at the unfinished State-SFT `checkpoint-14000`. It is a
diagnostic, not an evaluation of the final Stage-I adapter or the test split.
Both runs used the same 16 reaction IDs, adapter weights SHA-256
`5ad400212499ac32df48f9f06e432c439df15bdc24643cf8288ab7a492474507`,
seed 17, and T4/FP16 singleton decoding.

| View | Complete raw files | Recorded result |
| --- | --- | --- |
| Reference-state, one decision at a time | `gold_state_bs1/selection.json`, `evaluation.json`, `decisions.shard-00-of-01.jsonl` | 123/123 decisions present; 61/63 event-tool choices, 58/63 formally executed events, 43/63 exact events, 9/44 exact imports, 15/16 exact finishes. |
| Independent product-only, K=1 | `product_only_k1/independent_episodes_audit.json`, `independent_episodes_cases.jsonl`, `episodes.shard-00-of-01.jsonl` | 16/16 episodes present; 3/16 process-reliable structural endpoint hits, 12/16 terminal, 9/16 terminal/reference mismatches, 4/16 nonterminal. |

The two JSONL shards contain every generated decision or episode, including
model text, tool arguments, replayed actions, and per-case outcomes. The JSON
reports contain the original evaluation settings, counts, lineage and claim
boundaries. These six files are byte-for-byte copies of the corresponding
shared-Ceph `outputs/eval/reliable_mechet_state_ckpt14000_valid16_bs1_t4_provisional_20261007/`
and `outputs/eval/reliable_mechet_state_ckpt14000_valid16_stratified_k1_t4_provisional_20261007/`
artifacts. Their SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `gold_state_bs1/decisions.shard-00-of-01.jsonl` | `7a7fc5bb068ef1b6dda8f1a368bd2afd38bd7ab38e934a7ce29c576b5fac5d4b` |
| `gold_state_bs1/evaluation.json` | `c14bc6857387ba51092995cf867beca8a1b179fd0cd5abd3d0dd5ee5a0945ce5` |
| `gold_state_bs1/selection.json` | `a46870a00455386bbe5b5324822eefe4f8166b9e716fe8b32c33fdb8e9e21928` |
| `product_only_k1/episodes.shard-00-of-01.jsonl` | `4994686bc8612aa167fe0842333226589dba0816caacf2f3ab412ed7e4c189ea` |
| `product_only_k1/independent_episodes_audit.json` | `0ef03268301ad21ae3fcf7cb8d1fc1d01e91207bc0f71d4ddebedd294ffbc504` |
| `product_only_k1/independent_episodes_cases.jsonl` | `9ec8783eb0e5011a31e6be1fc4ea75516f0e7dbd4b45e4a6fd31eb5f0cb3a10b` |

## Training status at this handoff

- Stage I: Qwen3-0.6B State-SFT, one epoch on the strict-executable FlowER
  `257,167 / 2,890 / 28,967` reaction view; train expansion contains
  `2,007,421` decision rows. The ordinary Qingyuan 8xA100 task was
  `meteor_mechet_reliable_state_06b_1ep_8a100_qy_20261006_01`
  (instance `8b1d89c4a0d297dc01a1117ddd61597f`).
- The saved final trainer state reports `31,366 / 31,366` optimizer steps and
  epoch `1.0`. Final adapter weights are at
  `outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17/adapter_model.safetensors`
  in the shared artifact tree; weight SHA-256 is
  `fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd`.
  The final adapter and `checkpoint-31366` weights have identical SHA-256.
- Taiji recorded the **task** as failed after training/artifact save because the
  shell wrapper ended with `unexpected EOF while looking for matching` quote.
  This does not make the task a platform-level success; the saved training
  checkpoint and final adapter are separate, verified artifacts. The Stage-II
  parent lineage gate passed for the final weight SHA.
- Stage II Trajectory-SFT and Stage III EARHO have not been submitted. No
  final-adapter small test or formal full-validation/test inference is included
  in this handoff.
