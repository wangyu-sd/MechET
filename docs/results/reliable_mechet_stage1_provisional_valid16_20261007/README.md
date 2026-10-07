# Stage-I small-test artifacts and training snapshot (2026-10-07)

## RDKit trajectory visualization

- [16-case overview](stage1_valid16_overview.png): input product, recorded
  structural precursor, and predicted terminal precursor (or explicitly
  labelled last nonterminal state).
- [Trajectory comparison gallery](stage1_valid16_trajectories.html): all
  16 cases, with the frozen ground-truth tool/state trajectory and the saved
  product-only model rollout in independent side-by-side lanes. Download the
  HTML together with its 16 linked PNGs; GitHub does not render repository
  HTML inline.
- `stage1_valid16_case_01_val_1256.png` through
  `stage1_valid16_case_16_val_963.png`: directly previewable per-case trajectory
  PNGs. For example, [case 04, recorded endpoint matched](stage1_valid16_case_04_val_2344.png)
  and [case 01, nonterminal](stage1_valid16_case_01_val_1256.png).

The figure contains all 123 recorded GT decisions and all 107 model attempts,
including rejected proposals. RDKit draws the saved post-action molecular
states; the natural-language electron source/destination arrows are printed as
action labels. It does not invent geometric electron arrows on unmapped
structures. Rows in the two lanes are independently numbered: after the
trajectories diverge, equal row numbers do not imply equal chemical states.
The top comparison uses the recorded *structural* precursor view; each lane
retains the full visible intermediate mixtures. This is a visualization of the
existing checkpoint-14000 smoke, not a new inference run.

Reproduce with `scripts/visualize_reliable_stage1_valid16.py` using the
frozen strict-executable validation source, the natural-language-event-v2
validation decision file, and the saved product-only episode shard. It requires
RDKit and CairoSVG. The overview PNG SHA-256 is
`1b24016b9f2d1100f99efb5b0b2a909be1c0e94e41a063eb40e8e0770f78eb9e`;
the gallery HTML SHA-256 is
`323613575710659b604bb393df57c212d0f9d2549bd66ddfa7ab401b7359c1d8`.

This directory pairs the **saved outputs with the model-visible inputs** of the
matched 16-reaction validation smoke at the unfinished State-SFT
`checkpoint-14000`. It is a
diagnostic, not an evaluation of the final Stage-I adapter or the test split.
Both runs used the same 16 reaction IDs, adapter weights SHA-256
`5ad400212499ac32df48f9f06e432c439df15bdc24643cf8288ab7a492474507`,
seed 17, and T4/FP16 singleton decoding.

| View | Complete raw files | Recorded result |
| --- | --- | --- |
| Reference-state, one decision at a time | `gold_state_bs1/selection.json`, `evaluation.json`, `decisions.shard-00-of-01.jsonl` | 123/123 decisions present; 61/63 event-tool choices, 58/63 formally executed events, 43/63 exact events, 9/44 exact imports, 15/16 exact finishes. |
| Independent product-only, K=1 | `product_only_k1/independent_episodes_audit.json`, `independent_episodes_cases.jsonl`, `episodes.shard-00-of-01.jsonl` | 16/16 episodes present; 3/16 process-reliable structural endpoint hits, 12/16 terminal, 9/16 terminal/reference mismatches, 4/16 nonterminal. |

**Read the paired inputs and outputs here:**

- `gold_state_bs1/input_output.jsonl`: all **123** frozen system/user inputs,
  tool schemas, re-rendered model prompts, originally saved raw model
  completions, parsed calls, reference calls and per-decision scores.
- `product_only_k1/input_output.jsonl`: all **107** decisions in the 16
  product-only episodes. Each row has the system/user input, tool schema,
  re-rendered prompt, saved parsed model call, and executor acceptance/error
  and successor state. The first input of each episode is checked byte-for-byte
  against the matching frozen State-SFT input. Later inputs are reconstructed
  by independently executing the saved accepted-action prefix; every saved
  pre- and post-action state is checked against replay.

The original product-only evaluator **did not save the raw generated text**.
Thus `raw_model_completion` is explicitly `null` in those 107 paired rows; the
parsed call is the original saved output, not a fabricated verbatim completion.
The re-rendered prompt uses the saved Qwen3-0.6B tokenizer and the frozen
prompt/runtime code; it was not recorded as a string by the original run.
Tokenizer JSON SHA-256 is
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`;
chat template SHA-256 is
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
The paired-file SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `gold_state_bs1/input_output.jsonl` | `bddeab5e10e1c3c4ecb48d9e3295901bae2474115664d73a1ae9e7c0f72ba629` |
| `product_only_k1/input_output.jsonl` | `9d60fea926b5339fe3e6f055c4a673663b6766d4a5bc8c85f8af458491e83a22` |

Recreate these two files with `scripts/export_reliable_stage1_smoke_io.py`
using the frozen `data/flower_natural_language_event_sft_v2/valid.jsonl`,
`data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl`, the two saved
prediction shards below and the saved Qwen3-0.6B tokenizer directory.

The original two JSONL shards preserve the complete saved decision/episode
records and per-case outcomes. The JSON reports contain the original
evaluation settings, counts, lineage and claim boundaries. These six files
are byte-for-byte copies of the corresponding
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
