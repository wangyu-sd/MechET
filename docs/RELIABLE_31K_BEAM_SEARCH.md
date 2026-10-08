# State-SFT product-only beam search on mech-USPTO-31k

The 2026-10-07 zero-shot State-SFT transfer was **greedy K=1**: one proposed
next action per state, with no surviving branch after a rejected action. It
reached `finish_trace` for 663/1,253 executable trace-view reactions. This is
not the full 3,120-reaction endpoint test, and its rate is not a beam result.

Set `MECHET_SEARCH_MODE=beam` when running
`scripts/run_taiji_reliable_uspto31k_state_transfer_8a100.sh`. The separate
output directory is
`outputs/eval/reliable_mechet_state_uspto31k_trace1253_beam4_seed17_20261008/`;
the existing K=1 directory is never overwritten. The frozen product-only
prompt, Stage-I Qwen3-0.6B adapter, private product remapping, 40-decision/32-
import budgets, 1,253 IDs and executor remain unchanged. Each live molecular
state generates four deterministic token-beam next-action proposals; the
executor retains up to four distinct search states at every depth. Every
terminal candidate is kept. States with the same visible SMILES are **not**
merged when their private atom aliases, import inventory or visited sets differ.
Selection is by mean raw-model token log-probability, without a value critic or
access to the reference precursor. `top1_exact` is the top-ranked terminal;
`pass_at_beam` asks whether *any* generated terminal matches the reference and
is an oracle/coverage metric, not a usable top-1 predictor.

Each `results.shard-*.jsonl` row stores the complete explored one-reaction
electron-action tree:

- `search_tree`: every accepted state, including pruned and deduplicated nodes,
  with stable `node_id`, `parent_node_id`, depth, state, status and score;
- `attempts`: every parsed action offered to the executor, its parent/child IDs,
  raw action, acceptance, resulting state or rejection reason;
- `generations`: all raw model generations, including unparseable and duplicate
  proposals, with their log-probabilities and whether execution was attempted;
- `terminal_node_ids`, `frontier_node_ids` and `top_node_id`: reconstructable
  leaves and the final ranking decision.

The audit's `--beam-search` mode verifies parent/child edges and terminal
coverage, then reports both top-1 and oracle beam coverage. No reference
endpoint is read by generation, branch expansion, pruning or ranking. This is
an **electron-action search tree for a single reaction**, not multi-reaction
synthetic-route planning. Larger beam budgets increase inference compute and
must be reported alongside accuracy.

Local execution smoke (one reaction, beam width/proposals 2, T4/FP16,
`max_new_tokens=256`) is preserved at
`outputs/eval/reliable_mechet_state_uspto31k_beam_smoke1_20261008/` in the
shared workspace. It produced 30 state nodes, 38 executor attempts and 40 raw
generations, and passed the tree-edge audit. It found no terminal and is **not**
an accuracy estimate or a substitute for the 1,253-reaction beam evaluation.
