# Native electron-flow agent pilot

This pilot asks whether an unadapted Qwen3 model can solve product-only
retrosynthesis by interacting with the existing MechET electron executor. It is
an interface and failure-analysis experiment, not a new paper result.

## Contract

- **Actor input:** one unmapped product SMILES. Reference precursors are removed
  before the dialogue is constructed.
- **Actor output:** a sequence of native calls to `inspect`,
  `import_fragment`, `apply_step`, `undo_step`, and `finish`.
- **State ownership:** the executor assigns persistent `pN` positions, performs
  every electron move, and returns the resulting molecular state. Generated
  text is never evaluated as Python.
- **Endpoint ownership:** `finish` compiles and replays a nonempty trace and
  derives the precursor from execution. The model cannot submit an independent
  reactant string.
- **Scoring:** the reference enters only after an episode ends. Full-mixture
  exact match compares canonical, unmapped fragment multisets; the reported
  neutralized variant applies the repository's existing neutralization rule.
- **Claim boundary:** formal executability checks graph/electron bookkeeping. It
  does not establish chemical feasibility, selectivity, yield, or kinetics.

The runtime lives in `src/mechet/native_electron_agent.py`. The runner in
`scripts/infer_native_electron_agent.py` uses Qwen's native Hermes tool-call
template and an explicit host loop. Non-thinking is the default; thinking is an
opt-in diagnostic. Every accepted step is transactional, failed steps roll back
completely, and accepted steps can be undone.

## Development observations (2026-09-10)

These runs used tiny, previously inspected validation samples. They are useful
for deciding what to build next, but are **not held-out benchmark estimates**.
Machine-readable provenance and counts are in
`docs/results/native_electron_agent_pilot_20260910.json`.

| Condition | Completed | Submitted | Accepted-step episodes | Exact endpoint | Main observation |
|---|---:|---:|---:|---:|---|
| Counted-state SFT, checkpoint 2500, one shot | 16 | 16 | — | 0/16 | Syntax and termination were reliable, but formal execution was 1/16 |
| Original Qwen, native tools, thinking | 3/4 | 1/3 | 1/3 | 0/3 observed | One executable but incorrect endpoint; run stopped before all four cases |
| Original Qwen, native tools, non-thinking | 4/4 | 0/4 | 0/4 | 0/4 | Tool syntax became cheap, but action policy looped until the turn budget |

For the completed non-thinking native run, 96 turns generated 3,661 tokens and
issued 44 `inspect`, 32 `apply_step`, 10 `undo_step`, 8 `import_fragment`, and 2
`finish` calls. Thirty-nine calls failed; the dominant executor error was
`SOURCE_HAS_NO_ELECTRON_PAIR` (16). All four episodes reached the 24-turn
budget. This isolates the bottleneck: a native tool interface removes Python
syntax and hidden atom-map parsing as explanations, but an original Qwen model
still lacks a competent electron-action policy.

The older one-shot run provides a complementary diagnosis. It produced valid
program syntax and a finish call on all 16 samples, yet only one program
executed and its endpoint was wrong. Five programs had missing sites and five
had stale state assertions. Ignoring the assertions did not rescue those five:
they subsequently failed position lookup. Eight failed before their first
electron step and seven after one step. The failure is therefore not explained
by the 1.6% build quarantine or by formatting alone.

## Reproduction

Prepare JSONL rows with `id`, `target_smiles`, and optionally
`expected_precursor`. The optional reference is stripped from actor input and
used only for final scoring.

```bash
PYTHONPATH=src python scripts/infer_native_electron_agent.py \
  --model /path/to/Qwen3-8B \
  --cases /path/to/development_cases.jsonl \
  --output outputs/eval/native_agent_pilot \
  --limit 4 --max-turns 24
```

Add `--thinking` only for a matched diagnostic. A publishable evaluation still
requires a frozen held-out split, a frozen checkpoint/interface, complete
denominators, multiple seeds where required by the paper protocol, and matched
baselines.

## Artifacts intentionally not committed

Raw generations contain long model transcripts and local model paths, so this
PR records immutable hashes and aggregate statistics rather than copying those
runtime artifacts into Git. The source artifacts remain under `outputs/eval/`
in the experiment workspace named in the JSON record.
