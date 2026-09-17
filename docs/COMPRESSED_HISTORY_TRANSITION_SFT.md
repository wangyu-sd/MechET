# Compressed-history transition SFT

This condition extends the validated natural-language event SFT with a compact,
executor-owned history capsule. It uses only the standard replayed FlowER gold
trajectory. Actor errors, failed execution, reference suffixes, expected
precursors, and gold remaining-step counts are not model-visible.

The current molecular SMILES is the chemical state. Repeating every earlier
SMILES would make context grow quadratically without adding chemical
information. Each decision therefore receives only:

- the ordered types of accepted past actions;
- committed import-batch and fragment counts;
- the number of accepted electron events;
- the immediately preceding tool name and result code.

The runtime can reconstruct these fields from its own accepted transitions.
No continuous raw-text history window is required. Existing supervised tool
calls and paired authoritative tool results are unchanged.

Build a diagnostic artifact with:

```bash
python scripts/build_natural_language_history_sft.py \
  --source-dir data/flower_natural_language_event_sft_v1_rdkit2026 \
  --output-dir data/flower_natural_language_event_history_v1_smoke \
  --limit-reactions 16
```

Omit `--limit-reactions` for the frozen strict-executable reaction universe of
257,167 train, 2,890 validation, and 28,967 test reactions. Limited artifacts
are always marked training-forbidden.

