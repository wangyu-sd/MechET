# Forward expert data layout

Generated data and checkpoints are intentionally excluded from version control.
The recommended local layout is:

```text
data/
  raw/<source>/manifest.json
  forward_expert/
    reactions.jsonl
    reactions.report.json
    reactions.quarantine.jsonl
    steps/train.jsonl
    steps/valid.jsonl
    steps/test.jsonl
    steps/manifest.json
models/
  baselines/<model>/manifest.json
outputs/
  forward_expert/<run>/best/
  forward_expert/<run>/last/
```

`reactions.jsonl` is the source-level canonical record. `steps/*.jsonl` contains
only labeled source-sink trajectory steps. Rows without unambiguous arrow labels
remain useful for reaction-compatibility training and should not be assigned
synthetic arrow labels unless they are separately verified.

Licenses are source-specific. The downloader records upstream metadata but does
not grant redistribution rights. Restricted sources require explicit acceptance.
