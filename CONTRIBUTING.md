# Contributing

1. Keep MECH_ET v3 as the default CoT contract (`PERCEIVE` → `ET_SIGNATURE` → graph + `BE_DELTA` → reactants).
2. Prefer process-verifiable changes (format / reachability / BE exact / answer).
3. Do not commit full FlowER dumps or multi-GB JSONL; ship only `data/samples/`.
4. Run `pytest -q tests/test_mech_et.py` before opening a PR.
