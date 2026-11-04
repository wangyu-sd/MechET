"""Extract previously saved real failures as SOFTWARE fixtures; no inference."""
import gzip
import json
from pathlib import Path

HERE = Path(__file__).parent
OLD = HERE.parent/'flower_author_template_audit_20261019_v1/results_v2'
IDS = {'86fab32321c52b529b79d261463a51f29da55892b0ac11397b2fb29bb3324022',
       'gt:textbook-tool-sft:flower_mech_proof_test_9:0',
       'gt:textbook-tool-sft:flower_mech_proof_test_9:1'}
rows = {}
with gzip.open(OLD/'event_inputs.jsonl.gz', 'rt') as f:
    for line in f:
        r = json.loads(line)
        if r['event_id'] in IDS:
            rows[r['event_id']] = r
        if len(rows)==len(IDS):
            break
assert len(rows)==3
(HERE/'regression_fixtures.json').write_text(json.dumps(rows, indent=2)+'\n')
print('Saved three existing cases as unit fixtures, NOT an experimental evaluation.')
