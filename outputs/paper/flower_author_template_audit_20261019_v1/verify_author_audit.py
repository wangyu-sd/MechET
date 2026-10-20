"""Fresh unit/artifact verification; retain exact pre-cleanup matcher source."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from rdkit import rdBase

from run_author_audit import save_json, sha

HERE = Path(__file__).resolve().parent
OUT = HERE/'results_v2'


def main():
    d = json.loads((OUT/'delivery_summary.json').read_text())
    s = json.loads((OUT/'summary.json').read_text())
    for p,h in d['artifact_sha256'].items():
        assert sha(p) == h, p
    assert s['status'] == 'complete'
    assert sum(s['populations'].values()) == 189233
    assert sum(x['programs'] for x in s['program_coverage'].values()) == 37922
    assert sum(x['event_occurrences'] for x in s['program_coverage'].values()) == 124808
    # After matching finished, fix ONLY closing the catalog file handle. Keep
    # the exact original runtime source recoverable instead of falsifying its
    # recorded SHA to point at the cleaned-up file.
    p = HERE/'author_template_match.py'
    text = p.read_text()
    original = text.replace('enumerate(Path(path).read_text().splitlines(), 1)',
                            'enumerate(Path(path).open(), 1)')
    runtime_sha = hashlib.sha256(original.encode()).hexdigest()
    expected = next(h for k,h in d['local_code_sha256'].items() if k.endswith('/author_template_match.py'))
    assert runtime_sha == expected
    save_json(OUT/'runtime_matcher_source.json', dict(sha256=runtime_sha, source=original,
        post_run_change='I/O-only file-handle cleanup; no query, binding, chemistry or matching logic changed'))
    unit = subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],
                          capture_output=True,text=True)
    (OUT/'unit_tests.log').write_text(unit.stdout+unit.stderr)
    assert unit.returncode == 0
    assert 'Ran 15 tests' in unit.stderr
    assert 'ResourceWarning' not in unit.stderr
    gz = sorted(str(p) for p in OUT.glob('*.jsonl.gz'))
    checked = subprocess.run(['gzip','-t',*gz],capture_output=True,text=True)
    assert checked.returncode == 0, checked.stderr
    result = dict(status='passed', standalone_tests=15, gzip_integrity='passed',
        population_reconciliation='189233 events / 37922 programs / 124808 program event occurrences',
        rdkit_version=rdBase.rdkitVersion, python_version=sys.version,
        historical_delivery_hashes_verified=True, original_matcher_sha256=runtime_sha,
        post_run_changes=['close catalog input handle; add regression assertion for no ResourceWarning'],
        current_code_sha256={str(p):sha(p) for p in HERE.glob('*.py')},
        artifact_sha256={str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='verification.json'},
        main_repository_suite=dict(status='collection_errors',
            tests=['tests/test_mech_uspto31k_causal_import.py: missing mechet.a7_rescue',
                   'tests/test_nl_history_beam.py: missing mechet.in_place_grounded_flow']))
    save_json(OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('current_code_sha256','artifact_sha256')},indent=2))


if __name__=='__main__':
    main()
