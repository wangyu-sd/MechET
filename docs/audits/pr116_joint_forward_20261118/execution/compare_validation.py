"""Compare a completed PR116 held-out panel with the same frozen Stage-II roots."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SUBMISSIONS = PROJECT/'outputs/submissions'
R = SUBMISSIONS/'pr116_r_64x32_4h20_20261118_01'
BASELINE = SUBMISSIONS/'pr116_step0_4h20_512_20261118_01'
sys.path.insert(0, str(R/'code/src'))

from mechet.external_route_data import file_sha, iter_records, signature
from mechet.external_route_validation import summarize_validation
from mechet.route_joint_schedule import VALIDATION_COUNTS
from mechet.route_training_contract import atomic_json


def read(path):
    return json.loads(path.read_bytes())


def changes(ids, before, after, key):
    old = {i for i in ids if before[i][key]}
    new = {i for i in ids if after[i][key]}
    return dict(before=len(old), after=len(new), new=len(new-old), lost=len(old-new),
                retained=len(old&new))


def panel(cfg_path, step, ids):
    cfg = read(cfg_path)
    output = Path(cfg['output_dir'])
    root = output/'validation'/f'checkpoint-{step:06d}'
    report_file = root/'summary.json'
    outcomes_file = root/'root_outcomes.json'
    report = read(report_file)
    outcomes = read(outcomes_file)
    computed = summarize_validation(ids, outcomes, 4)
    if not computed['complete'] or any(report[k] != value for k, value in computed.items()):
        raise ValueError('validation root outcomes do not reproduce the published full panel')
    binding = report['binding']
    adapter = Path(report['checkpoint_path'])/'adapter_model.safetensors'
    if (binding['contract_sha256'] != file_sha(cfg_path) or binding['task_ids_sha256'] != signature(ids)
            or binding['protocol_sha256'] != signature(cfg['validation_protocol'])
            or report['protocol'] != cfg['validation_protocol'] or report['checkpoint_round'] != step
            or report['validation_reference_split'] != 'validation'
            or binding['actor_sha256'] != report['actor_sha256'] or file_sha(adapter) != report['actor_sha256']):
        raise ValueError('validation panel binding changed')
    return report, {o['task_id']: o for o in outcomes}, dict(
        summary_sha256=file_sha(report_file), outcomes_sha256=file_sha(outcomes_file),
        actor_sha256=report['actor_sha256'])


def compare(step):
    latest = R/'contract.json'
    previous = BASELINE/'contract.json'
    cfg, baseline = read(latest), read(previous)
    if (cfg['dataset_manifest_sha256'] != baseline['dataset_manifest_sha256']
            or cfg['validation_protocol'] != baseline['validation_protocol']):
        raise ValueError('baseline and ReFIT evaluator settings differ')
    cohort = Path(cfg['step_budget']['valid'])
    if file_sha(cohort) != file_sha(baseline['clean_valid_file']):
        raise ValueError('baseline and ReFIT validation cohort files differ')
    all_ids = [r['task_id'] for r in iter_records(cohort)]
    ids = all_ids[:VALIDATION_COUNTS[step]]
    _, before, source = panel(previous, 0, all_ids)
    report, after, destination = panel(latest, step, ids)
    result = dict(step=step, targets=len(ids), routes_per_target=4, task_ids_sha256=signature(ids),
        baseline_source=source, refit_source=destination,
        inventory_closed=changes(ids,before,after,'native_stock_closed'),
        reference_verified=changes(ids,before,after,'verified_closed'),
        censored_roots=changes(ids,before,after,'censored'),
        model_decision_requests=dict(before=sum(before[i]['model_decision_requests'] for i in ids),
                                     after=report['model_decision_requests']))
    output = HERE/f'refit_step_{step:06d}_paired_validation.json'
    if output.exists() and read(output) != result:
        raise ValueError('paired validation report identity changed')
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', type=int, choices=[8, 16, 32, 64])
    print(json.dumps(compare(parser.parse_args().step), indent=2))
