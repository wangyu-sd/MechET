"""Pre-training numerical software gate on an already allocated H20 (no update)."""
import argparse
import json
from pathlib import Path
import time

from mechet.external_route_data import file_sha
from mechet.route_native_admission import require_native_admission
from mechet.route_training_contract import atomic_json


def main(contract):
    cfg=json.loads(contract.read_text())
    if cfg.get('acceleration',{}).get('version')!='pr111_full_acceleration_v2':return
    expected=cfg['native_admission']
    if file_sha(expected['second_source_contract'])!=expected['second_source_contract_sha256']:
        raise ValueError('NATIVE_SECOND_SOURCE_CONTRACT_CHANGED')
    source=cfg['acceleration_continuation']
    if file_sha(source['source_contract'])!=expected['source_contract_sha256']:
        raise ValueError('NATIVE_SOURCE_CONTRACT_CHANGED')
    record=Path(cfg['output_dir'])/'runtime_validation/native_admission.json'
    latch=record.parent/'native_admission_failure.json'
    if latch.exists():raise ValueError('FROZEN_NATIVE_FAILURE_REQUIRES_REPAIR:'+str(latch))
    if record.exists():
        require_native_admission(cfg,file_sha(contract));return
    from scripts.validate_pr111_native_runtime import run
    output=record.parent/('native-attempt-'+str(time.time_ns()))
    report=output/'admission.json'
    from scripts import pr96_earho_contract as probability
    from mechet.route_probability_admission import ScoreWitnesses,replay_failed_probability_receipts
    try:
        # run() requires a fresh output directory. Witnesses start only after
        # the original score call, once that directory has been created.
        with ScoreWitnesses(probability,output/'teacher_scores.jsonl') as trace:
            run(argparse.Namespace(output=output,source_contract=Path(cfg['acceleration_continuation']['source_contract']),
                second_contract=Path(cfg['native_admission']['second_source_contract']),candidate_contract=contract,
                profile='h20_bf16',count=4,waves=2,repeats=2,reuse_cache=None))
        previous=expected.get('previous_probability_failure')
        if previous:
            for key in ('report','outputs'):
                if file_sha(previous[key])!=previous[key+'_sha256']:
                    raise ValueError('PREVIOUS_NATIVE_FAILURE_BYTES_CHANGED:'+key)
            import torch
            with torch.inference_mode(), ScoreWitnesses(probability,output/'previous_teacher_scores.jsonl'):
                replay=replay_failed_probability_receipts(previous['report'],previous['outputs'],trace.primary_model,
                    trace.pad_token_id,cfg['policy_precision'],probability.token_logps_batch,
                    expected_actor=expected['actor_sha256'])
            atomic_json(output/'previous_failed_case_replay.json',replay)
            combined=json.loads(report.read_text());combined['legal_checks']['previous_failed_case_replay']='PASS'
            combined['previous_failed_case_replay_sha256']=file_sha(output/'previous_failed_case_replay.json')
            atomic_json(report,combined)
    except BaseException as exc:
        atomic_json(latch,dict(contract_sha256=file_sha(contract),report=str(report),
            error=type(exc).__name__+':'+str(exc),retry='do not repeat unchanged frozen conditions; repair and revalidate'))
        if report.exists():
            failed=json.loads(report.read_text());failed.update(status='FAIL',hardware_gate_error=type(exc).__name__+':'+str(exc))
            atomic_json(report,failed)
        raise
    atomic_json(record,dict(report=str(report),sha256=file_sha(report)))
    require_native_admission(cfg,file_sha(contract))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',type=Path,required=True)
    main(parser.parse_args().contract)
