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
    if record.exists():
        require_native_admission(cfg,file_sha(contract));return
    from scripts.validate_pr111_native_runtime import run
    output=record.parent/('native-attempt-'+str(time.time_ns()))
    run(argparse.Namespace(output=output,source_contract=Path(cfg['acceleration_continuation']['source_contract']),
        second_contract=Path(cfg['native_admission']['second_source_contract']),candidate_contract=contract,
        profile='h20_bf16',count=4,waves=2,repeats=2,reuse_cache=None))
    report=output/'admission.json'
    atomic_json(record,dict(report=str(report),sha256=file_sha(report)))
    require_native_admission(cfg,file_sha(contract))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',type=Path,required=True)
    main(parser.parse_args().contract)
