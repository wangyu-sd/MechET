"""Bind full native rollout admission to real hardware and the stopped actor."""
from pathlib import Path
import json

from .external_route_data import file_sha

NATIVE_FILES={'src/mechet/route_gpu_backend.py','src/mechet/route_vllm_backend.py',
    'src/mechet/route_vllm_protocol.py','src/mechet/route_vllm_cache.py',
    'src/mechet/route_vllm_metrics.py','src/mechet/route_vllm_environment.py',
    'src/mechet/route_reference_cache.py','src/mechet/route_request_accounting.py',
    'scripts/serve_route_vllm.py','scripts/validate_pr111_native_runtime.py'}
REQUIRED_CHECKS={'native_probability','prompt_ids','complete_actions','native_cache_hit',
    'global_async_receipts','actual_native_refill_before_long_neighbor','actor_switch_cache_invalidation'}


def validate_compute(report,code):
    from .route_acceleration_continuation import COMPUTE_FILES
    hashes=report.get('validated_compute_files_sha256',{})
    if not (COMPUTE_FILES|NATIVE_FILES)<=hashes.keys():raise ValueError('NATIVE_COMPUTE_COVERAGE_INCOMPLETE')
    for name,digest in hashes.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_sha(code/name)!=digest:
            raise ValueError('NATIVE_ADMITTED_COMPUTE_CHANGED:'+name)


def validate_local_admission(path,code):
    report=json.loads(Path(path).read_text())
    if (report.get('status')!='PASS' or report.get('hardware_profile')!='t4_fp16'
            or not REQUIRED_CHECKS<=report.get('legal_checks',{}).keys()
            or any(v!='PASS' for v in report['legal_checks'].values())):
        raise ValueError('REAL_T4_NATIVE_LEGALITY_REQUIRED')
    validate_compute(report,code)
    # T4 speed is recorded, never substituted for actual H20 gain/admission.
    return report


def require_native_admission(cfg,contract_sha,*,code=None):
    if cfg.get('acceleration',{}).get('version')!='pr111_full_acceleration_v2':return
    expected=cfg.get('native_admission',{})
    if expected.get('required') is not True:raise ValueError('NATIVE_HARDWARE_ADMISSION_REQUIRED')
    pointer=Path(cfg['output_dir'])/'runtime_validation/native_admission.json'
    if not pointer.is_file():raise ValueError('NATIVE_HARDWARE_ADMISSION_REQUIRED')
    record=json.loads(pointer.read_text());path=Path(record['report'])
    if file_sha(path)!=record['sha256']:raise ValueError('NATIVE_HARDWARE_RECEIPT_CHANGED')
    report=json.loads(path.read_text());first=report.get('source_bindings',{}).get('first',{})
    if (report.get('status')!='PASS' or 'H20' not in report.get('hardware','')
            or report.get('hardware_profile')!='h20_bf16'
            or report.get('precision')!=cfg['policy_precision']
            or report.get('candidate_contract_sha256')!=contract_sha
            or first.get('adapter_sha256')!=expected.get('actor_sha256')
            or first.get('contract_sha256')!=expected.get('source_contract_sha256')
            or first.get('complete_sha256')!=expected.get('source_complete_sha256')
            or not REQUIRED_CHECKS<=report.get('legal_checks',{}).keys()
            or any(v!='PASS' for v in report['legal_checks'].values())):
        raise ValueError('NATIVE_HARDWARE_ADMISSION_BINDING_INVALID')
    validate_compute(report,code or Path(__file__).resolve().parents[2])
    return report
