"""Hash-bound FLA launch presets; no changes to the shared installed runtime."""
import hashlib
import json
import os
from pathlib import Path
import sys


PROFILE='l2norm_bt64_w4_v1'
CONFIG_DIR='configs/runtime/route_fla_l2norm'
CONFIG_FILES=tuple(CONFIG_DIR+'/'+name+'.json' for name in
                   ('l2norm_fwd_kernel','l2norm_bwd_kernel'))
LAUNCH_CONFIG=dict(kwargs={'BT':64},num_warps=4,num_stages=2)


def kernel_config_bindings(code_root):
    root=Path(code_root)
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in CONFIG_FILES}


def profile_environment(cfg,code_root):
    profile=cfg.get('fla_kernel_profile')
    if profile is None:
        if cfg.get('runtime_config_files_sha256'):
            raise ValueError('RUNTIME_CONFIG_WITHOUT_PROFILE')
        return {}
    if profile!=PROFILE:
        raise ValueError('UNKNOWN_RUNTIME_PROFILE:'+str(profile))
    bindings=cfg.get('runtime_config_files_sha256',{})
    if set(bindings)!=set(CONFIG_FILES):
        raise ValueError('RUNTIME_CONFIG_INVENTORY_MISMATCH')
    root=Path(code_root).resolve()
    for name in CONFIG_FILES:
        try:
            raw=(root/name).read_bytes()
        except OSError as exc:
            raise ValueError('RUNTIME_CONFIG_HASH_MISMATCH:'+name) from exc
        if hashlib.sha256(raw).hexdigest()!=bindings[name]:
            raise ValueError('RUNTIME_CONFIG_HASH_MISMATCH:'+name)
        if json.loads(raw).get('default_config')!=LAUNCH_CONFIG:
            raise ValueError('UNREVIEWED_L2NORM_LAUNCH_CONFIGURATION:'+name)
    return dict(FLA_CACHE_MODE='default',FLA_CONFIG_DIR=str(root/CONFIG_DIR))


def configure_profile(cfg,code_root):
    environment=profile_environment(cfg,code_root)
    cache=sys.modules.get('fla.ops.utils.cache')
    if environment and cache is not None:
        mode=getattr(getattr(cache,'FLA_CACHE_MODE',None),'value',None)
        if mode!=environment['FLA_CACHE_MODE'] or os.environ.get('FLA_CONFIG_DIR')!=environment['FLA_CONFIG_DIR']:
            raise ValueError('RUNTIME_PROFILE_AFTER_FLA_IMPORT')
    os.environ.update(environment)
    if cfg.get('acceleration',{}).get('version')=='pr111_full_acceleration_v2':
        from .route_fla_import import import_core_fla
        import_core_fla()
    return environment


def unit_receipt_path(cfg):
    return Path(cfg['output_dir'])/'runtime_validation/l2norm_gpu_unit.json'


def require_kernel_unit(cfg,contract_sha256):
    if cfg.get('fla_kernel_profile') is None:
        return None
    try:
        report=json.loads(unit_receipt_path(cfg).read_text())
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError('STARTUP_KERNEL_UNIT_REQUIRED') from exc
    if (report.get('status')!='PASS' or report.get('profile')!=cfg['fla_kernel_profile']
            or report.get('runtime_configs')!=cfg['runtime_config_files_sha256']
            or report.get('contract_sha256')!=contract_sha256
            or cfg.get('gpu_name','H20') not in report.get('gpu','')
            or report.get('source_script_sha256')!=cfg['code_files_sha256']['scripts/verify_route_fla_kernel_profile.py']
            or report.get('autotune_search_forbidden') is not True or len(report.get('cases',[]))!=8):
        raise ValueError('STARTUP_KERNEL_UNIT_BINDING_MISMATCH')
    return report
