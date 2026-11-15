"""Freeze the isolated rollout runtime independently of the HF learner."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .route_vllm_adapter import sha256

VERSIONS={'vllm':'0.24.0+cu129','torch':'2.11.0+cu129','transformers':'5.5.4'}


def validate_runtime(cfg):
    backend=cfg.get('inference_backend','hf')
    if backend not in ('hf','vllm'):raise ValueError('UNRECOGNIZED_ROUTE_INFERENCE_BACKEND')
    if backend=='hf':
        if cfg.get('vllm_runtime'):raise ValueError('VLLM_RUNTIME_WITH_HF_BACKEND')
        return
    runtime=cfg['vllm_runtime']
    if runtime.get('software_replay'):
        raise ValueError('VLLM_SOFTWARE_PROBE_FORBIDDEN_IN_FORMAL_CONTRACT')
    metadata_present='metadata_helper' in runtime or 'metadata_index' in runtime
    if metadata_present and (not runtime.get('metadata_helper')
            or not isinstance(runtime.get('metadata_index'),dict)
            or not runtime['metadata_index'].get('path')):
        raise ValueError('VLLM_INCOMPLETE_METADATA_OVERLAY')
    bundle_present='source_bundle_helper' in runtime or 'source_bundle' in runtime
    if bundle_present and (not runtime.get('source_bundle_helper')
            or not isinstance(runtime.get('source_bundle'),dict)
            or not runtime['source_bundle'].get('path') or not runtime['source_bundle'].get('sha256')
            or not runtime['source_bundle'].get('site_root')):
        raise ValueError('VLLM_INCOMPLETE_SOURCE_BUNDLE')
    native_key_present='torch_native_key' in runtime or 'torch_native_key_helper' in runtime
    if native_key_present and (not bundle_present or not runtime.get('torch_native_key_helper')
            or not isinstance(runtime.get('torch_native_key'),dict)
            or not runtime['torch_native_key'].get('path') or not runtime['torch_native_key'].get('sha256')):
        raise ValueError('VLLM_INCOMPLETE_TORCH_NATIVE_KEY')
    inventory=None
    if bundle_present and 'directory_inventory' in runtime['source_bundle']:
        inventory=runtime['source_bundle']['directory_inventory']
        if (not isinstance(inventory,dict) or not inventory.get('path')
                or not inventory.get('sha256') or not inventory.get('site_root')
                or not inventory.get('archive_sha256')):
            raise ValueError('VLLM_INCOMPLETE_DIRECTORY_INVENTORY')
        if (inventory['site_root']!=runtime['source_bundle']['site_root']
                or inventory['archive_sha256']!=runtime['source_bundle']['sha256']):
            raise ValueError('VLLM_DIRECTORY_INVENTORY_BINDING')
    shared=Path('/aaa/fionafyang/buddy1').resolve()
    for key in ('python','cache_root','cuda_compat','manifest','bootstrap','import_helper'):
        if not Path(runtime[key]).resolve().is_relative_to(shared):
            raise ValueError('VLLM_RUNTIME_REQUIRES_CEPH:'+key)
    if metadata_present:
        for path in (runtime['metadata_helper'],runtime['metadata_index']['path']):
            if not Path(path).resolve().is_relative_to(shared):
                raise ValueError('VLLM_METADATA_REQUIRES_CEPH')
    if native_key_present:
        for path in (runtime['torch_native_key_helper'],runtime['torch_native_key']['path']):
            if not Path(path).resolve().is_relative_to(shared):raise ValueError('VLLM_TORCH_NATIVE_KEY_REQUIRES_CEPH')
    if bundle_present:
        for path in (runtime['source_bundle_helper'],runtime['source_bundle']['path'],
                     runtime['source_bundle']['site_root']):
            if not Path(path).resolve().is_relative_to(shared):raise ValueError('VLLM_SOURCE_BUNDLE_REQUIRES_CEPH')
        site=Path(runtime['python']).parent.parent/'lib/python3.11/site-packages'
        if Path(runtime['source_bundle']['site_root']).resolve()!=site.resolve():
            raise ValueError('VLLM_SOURCE_BUNDLE_SITE_BINDING')
        if inventory and not Path(inventory['path']).resolve().is_relative_to(shared):
            raise ValueError('VLLM_DIRECTORY_INVENTORY_REQUIRES_CEPH')
    capacities=('max_num_batched_tokens','kv_cache_memory_bytes','request_timeout_seconds')
    if 'startup_timeout_seconds' in runtime:capacities+=('startup_timeout_seconds',)
    for key in capacities:
        if type(runtime[key]) is not int or runtime[key]<1:raise ValueError('VLLM_RUNTIME_CAPACITY:'+key)
    fraction=runtime['gpu_memory_utilization']
    if not isinstance(fraction,(int,float)) or not math.isfinite(fraction) or not 0<fraction<=.7:
        raise ValueError('VLLM_LEARNER_MEMORY_RESERVATION_REQUIRED')
    path=Path(runtime['manifest'])
    if cfg['files_sha256'].get(str(path))!=sha256(path):raise ValueError('VLLM_RUNTIME_MANIFEST_HASH_CHANGED')
    manifest=json.loads(path.read_text())
    if any(manifest['package_versions'].get(k)!=v for k,v in VERSIONS.items()):
        raise ValueError('VLLM_PINNED_PACKAGES_REQUIRED')
    if manifest['python']!=runtime['python'] or manifest['cuda_compat']!=runtime['cuda_compat']:
        raise ValueError('VLLM_RUNTIME_EXECUTABLE_BINDING')
    required=(Path(runtime['python']),Path(runtime['import_helper']),
        Path(runtime['bootstrap'])/'sitecustomize.py',
        Path(runtime['cuda_compat'])/'libcuda.so.1',
        Path(runtime['transformers_import_index']['path']))
    if metadata_present:
        required+=(Path(runtime['metadata_helper']),Path(runtime['metadata_index']['path']))
    if bundle_present:
        required+=(Path(runtime['source_bundle_helper']),Path(runtime['source_bundle']['path']))
    if inventory:required+=(Path(inventory['path']),)
    if native_key_present:
        required+=(Path(runtime['torch_native_key_helper']),Path(runtime['torch_native_key']['path']))
    if any(str(path.resolve()) not in manifest['files_sha256'] for path in required):
        raise ValueError('VLLM_RUNTIME_UNBOUND_BOOTSTRAP')
    if metadata_present and runtime['metadata_index'].get('sha256')!=manifest['files_sha256'][
            str(Path(runtime['metadata_index']['path']).resolve())]:
        raise ValueError('VLLM_METADATA_INDEX_BINDING_MISMATCH')
    if bundle_present and runtime['source_bundle']['sha256']!=manifest['files_sha256'][
            str(Path(runtime['source_bundle']['path']).resolve())]:
        raise ValueError('VLLM_SOURCE_BUNDLE_HASH_BINDING')
    if inventory and inventory['sha256']!=manifest['files_sha256'][str(Path(inventory['path']).resolve())]:
        raise ValueError('VLLM_DIRECTORY_INVENTORY_HASH_BINDING')
    if native_key_present and runtime['torch_native_key']['sha256']!=manifest['files_sha256'][
            str(Path(runtime['torch_native_key']['path']).resolve())]:
        raise ValueError('VLLM_TORCH_NATIVE_KEY_HASH_BINDING')
    def verify_file(item):
        name,digest=item
        if not Path(name).resolve().is_relative_to(shared) or sha256(name)!=digest:
            raise ValueError('VLLM_RUNTIME_FILE_CHANGED:'+name)
    # Preserve every resolved-path/hash check, overlap independent Ceph RPCs.
    with ThreadPoolExecutor(max_workers=16) as pool:
        for _ in pool.map(verify_file,manifest['files_sha256'].items()):pass


def configure_runtime(cfg,runtime):
    runtime=dict(runtime);path=Path(runtime['manifest']).resolve();runtime['manifest']=str(path)
    cfg.update(inference_backend='vllm',vllm_runtime=runtime)
    cfg['files_sha256'][str(path)]=sha256(path)
    validate_runtime(cfg)
