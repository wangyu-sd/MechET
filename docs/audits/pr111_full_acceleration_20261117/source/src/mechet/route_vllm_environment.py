"""Child-only environment: preserve learner ABI, isolate GPU and Ceph caches."""
import os
import hashlib
from pathlib import Path


def engine_import_paths(paths,code_root,bootstrap=None):
    """Keep stdlib/dependencies ahead of deployment-only module directories."""
    root=os.path.abspath(code_root)
    deployment={root,os.path.join(root,'src'),os.path.join(root,'scripts')}
    if bootstrap:deployment.add(os.path.abspath(bootstrap))
    result=[];seen=set()
    for path in paths:
        if not isinstance(path,str):continue
        key=os.path.abspath(path)
        if key in deployment or key in seen:continue
        result.append(path);seen.add(key)
    return result+[root,os.path.join(root,'src')]


def child_environment(cfg,local_rank):
    runtime=cfg['vllm_runtime'];root=Path(runtime['cache_root']).resolve()
    shared=Path('/aaa/fionafyang/buddy1').resolve()
    if not root.is_relative_to(shared):raise ValueError('VLLM_CACHE_REQUIRES_CEPH')
    shared_triton=runtime.get('shared_triton_cache_root')
    if shared_triton is not None:
        shared_triton=Path(shared_triton).resolve()
        if not shared_triton.is_relative_to(root):
            raise ValueError('TRITON_SHARED_CACHE_OUTSIDE_RUNTIME_ROOT')
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE',
                'GROUP_RANK','GROUP_WORLD_SIZE','ROLE_RANK','ROLE_WORLD_SIZE','ROLE_NAME',
                'MASTER_ADDR','MASTER_PORT','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR',
                'CUDA_HOME','CUDA_PATH'):
        env.pop(key,None)
    # torchrun sets USE_AGENT_STORE=True. A standalone vLLM rank0 must host
    # its own TCPStore; inheriting this flag turns it into a client waiting
    # for a nonexistent agent on vLLM's fresh random port (600s timeout).
    for key in list(env):
        if key.startswith('TORCHELASTIC_'):env.pop(key)
    visible=env.get('CUDA_VISIBLE_DEVICES')
    gpu=visible.split(',')[local_rank] if visible else str(local_rank)
    env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
               VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               VLLM_USE_FLASHINFER_SAMPLER='0',PYTHONSAFEPATH='1',
               VLLM_USE_V2_MODEL_RUNNER='0',
               TRITON_CACHE_AUTOTUNING='1')
    if cfg.get('acceleration',{}).get('version')=='pr111_full_acceleration_v2':
        env['TRITON_STORE_BINARY_ONLY']='1'
    env['PYTHONPATH']=runtime.get('bootstrap','')
    env['PATH']=str(Path(runtime['python']).parent)+':'+env.get('PATH','')
    # Linux AF_UNIX permits at most 107 pathname bytes. vLLM appends a UUID;
    # multiprocessing also creates sockets below TMPDIR. Keep both short on Ceph
    # independently of the (often much longer) immutable runtime/cache location.
    identity=hashlib.sha256(os.fsencode(root)).hexdigest()[:16]
    ipc=(shared/'.mechet-ipc'/str(os.getuid())/identity/f'r{local_rank:02d}').resolve()
    if not ipc.is_relative_to(shared) or len(os.fsencode(ipc))+1+36>107:
        raise ValueError('VLLM_IPC_PATH_NOT_SHORT_SHARED_STORAGE')
    ipc.mkdir(parents=True,exist_ok=True,mode=0o700)
    if ipc.stat().st_uid!=os.getuid() or ipc.stat().st_mode & 0o077:
        raise ValueError('VLLM_IPC_DIRECTORY_NOT_PRIVATE')
    env.update(TMPDIR=str(ipc),TMP=str(ipc),TEMP=str(ipc),VLLM_RPC_BASE_PATH=str(ipc))
    folders=dict(XDG_CACHE_HOME='cache',
        CUDA_CACHE_PATH='cuda',VLLM_CACHE_ROOT='vllm',VLLM_CONFIG_ROOT='vllm_config',
        VLLM_ASSETS_CACHE='vllm_assets',
        FLASHINFER_WORKSPACE_BASE='flashinfer',FLASHINFER_CACHE_DIR='flashinfer',
        TVM_FFI_CACHE_DIR='tvm_ffi',NUMBA_CACHE_DIR='numba',CUTE_DSL_CACHE_DIR='cute_dsl',
        TORCHINDUCTOR_CACHE_DIR='inductor',TRITON_CACHE_DIR='triton',
        HF_HOME='huggingface',TORCH_HOME='torch',TORCH_EXTENSIONS_DIR='torch_extensions')
    for key,folder in folders.items():
        path=(shared_triton if key=='TRITON_CACHE_DIR' and shared_triton is not None
              else (root/f'rank{local_rank:02d}'/folder).resolve())
        if not path.is_relative_to(shared):raise ValueError('VLLM_RESOLVED_CACHE_OUTSIDE_CEPH')
        path.mkdir(parents=True,exist_ok=True)
        env[key]=str(path)
    # Only this subprocess sees the forward-compatibility driver libraries.
    compat=Path(runtime['cuda_compat']).resolve()
    if not compat.is_relative_to(shared) or not (compat/'libcuda.so.1').exists():
        raise ValueError('VLLM_CUDA_COMPAT_NOT_VERIFIED')
    env['LD_LIBRARY_PATH']=str(compat)+':'+env.get('LD_LIBRARY_PATH','')
    return env
