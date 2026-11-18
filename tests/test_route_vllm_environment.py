from pathlib import Path
import os
import socket
from uuid import uuid4

import pytest


def test_child_uses_only_assigned_gpu_and_isolated_ceph_cache_paths(tmp_path,monkeypatch):
    from mechet.route_vllm_environment import child_environment
    compat=tmp_path/'compat';compat.mkdir();(compat/'libcuda.so.1').write_bytes(b'fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,5,6,7')
    monkeypatch.setenv('PYTHONPATH','/old/pinned/torch26')
    monkeypatch.setenv('RANK','3');monkeypatch.setenv('CUDA_HOME','/old/cuda118')
    original=dict(os.environ)
    env=child_environment({'vllm_runtime':{'cache_root':str(tmp_path/'cache'),
        'cuda_compat':str(compat),'python':str(tmp_path/'venv/bin/python')}},1)
    assert env['CUDA_VISIBLE_DEVICES']=='5'
    assert '/old/pinned/torch26' not in env['PYTHONPATH']
    assert 'RANK' not in env and 'CUDA_HOME' not in env
    for name in ('CUDA_CACHE_PATH','VLLM_CACHE_ROOT','FLASHINFER_WORKSPACE_BASE',
                 'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR'):
        assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve())
    assert env['LD_LIBRARY_PATH'].split(':')[0]==str(compat)
    assert env['PATH'].split(':')[0]==str(tmp_path/'venv/bin')
    assert dict(os.environ)==original


def test_unknown_backend_cannot_silently_fall_back_to_hf():
    from mechet.route_vllm_contract import validate_runtime
    with pytest.raises(ValueError,match='INFERENCE_BACKEND'):
        validate_runtime({'inference_backend':'vlllm'})


def test_native_child_cannot_inherit_a_torchrun_agent_store(tmp_path,monkeypatch):
    import json,subprocess,sys
    from mechet.route_vllm_environment import child_environment
    compat=tmp_path/'compat';compat.mkdir();(compat/'libcuda.so.1').write_bytes(b'fixture')
    poison={'TORCHELASTIC_USE_AGENT_STORE':'True','TORCHELASTIC_RUN_ID':'parent',
        'TORCHELASTIC_ERROR_FILE':'parent-error.json','TORCHELASTIC_RESTART_COUNT':'0',
        'LOCAL_WORLD_SIZE':'4','GROUP_RANK':'0','GROUP_WORLD_SIZE':'1',
        'ROLE_RANK':'3','ROLE_WORLD_SIZE':'4','ROLE_NAME':'trainer'}
    for key,value in poison.items():monkeypatch.setenv(key,value)
    env=child_environment({'vllm_runtime':{'cache_root':str(tmp_path/'cache'),
        'cuda_compat':str(compat),'python':str(tmp_path/'venv/bin/python')}},0)
    leaked=json.loads(subprocess.check_output([sys.executable,'-S','-c',
        'import os,json;print(json.dumps({k:v for k,v in os.environ.items() if k.startswith("TORCHELASTIC_") or k in '+repr(list(poison))+'}))'],env=env,text=True))
    assert leaked=={}, 'Native rank0 must create its own TCPStore, rather than connect to a nonexistent parent agent store'
    assert all(os.environ[k]==v for k,v in poison.items())


@pytest.mark.parametrize('inherited',[None,'/tmp/inherited-wrong-location'])
def test_long_ceph_cache_path_still_allows_real_unix_socket_binding(tmp_path,monkeypatch,inherited):
    from mechet.route_vllm_environment import child_environment
    compat=tmp_path/'compat';compat.mkdir();(compat/'libcuda.so.1').write_bytes(b'fixture')
    cache=tmp_path/('long-cache-'+'a'*100)
    if inherited is None:monkeypatch.delenv('VLLM_RPC_BASE_PATH',raising=False)
    else:monkeypatch.setenv('VLLM_RPC_BASE_PATH',inherited)
    env=child_environment({'vllm_runtime':{'cache_root':str(cache),
        'cuda_compat':str(compat),'python':str(tmp_path/'venv/bin/python')}},0)
    # Reproduce vLLM's real UUID filename, including its kernel byte limit.
    address=Path(env.get('VLLM_RPC_BASE_PATH',env['TMPDIR']))/str(uuid4())
    assert len(os.fsencode(address))<=107
    assert address.parent.resolve().is_relative_to(Path('/aaa/fionafyang/buddy1').resolve())
    assert env['TMPDIR']==env['VLLM_RPC_BASE_PATH']==env['TMP']==env['TEMP']
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        try:
            server.bind(str(address))
        finally:
            address.unlink(missing_ok=True)
    other=child_environment({'vllm_runtime':{'cache_root':str(cache),
        'cuda_compat':str(compat),'python':str(tmp_path/'venv/bin/python')}},1)
    assert other['VLLM_RPC_BASE_PATH']!=env['VLLM_RPC_BASE_PATH']


def test_formal_contract_cannot_enable_software_base_model_probe():
    from mechet.route_vllm_contract import validate_runtime
    with pytest.raises(ValueError,match='SOFTWARE_PROBE_FORBIDDEN'):
        validate_runtime({'inference_backend':'vllm','vllm_runtime':{'software_replay':True}})


def test_child_disables_nvcc_dependent_flashinfer_sampler(tmp_path,monkeypatch):
    from mechet.route_vllm_environment import child_environment
    compat=tmp_path/'compat';compat.mkdir();(compat/'libcuda.so.1').write_bytes(b'fixture')
    monkeypatch.setenv('VLLM_USE_FLASHINFER_SAMPLER','1')
    env=child_environment({'vllm_runtime':{'cache_root':str(tmp_path/'cache'),
        'cuda_compat':str(compat),'python':str(tmp_path/'venv/bin/python')}},0)
    assert env['VLLM_USE_FLASHINFER_SAMPLER']=='0'
    assert os.environ['VLLM_USE_FLASHINFER_SAMPLER']=='1'


def test_engine_startup_does_not_prepend_source_directories(tmp_path):
    from mechet.route_vllm_environment import child_environment
    compat=tmp_path/'compat';compat.mkdir();(compat/'libcuda.so.1').write_bytes(b'fixture')
    boot=tmp_path/'bootstrap';boot.mkdir()
    env=child_environment({'vllm_runtime':{'cache_root':str(tmp_path/'cache'),
        'cuda_compat':str(compat),'python':str(tmp_path/'venv/bin/python'),'bootstrap':str(boot)}},0)
    assert env['PYTHONPATH']==str(boot)
    assert env['PYTHONSAFEPATH']=='1'


def test_engine_persists_triton_autotuning_results(tmp_path,monkeypatch):
    from mechet.route_vllm_environment import child_environment
    compat=tmp_path/'compat';compat.mkdir();(compat/'libcuda.so.1').write_bytes(b'fixture')
    monkeypatch.setenv('TRITON_CACHE_AUTOTUNING','0')
    env=child_environment({'vllm_runtime':{'cache_root':str(tmp_path/'cache'),
        'cuda_compat':str(compat),'python':str(tmp_path/'venv/bin/python')}},0)
    assert env['TRITON_CACHE_AUTOTUNING']=='1'
    assert os.environ['TRITON_CACHE_AUTOTUNING']=='0'


def test_shared_triton_cache_reuses_verified_rank0_without_mixing_other_caches(tmp_path,monkeypatch):
    from mechet.route_vllm_environment import child_environment
    compat=tmp_path/'compat';compat.mkdir();(compat/'libcuda.so.1').write_bytes(b'fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    root=tmp_path/'cache';cfg={'vllm_runtime':{'cache_root':str(root),
        'cuda_compat':str(compat),'python':str(tmp_path/'venv/bin/python')}}
    original=child_environment(cfg,0)
    cfg['vllm_runtime']['shared_triton_cache_root']=original['TRITON_CACHE_DIR']
    assert child_environment(cfg,0)==original
    second=child_environment(cfg,1)
    assert second['TRITON_CACHE_DIR']==original['TRITON_CACHE_DIR']
    assert second['CUDA_VISIBLE_DEVICES']=='1'
    assert second['XDG_CACHE_HOME']!=original['XDG_CACHE_HOME']
    assert second['VLLM_RPC_BASE_PATH']!=original['VLLM_RPC_BASE_PATH']


def test_shared_triton_cache_cannot_write_outside_owned_runtime_root(tmp_path):
    from mechet.route_vllm_environment import child_environment
    compat=tmp_path/'compat';compat.mkdir();(compat/'libcuda.so.1').write_bytes(b'fixture')
    outside=tmp_path/'unrelated';link=tmp_path/'cache_link';link.symlink_to(outside)
    with pytest.raises(ValueError,match='TRITON_SHARED_CACHE_OUTSIDE_RUNTIME_ROOT'):
        child_environment({'vllm_runtime':{'cache_root':str(tmp_path/'cache'),
            'cuda_compat':str(compat),'python':str(tmp_path/'venv/bin/python'),
            'shared_triton_cache_root':str(link)}},0)
    assert not outside.exists()


@pytest.mark.parametrize('full,expected',[(False,'0'),(True,'1')])
def test_only_full_engine_child_skips_intermediate_ir_persistence(tmp_path,monkeypatch,full,expected):
    import subprocess,sys
    from mechet.route_vllm_environment import child_environment
    compat=tmp_path/'compat';compat.mkdir();(compat/'libcuda.so.1').write_bytes(b'fixture')
    monkeypatch.setenv('TRITON_STORE_BINARY_ONLY','0')
    cfg={'vllm_runtime':{'cache_root':str(tmp_path/'cache'),'cuda_compat':str(compat),
                         'python':str(tmp_path/'venv/bin/python')}}
    if full:cfg['acceleration']={'version':'pr111_full_acceleration_v2'}
    env=child_environment(cfg,0)
    actual=subprocess.check_output([sys.executable,'-S','-c',
        'import os;print(os.environ["TRITON_STORE_BINARY_ONLY"])'],env=env,text=True).strip()
    assert actual==expected
    assert os.environ['TRITON_STORE_BINARY_ONLY']=='0'
