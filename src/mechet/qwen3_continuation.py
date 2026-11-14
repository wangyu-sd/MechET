"""Separate, hash-bound Qwen3-8B condition; never weakens PR96 model guards."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from .qwen3_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION, PROTOCOL

ROOT = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
SHARED = Path('/aaa/fionafyang/buddy1')
PARENT = ROOT/'outputs/agent/natural_language_event_history_v2_qwen3_8b_h20_seed17_20260918/checkpoint-31366'
HISTORY = ROOT/'data/pr96_flower_mainonly_history_v2_20261023_06'
PARENT_SHA = 'a0f67c8ab799e5dad18ba97d824a7d4ae2a0280083eeeaca75c48ec903028c9b'
PARENT_CONFIG_SHA = 'cc017209968da49db43f898227f5f9e9502bd8107cf90e4c1e257fae999003ff'
TRAIN_SHA = '0ace674b6ebf6b4ba0a3542de8572c973e0d6a644a6923f651ffab74b2ab4748'
VALID_SHA = '0abb938fe0f0205a648ad7ff979fbe9ce217d1947f75ddac9997bb05e535b520'


def continuation_config(*, world_size=4):
    if world_size not in (4, 8):
        raise ValueError('QWEN3_FOUR_OR_EIGHT_H20_REQUIRED')
    output = ROOT/'outputs/agent/qwen3_8b_corrected_continue_4h20_20261113_01'
    cache = output/'tokens'
    if world_size == 8:
        output = ROOT/'outputs/agent/qwen3_8b_corrected_continue_8h20_20261113_01'
    return dict(condition_name=f'qwen3_8b_corrected_history_continue_{world_size}h20_v1',
        model_name_or_path=OFFICIAL_MODEL, initial_adapter_path=str(PARENT),
        train_file=str(ROOT/'outputs/agent/pr96_flower_qwen35_08b_mainonly_20261023_06/views/history/train.jsonl'),
        validation_file=str(HISTORY/'valid.jsonl'), output_dir=str(output),
        pretokenized_cache_dir=str(cache), pretokenization_world_size=4,
        limit_examples=0,
        qwen3_continuation=dict(world_size=world_size, max_gpus=world_size, parent_weight_sha256=PARENT_SHA,
            reference_approval=str(ROOT/'data/pr96_flower_reference_approval_20261023_06/reference_approval.json'),
            reference_approval_sha256='eb92320d9711e85bfcd8281cce8bc90cde496b4b47688405372809280d618ada',
            reference_manifest=str(ROOT/'data/pr96_flower_mainonly_source_arrows_v2_20261023_06/manifest.json'),
            reference_manifest_sha256='2a203ee08df9a6099f66d554797a72b93750b88cdfe760669431f2b665ff083c',
            history_manifest=str(HISTORY/'manifest.json'),
            history_manifest_sha256='edd299a832e7f15c53d134b74015097865737837d587c723db4527d2b9290f06',
            train_sha256=TRAIN_SHA, validation_sha256=VALID_SHA,
            policy_protocol=PROTOCOL, policy_model_family='qwen3',
            stage='corrected_single_reaction_continuation',
            memory_replay_status='separate_future_condition_not_silently_mixed'),
        contract=dict(require_trace_owned=False, require_tool_decision=True,
            expected_train_rows=1908035, expected_validation_rows=21355,
            expected_upstream_endpoint_fallback_rows=0,
            environment_revision='natural_language_electron_event_history_v2',
            executor_revision='pr95_main_product_only_actual_executor',
            stable_id_manifest=str(HISTORY/'manifest.json')),
        lora=dict(r=16, alpha=32, dropout=.05, target_modules=['q_proj','k_proj','v_proj','o_proj']),
        training=dict(model_revision=OFFICIAL_REVISION, trust_remote_code=False,
            num_train_epochs=1., max_steps=-1, max_length=32768, window_context_tokens=0,
            encoding_protocol=PROTOCOL, per_device_train_batch_size=1,
            gradient_accumulation_steps=64//world_size, per_device_eval_batch_size=1,
            learning_rate=5e-6, bf16=True, fp16=False, qlora=True, tf32=True,
            attention_implementation='sdpa', require_flash_sdp=True,
            gradient_checkpointing=True, assistant_only_loss=True, packing=False,
            group_by_length=True, validation_limit=0, eval_strategy='epoch',
            eval_accumulation_steps=1, dataloader_num_workers=4, logging_steps=10,
            save_steps=500, save_total_limit=3, seed=17, data_seed=17,
            use_liger_kernel=True, liger_kernel_config=dict(rope=True,rms_norm=True,
                swiglu=True,cross_entropy=False,fused_linear_cross_entropy=True),
            report_to=[]))


def shared_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(SHARED):
        raise ValueError('QWEN3_SHARED_STORAGE_REQUIRED:'+str(path))
    return path


def validate_config(cfg, *, world_size=None, gpu_names=None):
    declared = cfg.get('qwen3_continuation', {}).get('world_size', 4)
    expected = continuation_config(world_size=declared)
    for key in ('model_name_or_path', 'initial_adapter_path', 'train_file', 'validation_file',
                'limit_examples', 'qwen3_continuation', 'contract', 'lora', 'pretokenization_world_size'):
        if cfg.get(key) != expected[key]:
            raise ValueError('QWEN3_CONTINUATION_CONDITION_CHANGED:'+key)
    t = cfg['training']
    for key, value in expected['training'].items():
        if t.get(key) != value:
            raise ValueError('QWEN3_CONTINUATION_TRAINING_CHANGED:'+key)
    for key in ('output_dir', 'pretokenized_cache_dir', 'train_file', 'validation_file', 'initial_adapter_path'):
        shared_path(cfg[key])
    if world_size is not None and (world_size != declared or gpu_names is None or len(gpu_names) != declared
            or not all(re.search(r'\bH20\b', n, re.I) for n in gpu_names)):
        raise ValueError('QWEN3_EXACTLY_FOUR_H20_REQUIRED' if declared == 4 else 'QWEN3_EXACTLY_EIGHT_H20_REQUIRED')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_files(bindings):
    def verify(item):
        path, expected = item
        p = shared_path(path)
        before = p.stat()
        digest = _sha(p)
        after = p.stat()
        if digest != expected or (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise ValueError('QWEN3_FILE_BINDING_MISMATCH:'+str(p))
        return str(p), dict(sha256=digest, bytes=after.st_size,
                           mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns)
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(bindings)))) as pool:
        return dict(pool.map(verify, bindings.items()))


def validate_cache(cfg, cache):
    if (not isinstance(cache, dict) or cache.get('model_name_or_path') != OFFICIAL_MODEL
            or cache.get('model_revision') != OFFICIAL_REVISION
            or cache.get('encoding_protocol') != PROTOCOL or cache.get('world_size') != 4
            or cache.get('max_length') != cfg['training']['max_length']):
        raise ValueError('QWEN3_CACHE_MODEL_PROTOCOL_MISMATCH')
    for split, digest, count in [('train', TRAIN_SHA, 1908035), ('validation', VALID_SHA, 21355)]:
        item = cache['splits'][split]
        shards = [Path(p).name for p in item.get('arrow_files', [])]
        expected_shards = [f'{split}.rank{i:02d}.arrow' for i in range(4)]
        if (cache['sources'][split]['sha256'] != digest or item['n_rows'] != count
                or sorted(shards) != expected_shards or item.get('n_training_windows') != count
                or item.get('n_windowed_source_rows', 0) or item.get('truncation_count') != 0
                or item.get('min_supervised_tokens', 0) < 1
                or item.get('assistant_mask_methods') != ['qwen3_owned_tool_span_v1']):
            raise ValueError('QWEN3_CACHE_SOURCE_OR_MASK_MISMATCH:'+split)


def verify_arrow_counts(cfg, cache):
    import pyarrow as pa
    for split, item in cache['splits'].items():
        observed = 0
        for path in item['arrow_files']:
            p = shared_path(Path(cfg['pretokenized_cache_dir'])/Path(path).name)
            with pa.memory_map(str(p), 'r') as source:
                observed += sum(batch.num_rows for batch in pa.ipc.open_stream(source))
        if observed != item['n_rows']:
            raise ValueError('QWEN3_CACHE_ACTUAL_ROW_COUNT:'+split)


def adapter_identity(output):
    return _sha(shared_path(output)/'adapter_model.safetensors')


def verify_bindings(cfg, cache=None):
    """Hash actual inputs once under a shared lock, reuse only stat-identical files.

    All four ranks share the receipt. This avoids four parallel full Ceph scans;
    any input/config/cache metadata change invalidates the receipt and is rehashed.
    """
    import fcntl
    validate_config(cfg)
    b = cfg['qwen3_continuation']
    bindings = {cfg['train_file']: TRAIN_SHA, cfg['validation_file']: VALID_SHA,
                str(PARENT/'adapter_model.safetensors'): PARENT_SHA,
                str(PARENT/'adapter_config.json'): PARENT_CONFIG_SHA}
    for name in ('reference_approval', 'reference_manifest', 'history_manifest'):
        bindings[b[name]] = b[name+'_sha256']
    if cache is not None:
        validate_cache(cfg, cache)
        for item in cache['splits'].values():
            for file in item['arrow_files']:
                p = Path(cfg['pretokenized_cache_dir'])/Path(file).name
                bindings[str(p)] = item['arrow_sha256'][p.name]
    key = hashlib.sha256(json.dumps(dict(bindings=bindings, cache=cache),sort_keys=True).encode()).hexdigest()
    # verify_files records resolved Ceph paths. Keep the existing raw-path key
    # (so old valid receipts remain usable), but compare the same path namespace.
    resolved_bindings = {}
    for path, digest in bindings.items():
        canonical = str(shared_path(path))
        if canonical in resolved_bindings and resolved_bindings[canonical] != digest:
            raise ValueError('QWEN3_BINDING_ALIAS_CONFLICT')
        resolved_bindings[canonical] = digest
    directory = shared_path(cfg['output_dir'])/'runtime'
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/('input-verification-'+key+'.json')
    with (directory/'input-verification.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if (value.get('key') == key and set(value.get('files',{})) == set(resolved_bindings)
                    and value.get('cache_verified') == (cache is not None)):
                unchanged = True
                for path, record in value['files'].items():
                    st = Path(path).stat()
                    unchanged &= (record.get('sha256') == resolved_bindings[path]
                                  and (st.st_size,st.st_mtime_ns,st.st_ctime_ns) ==
                                  (record['bytes'],record['mtime_ns'],record['ctime_ns']))
                if unchanged:
                    return value
        files = verify_files(bindings)
        if cache is not None:
            verify_arrow_counts(cfg, cache)
        approval = json.loads(Path(b['reference_approval']).read_text())
        if (approval.get('reference_training_authorized') is not True
                or approval['reference_manifest_sha256'] != b['reference_manifest_sha256']
                or approval['history_manifest_sha256'] != b['history_manifest_sha256']):
            raise ValueError('QWEN3_CORRECTED_REFERENCE_APPROVAL')
        parent_cfg = json.loads((PARENT/'adapter_config.json').read_text())
        if parent_cfg.get('base_model_name_or_path') != OFFICIAL_MODEL or parent_cfg.get('r') != 16:
            raise ValueError('QWEN3_PARENT_ADAPTER_CONFIG')
        value = dict(key=key, files=files, cache_verified=cache is not None,
                     condition=cfg['condition_name'], gpu_training_verified=False)
        temporary = receipt.with_suffix('.partial')
        with temporary.open('w') as f:
            json.dump(value,f,sort_keys=True); f.flush()
            import os
            os.fsync(f.fileno())
        temporary.replace(receipt)
        return value


def _checkpoint_identity(cfg):
    validate_config(cfg)
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def save_checkpoint_binding(cfg, checkpoint):
    p = shared_path(checkpoint)
    names = ['adapter_model.safetensors','adapter_config.json','optimizer.pt',
             'scheduler.pt','trainer_state.json', *['rng_state_'+str(i)+'.pth' for i in range(cfg['qwen3_continuation']['world_size'])]]
    files = {name:_sha(p/name) for name in names}
    state = json.loads((p/'trainer_state.json').read_text())
    value = dict(schema='qwen3_corrected_continuation_checkpoint_v1',
                 identity=_checkpoint_identity(cfg), step=state['global_step'], files=files)
    target = p/'qwen3_continuation_manifest.json'
    temporary = target.with_suffix('.partial')
    with temporary.open('w') as f:
        import os
        json.dump(value,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
    temporary.replace(target)
    return value


def verify_resume(cfg, checkpoint):
    p = shared_path(checkpoint)
    value = json.loads((p/'qwen3_continuation_manifest.json').read_text())
    required = {'adapter_model.safetensors','adapter_config.json','optimizer.pt',
                'scheduler.pt','trainer_state.json', *{'rng_state_'+str(i)+'.pth' for i in range(cfg['qwen3_continuation']['world_size'])}}
    if (value.get('schema') != 'qwen3_corrected_continuation_checkpoint_v1'
            or value.get('identity') != _checkpoint_identity(cfg)
            or set(value.get('files',{})) != required):
        raise ValueError('QWEN3_CHECKPOINT_LINEAGE_MISMATCH')
    verify_files({str(p/name):digest for name,digest in value['files'].items()})
    if json.loads((p/'trainer_state.json').read_text())['global_step'] != value['step']:
        raise ValueError('QWEN3_CHECKPOINT_POSITION_MISMATCH')
    return value


def checkpoint_callback(cfg):
    from transformers import TrainerCallback
    class BindingCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            import torch.distributed as dist
            distributed = dist.is_available() and dist.is_initialized()
            if distributed:
                dist.barrier()  # Every rank's RNG file must be durable first.
            error = [None]
            if state.is_world_process_zero:
                try:
                    save_checkpoint_binding(cfg,Path(args.output_dir)/f'checkpoint-{state.global_step}')
                except Exception as exc:
                    error[0] = repr(exc)
            if distributed:
                dist.broadcast_object_list(error,src=0)
            if error[0]:
                raise ValueError('QWEN3_CHECKPOINT_BINDING_FAILED:'+error[0])
            return control
    return BindingCallback()
