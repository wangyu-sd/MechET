"""Lossless PEFT text-model name bridge to vLLM's multimodal wrapper."""
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil
import struct
import tempfile


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda:reader.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def verify_hashes(bindings):
    """Verify every immutable input, overlapping independent shared-storage IO."""
    def check(item):
        path,wanted=item
        if sha256(path)!=wanted:raise ValueError('BOUND_FILE_HASH:'+str(path))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for _ in pool.map(check,bindings.items()):pass


def prepare_text_adapter(source,destination,expected_sha):
    """Rewrite safetensors header only; preserve the entire tensor payload."""
    source,destination=Path(source),Path(destination)
    weights=source/'adapter_model.safetensors';config=source/'adapter_config.json'
    if sha256(weights)!=expected_sha:raise ValueError('VLLM_ADAPTER_SOURCE_HASH_CHANGED')
    binding=dict(source_sha256=expected_sha,source_config_sha256=sha256(config),
                 mapping='peft_text_to_qwen35_wrapper_v1')
    if destination.exists():
        receipt=json.loads((destination/'mapping.json').read_text())
        if any(receipt.get(k)!=v for k,v in binding.items()):raise ValueError('VLLM_ADAPTER_BINDING_CHANGED')
        if (sha256(destination/'adapter_model.safetensors')!=receipt['mapped_sha256']
                or sha256(destination/'adapter_config.json')!=binding['source_config_sha256']):
            raise ValueError('VLLM_MAPPED_ADAPTER_HASH_CHANGED')
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=destination.name+'.pending-',dir=destination.parent))
    try:
        with weights.open('rb') as reader:
            size=struct.unpack('<Q',reader.read(8))[0]
            if not 0<size<16*1024*1024:raise ValueError('VLLM_ADAPTER_HEADER_INVALID')
            header=json.loads(reader.read(size));mapped={};names={}
            for name,metadata in header.items():
                if name=='__metadata__':mapped[name]=metadata;continue
                prefix='base_model.model.model.layers.'
                if not name.startswith(prefix) or not name.endswith(('.lora_A.weight','.lora_B.weight')):
                    raise ValueError('VLLM_ADAPTER_UNRECOGNIZED_TENSOR:'+name)
                new=name.replace(prefix,'base_model.model.model.language_model.layers.',1)
                names[name]=new;mapped[new]=metadata
            if not names:raise ValueError('VLLM_ADAPTER_EMPTY')
            encoded=json.dumps(mapped,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
            with (stage/'adapter_model.safetensors').open('wb') as sink:
                sink.write(struct.pack('<Q',len(encoded)));sink.write(encoded)
                shutil.copyfileobj(reader,sink,8*1024*1024);sink.flush();os.fsync(sink.fileno())
        shutil.copyfile(config,stage/'adapter_config.json')
        receipt=dict(binding,tensor_names=names,tensor_count=len(names),
                     mapped_sha256=sha256(stage/'adapter_model.safetensors'))
        with (stage/'mapping.json').open('w') as sink:
            json.dump(receipt,sink);sink.flush();os.fsync(sink.fileno())
        os.rename(stage,destination)
    except Exception:
        shutil.rmtree(stage);raise
    return destination


def active_buffer(module,is_a,active_index,packed_slot,shape):
    stacks=module.lora_a_stacked if is_a else module.lora_b_stacked
    part=0 if packed_slot is None else packed_slot
    if isinstance(stacks,(tuple,list)):
        if part>=len(stacks):raise ValueError('VLLM_ACTIVE_BUFFER_PACKED_SLICE')
        stack=stacks[part]
    else:
        if part:raise ValueError('VLLM_ACTIVE_BUFFER_PACKED_SLICE')
        stack=stacks
    value=stack[active_index]
    if value.ndim==3 and value.shape[0]==1:value=value[0]
    if value.ndim!=2 or any(a<b for a,b in zip(value.shape,shape)):
        raise ValueError('VLLM_ACTIVE_BUFFER_SHAPE')
    return value[:shape[0],:shape[1]]


def expanded_registered_weights(module,weights):
    """Use the installed layer's native packed-group expansion at TP1."""
    if getattr(module,'tp_size',1)!=1:raise ValueError('VLLM_GPU_AUDIT_REQUIRES_TP1')
    a,b=weights.lora_a,weights.lora_b
    stacks=module.lora_b_stacked
    slices=len(stacks) if isinstance(stacks,(tuple,list)) else 1
    if not isinstance(b,(tuple,list)):
        if slices==1:return [a],[b]
        sizes=module.base_layer.output_sizes
        if len(sizes)!=slices or sum(sizes)!=b.shape[0]:raise ValueError('VLLM_GPU_AUDIT_PACKED_SHAPE')
        return [a]*slices,list(b.split(sizes,dim=0))
    a,b=list(a),list(b)
    if len(b)!=slices:a,b=module.expand_packed_lora(a,b)
    if len(a)!=slices or len(b)!=slices:raise ValueError('VLLM_GPU_AUDIT_PACKED_COUNT')
    return a,b


class RouteVllmAuditWorker:
    """vLLM worker extension: reject missing modules and mismatched loaded A/B."""
    def verify_route_lora(self,adapter_id,path):
        import torch
        from safetensors.torch import load_file
        from vllm.lora.utils import parse_fine_tuned_lora_name
        manager=self.model_runner.lora_manager._adapter_manager
        loaded=manager.get_adapter(adapter_id)
        if loaded is None:raise ValueError('VLLM_ADAPTER_NOT_REGISTERED')
        try:active_index=manager.lora_index_to_id.index(adapter_id)
        except ValueError as exc:raise ValueError('VLLM_ADAPTER_NOT_ACTIVE') from exc
        config=json.loads((Path(path)/'adapter_config.json').read_text())
        if any(config.get(k) for k in ('use_dora','use_rslora','rank_pattern','alpha_pattern')):
            raise ValueError('VLLM_ADAPTER_SCALING_UNSUPPORTED')
        scale=config['lora_alpha']/config['r']
        mapper=getattr(manager.model,'hf_to_vllm_mapper',None)
        count=0;modules=set();targets=set();dtypes=set();max_cast_error=0.
        for name,tensor in load_file(str(Path(path)/'adapter_model.safetensors')).items():
            module,is_a=parse_fine_tuned_lora_name(name,mapper)
            packed=None;slot=None
            if module in manager.modules:target=module
            else:
                matches=[(target,parts.index(module)) for target,parts in manager.packed_modules.items()
                         if module in parts and target in manager.modules]
                if len(matches)!=1:raise ValueError('VLLM_ADAPTER_MODULE_NOT_WRAPPED:'+module)
                target,slot=matches[0];packed=True
            weights=loaded.loras.get(target)
            if weights is None:raise ValueError('VLLM_ADAPTER_MODULE_NOT_LOADED:'+target)
            actual=weights.lora_a if is_a else weights.lora_b
            scaling=weights.scaling
            if packed:actual=actual[slot];scaling=scaling[slot]
            if actual is None:raise ValueError('VLLM_ADAPTER_TENSOR_MISSING:'+name)
            expected=tensor.to(dtype=actual.dtype,device=actual.device)
            if not is_a:expected=expected*scale;actual=actual*scaling
            if actual.shape!=expected.shape or not torch.equal(actual,expected):
                raise ValueError('VLLM_ADAPTER_TENSOR_MISMATCH:'+name)
            registered=weights.lora_a if is_a else weights.lora_b
            if packed:registered=registered[slot]
            master=tensor.float()*(1 if is_a else scale)
            effective=actual.float().cpu()
            max_cast_error=max(max_cast_error,float((master-effective).abs().max()))
            dtypes.add((str(tensor.dtype),str(registered.dtype)))
            targets.add(target)
            count+=1;modules.add(module)
        physical=0;gpu_dtypes=set()
        for target in sorted(targets):
            layer=manager.modules[target]
            a,b=expanded_registered_weights(layer,loaded.loras[target])
            for is_a,parts in ((True,a),(False,b)):
                for part,registered in enumerate(parts):
                    if registered is None:raise ValueError('VLLM_GPU_AUDIT_MISSING_PACKED_PART:'+target)
                    gpu=active_buffer(layer,is_a,active_index,part,registered.shape)
                    if gpu.device.type!='cuda':raise ValueError('VLLM_ACTIVE_BUFFER_NOT_CUDA:'+target)
                    if not torch.equal(gpu,registered.to(device=gpu.device,dtype=gpu.dtype)):
                        raise ValueError('VLLM_ACTIVE_BUFFER_MISMATCH:'+target+':'+str(part))
                    physical+=1;gpu_dtypes.add(str(gpu.dtype))
        mapping=json.loads((Path(path)/'mapping.json').read_text())
        if count!=mapping['tensor_count']:raise ValueError('VLLM_ADAPTER_TENSOR_COUNT')
        return dict(tensor_count=count,module_count=len(modules),
                    audit_scope='registered_and_active_gpu_after_explicit_dtype_cast',
                    active_slot=active_index,active_gpu_buffers=physical,
                    source_registered_dtypes=sorted(dtypes),active_gpu_dtypes=sorted(gpu_dtypes),
                    max_cast_abs_error=max_cast_error,
                    source_sha256=mapping['source_sha256'],mapped_sha256=mapping['mapped_sha256'])
