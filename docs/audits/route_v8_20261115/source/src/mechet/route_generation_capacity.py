"""Bound physical prefill batches; never truncate a history or drop a request."""

DEFAULT_QKV_CHANNELS=6144
DEFAULT_MAX_QKV_ELEMENTS=2**31-1
DEFAULT_ATTENTION_BUDGET_BYTES=12*1024**3


def capacity_kwargs(cfg):
    return dict(qkv_channels=cfg.get('generation_qkv_channels',DEFAULT_QKV_CHANNELS),
        max_qkv_elements=cfg.get('generation_max_qkv_elements',DEFAULT_MAX_QKV_ELEMENTS),
        attention_budget_bytes=cfg.get('generation_attention_budget_bytes',DEFAULT_ATTENTION_BUDGET_BYTES))


def validate_model_capacity(model,cfg):
    limits=capacity_kwargs(cfg)
    config=getattr(model,'config',None)
    names=('linear_num_key_heads','linear_key_head_dim','linear_num_value_heads','linear_value_head_dim')
    if config is not None and all(hasattr(config,name) for name in names):
        actual=2*config.linear_num_key_heads*config.linear_key_head_dim
        actual+=config.linear_num_value_heads*config.linear_value_head_dim
        if limits['qkv_channels']!=actual:raise ValueError('GENERATION_QKV_CHANNELS_MISMATCH')
    generation_batches([],max_batch_size=1,**limits)
    return limits


def generation_batches(lengths,*,max_batch_size,qkv_channels=DEFAULT_QKV_CHANNELS,
                       max_qkv_elements=DEFAULT_MAX_QKV_ELEMENTS,
                       attention_budget_bytes=DEFAULT_ATTENTION_BUDGET_BYTES):
    for value in (max_batch_size,qkv_channels,max_qkv_elements,attention_budget_bytes):
        if type(value) is not int or value<1:raise ValueError('INVALID_PREFILL_CAPACITY')
    if max_qkv_elements>=2**32:raise ValueError('UNSAFE_CONVOLUTION_INDEX_CAPACITY')
    def fits(size,width):
        # The conv kernel uses uint32 batch offsets. The default is deliberately
        # below the signed32 boundary too. SDPA can hold both FP32 and BF16 BTT
        # attention masks; reserve six bytes/cell within a separate workspace cap.
        return (size<=max_batch_size and size*width*qkv_channels<=max_qkv_elements
                and size*width*width*6<=attention_budget_bytes)
    batches=[];current=[];width=0
    for index,length in enumerate(lengths):
        if type(length) is not int or length<1:raise ValueError('INVALID_PROMPT_LENGTH')
        if not fits(1,length):raise ValueError('REQUEST_EXCEEDS_PREFILL_CAPACITY')
        if current and not fits(len(current)+1,max(width,length)):
            batches.append(current);current=[];width=0
        current.append(index);width=max(width,length)
    if current:batches.append(current)
    return batches
