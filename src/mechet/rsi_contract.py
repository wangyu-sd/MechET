"""Immutable input identity and deterministic reaction-level RSI selection."""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path

BASE_REVISION='c1899de289a04d12100db370d81485cdf75e47ca'
R0_SHA='40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def stable_seed(*parts):
    return int(fingerprint(parts)[:12],16)%(2**31-1)


def authenticate_adapter(path,expected_sha=None):
    path=Path(path).resolve();weight=path/'adapter_model.safetensors'
    digest=sha256(weight)
    if expected_sha and digest!=expected_sha:raise ValueError('adapter SHA mismatch')
    cfg=json.loads((path/'adapter_config.json').read_text())
    if cfg['base_model_name_or_path']!='Qwen/Qwen3-0.6B' or cfg['r']!=16:
        raise ValueError('adapter base/rank mismatch')
    if set(cfg['target_modules'])!={'q_proj','k_proj','v_proj','o_proj'}:raise ValueError('LoRA modules mismatch')
    result={'path':str(path),'weight_sha256':digest,'adapter_config_sha256':sha256(path/'adapter_config.json'),
            'base_revision':BASE_REVISION,'lora':cfg,'status':'saved_adapter'}
    if path.name.startswith('checkpoint-'):
        state=json.loads((path/'trainer_state.json').read_text());step=int(path.name.split('-')[-1])
        if state['global_step']!=step or step<=0 or state['max_steps']<step:raise ValueError('checkpoint step mismatch')
        result.update(status='final_checkpoint' if step==state['max_steps'] else 'provisional_checkpoint',
                      global_step=step,max_steps=state['max_steps'],epoch=state.get('epoch'))
    result['tokenizer_sha256']={name:sha256(path/name) for name in ['tokenizer.json','tokenizer_config.json'] if (path/name).exists()}
    return result


def select_stratified(rows,count,seed):
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate reaction IDs')
    if count>len(rows) or count<1:raise ValueError('insufficient reaction denominator')
    bins=defaultdict(list)
    for row in rows:
        key=(min(int(row['event_count']),5),min(int(row['import_count']),3))
        bins[key].append(row)
    for values in bins.values():values.sort(key=lambda r:(stable_seed(seed,r['id']),r['id']))
    keys=sorted(bins);selected=[];position=0
    while len(selected)<count:
        for key in keys:
            if position<len(bins[key]):selected.append(bins[key][position])
            if len(selected)==count:break
        position+=1
    return selected


def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


class JsonJournal:
    """Durable append-only raw evidence, separate from uninterrupted stdout."""
    def __init__(self,path):
        import threading
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.file=self.path.open('a',buffering=1)
    def append(self,row):
        import os
        with self.lock:
            self.file.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
            self.file.flush();os.fsync(self.file.fileno())
    def close(self):self.file.close()
