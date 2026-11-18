"""Recheck every original failed native receipt; no favorable-draw selection."""
import json
import math
import os
from pathlib import Path


class ScoreWitnesses:
    """Capture exact teacher outputs without replacing or modifying their values."""
    def __init__(self,module,path):
        self.module,self.path=module,Path(path);self.primary_model=None;self.pad_token_id=None
        self.original=module.token_logps_batch

    def __enter__(self):
        def observed(model,sequences,pad_token_id=0,**kwargs):
            values=self.original(model,sequences,pad_token_id,**kwargs)
            if self.primary_model is None:self.primary_model=model;self.pad_token_id=pad_token_id
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a') as sink:
                for (prompt,response),value in zip(sequences,values):
                    array=value.tolist() if hasattr(value,'tolist') else list(value)
                    sink.write(json.dumps(dict(prompt_token_ids=prompt,generated_token_ids=response,
                        teacher_logprobs=array,precision=kwargs.get('precision','legacy_fp32_lora')))+'\n')
                sink.flush();os.fsync(sink.fileno())
            return values
        self.module.token_logps_batch=observed
        return self

    def __exit__(self,*_):self.module.token_logps_batch=self.original


def replay_failed_probability_receipts(report_path,values_path,model,pad,precision,score,*,expected_actor,tolerance=.25):
    report=json.loads(Path(report_path).read_text());rows=json.loads(Path(values_path).read_text())
    if (report.get('status')!='FAIL' or
            report.get('source_bindings',{}).get('first',{}).get('adapter_sha256')!=expected_actor):
        raise ValueError('PREVIOUS_FAILURE_ACTOR_BINDING_CHANGED')
    if not rows or len({r['request_id'] for r in rows})!=len(rows):raise ValueError('PREVIOUS_FAILURE_RECEIPTS_INVALID')
    for row in rows:
        ids=row['generated_token_ids'];lp=row['behavior_logprobs']
        if not row['prompt_token_ids'] or not ids or len(ids)!=len(lp) or any(not math.isfinite(x) or x>0 for x in lp):
            raise ValueError('PREVIOUS_FAILURE_RECEIPTS_INVALID')
    maxima={};witnesses={}
    for size in (1,4):
        errors=[];witness=[]
        for start in range(0,len(rows),size):
            batch=rows[start:start+size]
            values=score(model,[(r['prompt_token_ids'],r['generated_token_ids']) for r in batch],pad,precision=precision)
            if len(values)!=len(batch):raise ValueError('PREVIOUS_FAILURE_SCORE_DENOMINATOR')
            for row,value in zip(batch,values):
                actual=value.tolist() if hasattr(value,'tolist') else list(value)
                if len(actual)!=len(row['generated_token_ids']) or any(not math.isfinite(x) or x>0 for x in actual):
                    raise ValueError('PREVIOUS_FAILURE_SCORE_INVALID')
                errors.extend(abs(a-b) for a,b in zip(actual,row['behavior_logprobs']))
                witness.append(dict(request_id=row['request_id'],teacher_logprobs=actual))
        maxima['max_error_b'+str(size)]=max(errors);witnesses['b'+str(size)]=witness
        if max(errors)>tolerance:raise ValueError('PREVIOUS_FAILED_RECEIPT_PARITY_UNCHANGED_025_GATE:b'+str(size))
    return dict(status='PASS',receipts=len(rows),positions_per_pass=sum(len(r['generated_token_ids']) for r in rows),
        tolerance=tolerance,**maxima,witnesses=witnesses)
