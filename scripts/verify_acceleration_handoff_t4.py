"""Bind a passed T4 software admission to the final stopped task checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path


def run(args):
    from scripts.validate_pr111_t4 import ROOT,WORK,saved_inputs,sha,write
    from mechet.route_transformers_import import import_transformers_indexed
    original=json.loads(args.admission.read_text())
    if original['status']!='PASS':raise ValueError('PASSED_PAIRED_ADMISSION_REQUIRED')
    for name,h in original['validated_compute_files_sha256'].items():
        if sha(WORK/name)!=h:raise ValueError('COMPUTE_CODE_CHANGED')
    cfg=json.loads((ROOT/'outputs/submissions/pr109_standard_d3_refit_20261116_02/contract.json').read_text())
    import_transformers_indexed(cfg)
    import torch
    from transformers.utils import import_utils
    import_utils.is_flash_linear_attention_available=lambda:False
    import_utils.is_causal_conv1d_available=lambda:False
    from transformers import AutoTokenizer,GenerationConfig,LogitsProcessorList
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
    for name in ('FusedRMSNormGated','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule',
                 'causal_conv1d_fn','causal_conv1d_update'):setattr(upstream,name,None)
    from peft import PeftModel,set_peft_model_state_dict
    from safetensors.torch import load_file
    from mechet.qwen35_protocol import load_language_model,OFFICIAL_REVISION
    from mechet.route_policy_sampling import TokenLogprobs,PrivateSampling
    from scripts.pr96_earho_contract import token_logps_batch
    torch.cuda.set_device(0);torch.set_num_threads(4)
    if 'T4' not in torch.cuda.get_device_name(0):raise ValueError('LOCAL_T4_REQUIRED')
    snapshots={}
    for task,path in {'d3':args.d3_contract,'d1':args.d1_contract}.items():
        source=json.loads(path.read_text());stop=json.loads((args.stop_dir/(task+'_source_stopped.json')).read_text())
        if stop['status']!='END' or stop['source_contract_sha256']!=sha(path):raise ValueError('STOP_CONTRACT_MISMATCH')
        pointer=json.loads((Path(source['output_dir'])/'latest.json').read_text());checkpoint=Path(pointer['path'])
        complete=json.loads((checkpoint/'complete.json').read_text())
        if sha(checkpoint/'complete.json')!=pointer['complete_sha256']:raise ValueError('FINAL_COMPLETE_CHANGED')
        expected=complete['payload_sha256']['adapter_model.safetensors']
        if sha(checkpoint/'adapter_model.safetensors')!=expected:raise ValueError('FINAL_ACTOR_CHANGED')
        snapshots[task]=dict(checkpoint=str(checkpoint),adapter_sha256=expected,source_complete_sha256=pointer['complete_sha256'],
            optimizer_step=complete['optimizer_step'],next_round=complete['next_round'])
    print(json.dumps(dict(stage='loading-final-stopped-actors',snapshots=snapshots)),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(cfg['base_path'],revision=OFFICIAL_REVISION,local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    actor,_=load_language_model(cfg['base_path'],dtype=torch.float32)
    actor=PeftModel.from_pretrained(actor,snapshots['d3']['checkpoint'],is_trainable=False).cuda().eval()
    prompts,_,_=saved_inputs(2);encoded={k:v.cuda() for k,v in tokenizer.pad(dict(input_ids=prompts),padding=True,return_tensors='pt').items()}
    width=encoded['input_ids'].shape[1];original['latest_checkpoint_legality']={}
    for task in ['d3','d1']:
        if task=='d1':set_peft_model_state_dict(actor,load_file(str(Path(snapshots[task]['checkpoint'])/'adapter_model.safetensors')),adapter_name='default')
        capture=TokenLogprobs(tokenizer.eos_token_id)
        processor=PrivateSampling([task+'/final/'+str(i) for i in range(2)],17,'cuda:0',finished=lambda:capture.done,prompt_width=width)
        generation=GenerationConfig(do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
        with torch.inference_mode():result=actor.generate(**encoded,generation_config=generation,max_new_tokens=256,logits_processor=LogitsProcessorList([capture,processor]))
        capture.finish(result[:,-1]);probabilities=capture.values();maximum=0.;complete_count=0
        for i,row in enumerate(result):
            ids=row[width:].tolist()
            if tokenizer.eos_token_id in ids:ids=ids[:ids.index(tokenizer.eos_token_id)+1];complete_count+=1
            with torch.no_grad():expected=token_logps_batch(actor,[(prompts[i],ids)],tokenizer.pad_token_id)[0]
            actual=torch.tensor(probabilities[i],device=expected.device)
            if actual.numel()!=len(ids) or not torch.isfinite(actual).all():raise ValueError('FINAL_ACTOR_PROBABILITY_INVALID')
            maximum=max(maximum,float((actual-expected).abs().max()))
        if maximum>.25 or complete_count!=2:raise ValueError('FINAL_ACTOR_LEGALITY_FAILED')
        record=dict(snapshots[task],status='PASS',requests=2,complete_outputs=complete_count,max_logprob_drift=maximum)
        original['latest_checkpoint_legality'][task]=record;original['legal_checks'][task+'_final_stopped_actor']='PASS'
        print(json.dumps(dict(task=task,**record)),flush=True)
    original['handoff_check_source_admission_sha256']=sha(args.admission)
    write(args.output,original)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['admission','d3-contract','d1-contract','stop-dir','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
