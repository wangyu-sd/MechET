import json

import pytest


def test_queue_waits_for_actual_cache_and_refuses_wrong_model(tmp_path):
    from mechet.qwen3_queue import cache_readiness
    from mechet.qwen3_continuation import continuation_config
    cfg = continuation_config(world_size=8)
    cfg['pretokenized_cache_dir'] = str(tmp_path)
    assert cache_readiness(cfg) == 'waiting_for_cpu_cache'
    (tmp_path / 'manifest.json').write_text(json.dumps({'model_name_or_path': 'wrong'}))
    with pytest.raises(ValueError, match='CACHE'):
        cache_readiness(cfg)


def test_resource_manifest_cannot_exceed_eight_or_wait_inside_gpu(tmp_path):
    from mechet.qwen3_queue import validate_submission
    from mechet.qwen3_continuation import continuation_config
    cfg = continuation_config(world_size=8)
    task = dict(host_num=1, host_gpu_num=8, business_flag='AI4LSLab_KEMO_ZJK_H20',
                keep_running_after_trainer_finish=False)
    validate_submission(cfg, task)
    with pytest.raises(ValueError):
        validate_submission(cfg, dict(task, host_gpu_num=16))
    with pytest.raises(ValueError):
        validate_submission(cfg, dict(task, keep_running_after_trainer_finish=True))


def test_waiting_for_cache_cannot_erase_submission_intent():
    from mechet.qwen3_queue import waiting_status
    prior = dict(state='submit_intent', task_flag='one-eight-card-task')
    now = waiting_status(prior, task_flag=prior['task_flag'])
    assert now['state'] == 'submit_intent'
    assert now['readiness'] == 'waiting_for_cpu_cache'
    assert prior['state'] == 'submit_intent'


def test_submission_inputs_are_rechecked_after_wait(tmp_path):
    import hashlib
    from mechet.qwen3_queue import submission_inputs
    from mechet.qwen3_continuation import continuation_config
    cfg = tmp_path/'config.json'; task = tmp_path/'task.json'
    cfg.write_text(json.dumps(continuation_config(world_size=8)))
    task.write_text(json.dumps(dict(host_num=1, host_gpu_num=8, business_flag='AI4LSLab_KEMO_ZJK_H20', keep_running_after_trainer_finish=False)))
    plan = dict(training_config=str(cfg), task_config=str(task),
                files_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (cfg,task)})
    submission_inputs(plan)
    task.write_text(json.dumps(dict(host_num=1, host_gpu_num=16)))
    with pytest.raises(ValueError, match='BINDING'):
        submission_inputs(plan)
