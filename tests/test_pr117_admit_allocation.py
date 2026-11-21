from importlib import import_module

import pytest


def test_existing_worker_handoff_requires_exact_running_instance_and_gpu():
    admission = import_module("scripts.pr117_admit_allocation")
    instance = "a" * 32
    ready = dict(hostname=f"worker-{instance}", gpu_model="NVIDIA A100-SXM4-80GB",
                 gpu_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    scheduler = f"| 2026-11-26 00:03:00 | {instance} | false | TRAINING_RUNNING | 0 | start |\n"
    probe = f"{ready['hostname']}\n{ready['gpu_uuid']}, {ready['gpu_model']}\n"

    assert admission.verify_worker_observation(instance, ready, scheduler, probe) == ready
    with pytest.raises(ValueError, match="running"):
        admission.verify_worker_observation(instance, ready, scheduler.replace("TRAINING_RUNNING", "END"), probe)
    with pytest.raises(ValueError, match="GPU identity"):
        admission.verify_worker_observation(instance, ready, scheduler, probe.replace("80GB", "40GB"))
