from dataclasses import dataclass
from typing import Optional


@dataclass
class SlurmConfig:
    """Configuration for Slurm.

    Args:
        time: Time limit for the job in format "1-23:45" for 1 day, 23h, 45m
        mem: Requested memory limit for the job in format "12000M" for 12G
        cpus: Number of CPUs per task
        gpu: Whether to use a GPU, or the name of a specific GPU.
    """

    time: str
    mem: str
    cpus: int
    gpu: Optional[bool | str] = False
    n_gpu: Optional[int] = 1
    nodes: Optional[int] = 1
    gpu_mem: Optional[int] = 32

    def __post_init__(self):
        # TODO: Add sanity checks for gpu, mem, time, cpus_per_task
        pass


SMALL_GPU_TEST = SlurmConfig(time="0-0:30", mem="8000M", cpus=6, gpu="p100")
DEFAULT_GPU_TEST = SlurmConfig(time="0-2:00", mem="8000M", cpus=6, gpu="v100l")
SMALL_2_GPU_TEST = SlurmConfig(time="0-0:30", mem="8000M", cpus=6, gpu="p100", n_gpu=2)
DEFAULT_2_GPU_TEST = SlurmConfig(
    time="0-0:20", mem="8000M", cpus=6, gpu="v100l", n_gpu=2
)
DEFAULT_3_GPU_TEST = SlurmConfig(
    time="0-0:20", mem="8000M", cpus=6, gpu="v100l", n_gpu=3
)


SMALL_2_GPU_48H = SlurmConfig(time="0-48:00", mem="32G", cpus=6, gpu="p100", n_gpu=2)
SMALL_4_GPU_32H = SlurmConfig(time="0-32:00", mem="32G", cpus=6, gpu="p100", n_gpu=2)

DEFAULT_GPU_32H = SlurmConfig(time="0-32:00", mem="32G", cpus=6, gpu="v100l")
DEFAULT_GPU_72H = SlurmConfig(time="0-72:00", mem="32G", cpus=6, gpu="v100l")
DEFAULT_GPU_90H = SlurmConfig(time="0-90:00", mem="32G", cpus=6, gpu="v100l")
DEFAULT_GPU_110H = SlurmConfig(time="0-110:00", mem="32G", cpus=6, gpu="v100l")
DEFAULT_3_GPU_26H = SlurmConfig(time="0-26:00", mem="32G", cpus=6, gpu="v100l", n_gpu=3)
DEFAULT_3_GPU_32H = SlurmConfig(time="0-32:00", mem="32G", cpus=6, gpu="v100l", n_gpu=3)
DEFAULT_3_GPU_36H = SlurmConfig(time="0-36:00", mem="32G", cpus=6, gpu="v100l", n_gpu=3)
DEFAULT_3_GPU_46H = SlurmConfig(time="0-46:00", mem="32G", cpus=6, gpu="v100l", n_gpu=3)
DEFAULT_4_GPU_24H = SlurmConfig(time="0-24:00", mem="32G", cpus=6, gpu="v100l", n_gpu=4)
DEFAULT_4_GPU_72H = SlurmConfig(time="0-72:00", mem="32G", cpus=6, gpu="v100l", n_gpu=4)
