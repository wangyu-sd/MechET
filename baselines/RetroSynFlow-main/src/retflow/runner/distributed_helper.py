import logging
import os
from datetime import timedelta

import torch
from torch import distributed as dist
from torch import nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from retflow import config


class DistributedHelper(object):
    def __init__(self, ddp_gpu_ids, init_method):
        self.ddp_gpu_ids = [int(id) for id in ddp_gpu_ids.split(",")]
        self.init_method = init_method

        if ddp_gpu_ids is None:
            assert (
                torch.cuda.device_count() > 1
            ), "Number of GPU must be more than one to use distributed learning!"

        self.gpu_name = "dummy"
        self.init_ddp()
        self.device = config.get_device()

    def init_ddp(self):
        """
        Initialize DDP distributed training if necessary.
        Note: we have to initialize DDP mode before initialize the logging file, otherwise the multiple DDP
        processes' loggings will interfere with each other.
        """
        print("Number of available GPU to use: {}".format(torch.cuda.device_count()))
        self.init_ddp_backend()
        self.gpu_name = torch.cuda.get_device_name()
        print(
            "Setup DDP for process {:d} using GPUs {} (ID) with NCCL backend. GPU for this process: {:s}".format(
                os.getpid(), self.ddp_gpu_ids, self.gpu_name
            )
        )

    def init_ddp_backend(self):
        """
        Start DDP engine using NCCL backend.
        """
        ddp_status, env_dict = self.get_ddp_status()
        local_rank = env_dict["LOCAL_RANK"]

        if self.ddp_gpu_ids is not None:
            assert isinstance(self.ddp_gpu_ids, list)
            num_gpus = len(self.ddp_gpu_ids)
            gpu_id = int(self.ddp_gpu_ids[local_rank % num_gpus])
            torch.cuda.set_device(gpu_id)  # set single gpu device per process
        else:
            torch.cuda.set_device(local_rank)  # set single gpu device per process
        dist.init_process_group(
            backend="nccl",
            init_method=self.init_method,
            rank=env_dict["WORLD_RANK"],
            world_size=env_dict["WORLD_SIZE"],
            timeout=timedelta(hours=1)
        )

    def dist_adapt_model(self, model):
        """
        Setup distributed learning for network.
        """
        logging.info("Adapt the model for distributed training...")
        # DDP
        model = DDP(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=True,
        )  # single CUDA device per process
        logging.info(
            "Distributed ON. Mode: DDP. Backend: {:s}, Rank: {:d} / World size: {:d}. "
            "Current device: {}, spec: {}".format(
                dist.get_backend(),
                dist.get_rank(),
                dist.get_world_size(),
                torch.cuda.current_device(),
                self.gpu_name,
            )
        )

        return model

    def ddp_sync(self):
        if dist.is_initialized():
            dist.barrier()
        else:
            pass

    def clean_up(self):
        self.ddp_sync()
        if dist.is_initialized():
            dist.destroy_process_group()
        else:
            pass

    @staticmethod
    def get_rank():
        if dist.is_initialized():
            return dist.get_rank()

    @staticmethod
    def get_ddp_status():
        """
        Get DDP-related env. parameters.
        """
        if "LOCAL_RANK" in os.environ:
            # Environment variables set by torch.distributed.launch or torchrun
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            world_rank = int(os.environ["RANK"])
        elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            # Environment variables set by mpirun
            local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
            world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        else:
            raise NotImplementedError

        env_dict = {
            "MASTER_ADDR": os.environ["MASTER_ADDR"],
            "MASTER_PORT": os.environ["MASTER_PORT"],
            "LOCAL_RANK": local_rank,
            "WORLD_SIZE": world_size,
            "WORLD_RANK": world_rank,
        }
        ddp_status = "Process PID: {}. DDP setup: {} ".format(os.getpid(), env_dict)
        return ddp_status, env_dict


# Independent function helpers
def get_ddp_save_flag():
    """
    Return saving flag for DDP mode, only rank 0 process makes the output.
    """
    flag_save = True
    if dist.is_initialized():
        if dist.get_rank() != 0:
            flag_save = False
    return flag_save


def dist_save_model(data_to_save, to_save_path):
    """
    Wrapper to save based on DDP status (for main process only).
    """
    if get_ddp_save_flag():
        torch.save(data_to_save, to_save_path)
