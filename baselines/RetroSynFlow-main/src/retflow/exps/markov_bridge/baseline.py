from retflow import Experiment
from retflow.datasets import RetroDataset
from retflow.methods import GraphMarkovBridge
from retflow.models import GraphTransformer
from retflow.optimizers.optimizer import AdamW
from retflow.optimizers.schedulers import ConsLR
from retflow.problems import Retrosynthesis
from retflow.runner import cli_runner, slurm_config

model = GraphTransformer()
dataset = RetroDataset(name="USPTO", batch_size=64)
method = GraphMarkovBridge()

experiment = Experiment(
    problem=Retrosynthesis(model, dataset, method),
    optim=AdamW(lr=2e-4, lr_sched=ConsLR()),
    epochs=1000,
    sample_epoch=100,
    num_samples=128,
    examples_per_sample=5,
    seed=42,
    group="markov_bridge",
    name="vlb",
)


if __name__ == "__main__":
    cli_runner([experiment], slurm_config.DEFAULT_GPU_110H)
