from retflow import (ADAMW, Experiment, GraphDiscreteFM, GraphTransformer,
                     RetroDataset, Retrosynthesis)
from retflow.runner import cli_runner, slurm_config

model = GraphTransformer()
dataset = RetroDataset(name="USPTO", batch_size=32)
method = GraphDiscreteFM()

experiment = Experiment(
    problem=Retrosynthesis(model, dataset, method),
    optim=ADAMW,
    epochs=300,
    sample_epoch=25,
    num_samples=128,
    examples_per_sample=5,
    seed=42,
    group="discrete_fm_product",
    name="baseline",
)

if __name__ == "__main__":
    cli_runner([experiment], slurm_config.DEFAULT_3_GPU_46H)
