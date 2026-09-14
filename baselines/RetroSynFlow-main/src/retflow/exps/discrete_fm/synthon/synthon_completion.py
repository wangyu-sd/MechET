from retflow import (ADAMW, Experiment, GraphDiscreteFM, GraphTransformer,
                     SynthonCompletion, SynthonDataset)
from retflow.runner import cli_runner, slurm_config

model = GraphTransformer()

dataset = SynthonDataset(name="SynthonUSPTO", batch_size=32 * 3)
method = GraphDiscreteFM()

experiment = Experiment(
    problem=SynthonCompletion(model, dataset, method, use_product_context=True),
    optim=ADAMW,
    epochs=500,
    sample_epoch=100,
    num_samples=128,
    examples_per_sample=5,
    seed=42,
    group="discrete_fm_synthon_completion",
    name="product_context",
)

if __name__ == "__main__":
    cli_runner([experiment], slurm_config.DEFAULT_3_GPU_46H)
