from retflow import Experiment
from retflow.datasets import SynthonDataset
from retflow.methods import GraphDiscreteFM
from retflow.models import GraphTransformer
from retflow.optimizers.optimizer import AdamW
from retflow.optimizers.schedulers import ConsLR
from retflow.problems import SynthonCompletion
from retflow.runner import cli_runner, slurm_config

model = GraphTransformer()

dataset = SynthonDataset(name="SynthonUSPTO", batch_size=32 * 3)
method = GraphDiscreteFM()

experiment = Experiment(
    problem=SynthonCompletion(model, dataset, method),
    optim=AdamW(lr=2e-4, lr_sched=ConsLR()),
    epochs=800,
    sample_epoch=100,
    num_samples=128,
    examples_per_sample=5,
    seed=42,
    group="discrete_fm_synthon_completion",
    name="without_product_context",
)


if __name__ == "__main__":
    cli_runner([experiment], slurm_config.DEFAULT_3_GPU_46H)
