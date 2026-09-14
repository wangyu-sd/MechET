from retflow import (ADAMW, Experiment, ExperimentEvaluator, GraphDiscreteFM,
                     GraphTransformer, SynthonRetrosynthesis,
                     TorchDrugRetroDataset)
from retflow.runner import cli_runner, slurm_config

model = GraphTransformer()

dataset = TorchDrugRetroDataset(
    name="DrugUSPTO", batch_size=32 * 3, product_context=True
)
method = GraphDiscreteFM(steps=100)

problem=SynthonRetrosynthesis(
    model,
    dataset,
    method,
    synthon_topk=2,
    samples_per_synthon=[70, 30],
    product_context=True,
)

experiment = Experiment(
    problem=problem,
    optim=ADAMW,
    epochs=600,
    sample_epoch=100,
    num_samples=128,
    examples_per_sample=5,
    seed=42,
    group="discrete_fm_synthon_completion",
    name="product_context",
)

test_method = GraphDiscreteFM(steps=100)

experiment_eval = ExperimentEvaluator(
    experiment=experiment,
    test_method=test_method,
    examples_per_sample=100,
    checkpoint_name="model_epoch_400.pt",
    output_name=f"retro_k={experiment.problem.synthon_topk}_70_30_full",
)


if __name__ == "__main__":
    cli_runner([experiment_eval], slurm_config.DEFAULT_GPU_32H)
