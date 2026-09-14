from retflow.experiment_eval import ExperimentEvaluator
from retflow.exps.markov_bridge.baseline import experiment
from retflow.methods import GraphMarkovBridge
from retflow.runner import cli_runner, slurm_config

steps = [5, 10, 20, 25, 50, 100]

test_methods = [GraphMarkovBridge(steps=step) for step in steps]

experiments_evals = [
    ExperimentEvaluator(
        experiment=experiment,
        test_method=test_method,
        test_batch_size=256,
        examples_per_sample=50,
        checkpoint_name="final_model.pt",
        output_name=f"default_vlb_{test_method.steps}_steps",
    )
    for test_method in test_methods
]


if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_32H)
