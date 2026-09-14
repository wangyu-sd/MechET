from retflow.experiment_eval import ExperimentEvaluator
from retflow.exps.discrete_fm.product.product_baseline import experiment
from retflow.methods import GraphDiscreteFM
from retflow.runner import cli_runner, slurm_config

steps = [5, 10, 20, 25, 50, 100]

test_methods = [GraphDiscreteFM(steps=step) for step in steps]

experiments_evals = [
    ExperimentEvaluator(
        experiment=experiment,
        test_method=test_method,
        examples_per_sample=50,
        checkpoint_name="model_epoch_300.pt",
        output_name=f"linear_{test_method.steps}_steps_50_reactants",
    )
    for test_method in test_methods
]

if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_32H)
