from retflow.experiment_eval import ExperimentEvaluator
from retflow.exps.discrete_fm.product.product_baseline import experiment
from retflow.runner import cli_runner, slurm_config

experiments_evals = [
    ExperimentEvaluator(
        experiment=experiment,
        test_method=experiment.problem.method,
        examples_per_sample=100,
        checkpoint_name="model_epoch_300.pt",
        output_name=f"linear_100_reactants",
    )
]

if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_32H)
