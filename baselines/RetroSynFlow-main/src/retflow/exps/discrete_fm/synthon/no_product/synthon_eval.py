from retflow.experiment_eval import ExperimentEvaluator
from retflow.exps.discrete_fm.synthon.no_product.synthon import experiment
from retflow.methods import GraphDiscreteFM
from retflow.runner import cli_runner, slurm_config

test_method = GraphDiscreteFM()

experiments_evals = [
    ExperimentEvaluator(
        experiment=experiment,
        test_method=test_method,
        examples_per_sample=100,
        checkpoint_name="model_epoch_600.pt",
        output_name="no_product_100_reactants",
    )
]

if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_32H)
