from retflow.experiment_eval import ExperimentEvaluator
from retflow.exps.discrete_fm.synthon.synthon_completion import experiment
from retflow.methods import GraphDiscreteFM
from retflow.runner import cli_runner, slurm_config

test_method = GraphDiscreteFM()

experiments_evals = [
    ExperimentEvaluator(
        experiment=experiment,
        test_method=test_method,
        examples_per_sample=100,
        checkpoint_name=f"model_epoch_400.pt",
        output_name=f"product_context_100_reactants",
    )
]

if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_32H)
