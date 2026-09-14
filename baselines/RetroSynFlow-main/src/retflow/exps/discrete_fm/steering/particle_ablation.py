from retflow.experiment_eval import ExperimentEvaluator
from retflow.exps.discrete_fm.product.product_baseline import experiment
from retflow.methods import (FKSteeringDiscreteFM, ForwardSynthesisReward,
                             LinearTimeScheduler, UniformTimeSampler)
from retflow.runner import cli_runner, slurm_config

num_particles = [2, 4, 6, 8]

test_methods = [
    FKSteeringDiscreteFM(
        steps=50,
        edge_time_sched=LinearTimeScheduler(),
        node_time_sched=LinearTimeScheduler(),
        time_sampler=UniformTimeSampler(),
        edge_weight_loss=5.0,
        reward_fn=ForwardSynthesisReward(n_best=1),
        resample_freq=10,
        num_particles=num,
        initial_temperature=1.0,
        lmbda=2.0,
    )
    for num in num_particles
]

experiments_evals = [
    ExperimentEvaluator(
        experiment=experiment,
        test_method=test_method,
        examples_per_sample=50,
        checkpoint_name="model_epoch_300.pt",
        output_name=f"fk_steering_k={test_method.num_particles}_lmbda=2.0_50_reactants",
    )
    for test_method in test_methods
]

if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_110H)
