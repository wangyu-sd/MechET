from retflow.experiment_eval import ExperimentEvaluator
from retflow.exps.discrete_fm.product.product_baseline import experiment
from retflow.methods import (FKSteeringDiscreteFM, ForwardSynthesisReward,
                             LinearTimeScheduler, UniformTimeSampler)
from retflow.runner import cli_runner, slurm_config

test_method = FKSteeringDiscreteFM(
    steps=50,
    edge_time_sched=LinearTimeScheduler(),
    node_time_sched=LinearTimeScheduler(),
    time_sampler=UniformTimeSampler(),
    edge_weight_loss=5.0,
    reward_fn=ForwardSynthesisReward(n_best=1),
    resample_freq=10,
    num_particles=4,
    initial_temperature=1.2,
    lmbda=2.0,
)

experiments_evals = [
    ExperimentEvaluator(
        experiment=experiment,
        test_method=test_method,
        examples_per_sample=100,
        checkpoint_name="model_epoch_300.pt",
        output_name=f"fk_steering_k=4_lmbda=2.0",
    )
]

if __name__ == "__main__":
    cli_runner(experiments_evals, slurm_config.DEFAULT_GPU_72H)
