import hashlib
import os
import pickle
from dataclasses import dataclass

from retflow.experiment import Experiment
from retflow.methods.method import Method
from retflow.utils.eval_helper import process_data_compute_metrics


@dataclass
class ExperimentEvaluator:
    experiment: Experiment
    test_method: Method
    examples_per_sample: int
    checkpoint_name: str | None = None
    output_name: str | None = None
    seed: int | None = None

    def run(self, on_valid: bool = False, compute_round_trip=False):
        if not isinstance(self.test_method, type(self.experiment.problem.method)):
            raise ValueError(
                "The test method must be the same method used to train in the experiment."
            )

        if self.seed is not None:
            self.experiment._set_seed(self.seed)
        else:
            self.experiment.set_seed()

        if self.checkpoint_name:
            model_checkpoint = self.experiment.save_directory() / self.checkpoint_name
        else:
            model_checkpoint = self.experiment.save_directory() / "final_model.pt"

        if not os.path.exists(self.experiment.save_directory()):
            os.makedirs(self.experiment.save_directory())

        if not model_checkpoint.is_file():
            raise Exception(
                "Model checkpoint does not exist. "
                "Provide a checkpoint name if the experiment has not finished training."
                "Double check checkpoint name."
            )

        self.experiment.problem.dataset.batch_size = 1
        self.experiment.problem.method = self.test_method
        self.experiment.problem.setup_problem_eval(model_checkpoint, on_valid)

        output_data = self.experiment.problem.sample_generation_eval(
            self.examples_per_sample
        )
        out_path = self.save_file_path(on_valid)
        with open(out_path, "wb") as file:
            pickle.dump(output_data, file)

        process_data_compute_metrics(
            out_path, self.examples_per_sample, compute_round_trip
        )

    def save_file_path(self, on_valid: bool = False):
        if self.experiment.name:
            self.experiment.save_directory() / f"{self.experiment.name}.pickle"
        output_data_name = self.output_name if self.output_name else self.eval_id()
        if on_valid:
            output_data_name += "_valid"
        return self.experiment.save_directory() / f"{output_data_name}.pickle"

    def eval_id(self):
        return hashlib.sha1(str.encode(str(self))).hexdigest()
