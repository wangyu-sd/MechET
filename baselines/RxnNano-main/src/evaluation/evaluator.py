# -*- coding: utf-8 -*-
from vllm import SamplingParams
from typing import Dict, List
import json
from datetime import datetime
from pathlib import Path
import time

from .metrics import (
    extract_xml_answer, extract_xml_think, compute_top_k_accuracy,
    normalize_smiles, parse_predicted_answer, deduplicate_and_rerank_with_scores,
    is_valid_smiles
)
from .prompt_formatters import get_prompt_formatter
from ..utils.logging import setup_logging, save_results_to_csv
from ..utils.smiles_utils import extract_smiles_from_prompt
from ..config.settings import EvaluationConfig

class Evaluator:
    def __init__(self, model, tokenizer, config: EvaluationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            n=config.generate_n,
            top_k=config.top_k,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            min_p=0.05,
            logprobs=1,
            seed=config.seed,
        )

    def evaluate_dataset(self, task: str, dataset, dataset_name: str, prompt_type: str, 
                        batch_size: int, logger=None):
        """Evaluate the model on a given dataset."""
        expected_key = 'reactants' if 'retrosynthesis' in task else 'product'
        predicted_key = expected_key

        total_samples = len(dataset)
        top_k_correct = {k: 0 for k in self.config.top_k_metrics_list}
        top_k_accuracies = {k: [] for k in self.config.top_k_metrics_list}
        prediction_rows = []

        print(f"Evaluating {task} on {total_samples} examples with generate_n={self.config.generate_n}, n={self.config.n}, top_k_metrics={self.config.top_k_metrics} (batch size: {batch_size})")
        if logger and not self.config.disable_logging_csv:
            logger.info(f"Evaluating {task} on {total_samples} examples with generate_n={self.config.generate_n}, n={self.config.n}, top_k_metrics={self.config.top_k_metrics} (batch size: {batch_size})")

        # Prepare prompts
        prompts = [
            self.tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True
            )
            for example in dataset
        ]
        expected_outputs = [example["expected"] for example in dataset]

        # Process in batches
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_prompts = prompts[batch_start:batch_end]
            batch_expected = expected_outputs[batch_start:batch_end]

            print(f"Processing {task} batch {batch_start // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}...")
            if logger and not self.config.disable_logging_csv:
                logger.info(f"Processing {task} batch {batch_start // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}...")

            batch_started = time.perf_counter()
            batch_outputs = self.model.generate(
                batch_prompts,
                sampling_params=self.sampling_params,
            )
            batch_runtime_ms = (time.perf_counter() - batch_started) * 1000.0

            self._process_batch_outputs(
                batch_outputs, batch_expected, batch_start, total_samples,
                task, expected_key, predicted_key, top_k_correct, top_k_accuracies,
                dataset, logger, prediction_rows,
                batch_runtime_ms / max(1, len(batch_outputs))
            )

        if self.config.predictions_output:
            output_path = Path(self.config.predictions_output)
            if len(self.config.selected_tasks) > 1:
                output_path = output_path.with_name(
                    f"{output_path.stem}.{task}{output_path.suffix or '.jsonl'}"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                for row in prediction_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Predictions saved to {output_path}")

        results = self._compute_final_metrics(
            task, dataset_name, prompt_type, total_samples, top_k_correct, logger
        )
        results["sampling"] = {
            "seed": self.config.seed,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "generate_n": self.config.generate_n,
            "retained_n": self.config.n,
            "max_tokens": self.config.max_tokens,
        }
        results["prediction_rows"] = len(prediction_rows)
        return results

    def _process_batch_outputs(self, batch_outputs, batch_expected, batch_start, total_samples,
                              task, expected_key, predicted_key, top_k_correct, top_k_accuracies,
                              dataset, logger, prediction_rows, runtime_ms_per_sample):
        """Process a batch of model outputs."""
        for i, output in enumerate(batch_outputs):
            sample_idx = batch_start + i + 1
            expected = batch_expected[i][expected_key]
            all_generations = output.outputs
            predicted_list = []
            scores = []
            best_output_text = ""
            best_gen_idx = -1
            raw_outputs = []

            for gen_idx, generation_output in enumerate(all_generations):
                output_text = generation_output.text
                raw_outputs.append(output_text)
                predicted_answer = extract_xml_answer(output_text)
                predicted = parse_predicted_answer(predicted_answer, predicted_key)

                # Calculate cumulative logprob for scoring
                score = 0.0
                if hasattr(generation_output, 'logprobs') and generation_output.logprobs:
                    try:
                        for token_logprobs in generation_output.logprobs:
                            if token_logprobs:
                                for token_id, logprob in token_logprobs.items():
                                    score += float(getattr(logprob, "logprob", logprob))
                                    break  # Assume first is the chosen token
                    except Exception as e:
                        if logger and not self.config.disable_logging_csv:
                            logger.warning(f"{task} Sample {sample_idx}, Gen {gen_idx + 1}: Error computing logprobs: {e}")

                if is_valid_smiles(predicted):
                    predicted_list.append(predicted)
                    scores.append(score)

                norm_predicted = normalize_smiles(predicted)
                norm_expected = normalize_smiles(expected)
                if norm_predicted == norm_expected:
                    best_output_text = output_text
                    best_gen_idx = gen_idx
                elif not best_output_text:
                    best_output_text = output_text
                    best_gen_idx = gen_idx
                elif not best_output_text and predicted_answer:
                    best_output_text = output_text
                    best_gen_idx = gen_idx

            if not best_output_text and raw_outputs:
                best_output_text = raw_outputs[0]
                best_gen_idx = 0

            # Deduplicate and rerank predictions
            ranked_predictions = []
            if predicted_list:
                ranked_predictions = deduplicate_and_rerank_with_scores(
                    predicted_list, scores, self.config.n
                )
                predicted_list = [pred for pred, _ in ranked_predictions]

            source_row = dataset[batch_start + i]
            prediction_rows.append({
                "stable_id": source_row.get("stable_id", str(sample_idx)),
                "product": source_row.get("product", ""),
                "reference_precursors": expected,
                "candidates": [
                    {"rank": rank, "precursors": pred, "score": score}
                    for rank, (pred, score) in enumerate(ranked_predictions, start=1)
                ],
                "runtime_ms": runtime_ms_per_sample,
                "source_method": "RxnNano",
                "checkpoint": self.config.model_name,
                "raw_generations": raw_outputs,
            })

            # Compute top-k accuracies
            for k in self.config.top_k_metrics_list:
                is_correct = compute_top_k_accuracy(expected, predicted_list, k)
                top_k_correct[k] += int(is_correct)
                top_k_accuracies[k].append(1.0 if is_correct else 0.0)

            # Log sample details if logging enabled
            if logger and not self.config.disable_logging_csv:
                self._log_sample_details(
                    logger, task, dataset, batch_start, i, sample_idx, total_samples,
                    expected, predicted_list, best_output_text, best_gen_idx, raw_outputs
                )

    def _log_sample_details(self, logger, task, dataset, batch_start, i, sample_idx, total_samples,
                           expected, predicted_list, best_output_text, best_gen_idx, raw_outputs):
        """Log detailed information for each sample."""
        content = dataset[batch_start + i]['prompt'][1]['content']
        sentence = extract_smiles_from_prompt(content, task)
        
        logger.info(f"{task.upper()} Sample {sample_idx}/{total_samples} (generate_n={self.config.generate_n}, n={self.config.n}, top_k_metrics={self.config.top_k_metrics})")
        
        if 'retrosynthesis' in task:
            logger.info(f"Product SMILES: {sentence}")
        else:
            logger.info(f"Reactants SMILES: {sentence}")
            
        logger.info(f"Best Generation (Index {best_gen_idx + 1}) Think/Reasoning:\n---\n{extract_xml_think(best_output_text, self.config.prompt_type)}\n---")
        logger.info(f"Raw Outputs:\n---\n{json.dumps(raw_outputs, indent=2)}\n---")
        logger.info(f"Predicted (after dedup/rerank): {json.dumps(predicted_list)}")
        logger.info(f"Expected: {expected}")
        
        for k in self.config.top_k_metrics_list:
            logger.info(f"Top-{k} Correct: {compute_top_k_accuracy(expected, predicted_list, k)}")
        logger.info("---")

    def _compute_final_metrics(self, task, dataset_name, prompt_type, total_samples, top_k_correct, logger):
        """Compute and return final evaluation metrics."""
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'task': task,
            'prompt_type': prompt_type,
            'dataset': dataset_name,
            'model': self.config.model_name,
            'generate_n': self.config.generate_n,
            'n': self.config.n,
            'top_k_metrics': self.config.top_k_metrics,
        }

        print(f"{task.upper()} {dataset_name} Evaluation Results (generate_n={self.config.generate_n}, n={self.config.n}, top_k_metrics={self.config.top_k_metrics}):")
        for k in self.config.top_k_metrics_list:
            accuracy = top_k_correct[k] / total_samples if total_samples > 0 else 0.0
            print(f"Top-{k} Accuracy: {accuracy:.4f}")
            results[f'top{k}_accuracy'] = accuracy
            if logger and not self.config.disable_logging_csv:
                logger.info(f"Top-{k} Accuracy: {accuracy:.4f}")

        # Save results to CSV if logging enabled
        if not self.config.disable_logging_csv:
            csv_file = save_results_to_csv(results, task, dataset_name, self.config)
            if logger:
                logger.info(f"Results saved to {csv_file}")

        return results
