# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any

def setup_logging(task: str, model_name: str, dataset_name: str, config, disable_logging: bool = False):
    """Set up logging for evaluation."""
    if disable_logging:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        return logger, None
    
    log_dir = f"logs/evaluate_{task}/{model_name.replace('/', '_')}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'eval_{task}_{dataset_name}_bestof{config.n}_topk{config.top_k_metrics}_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model: {model_name}")
    
    return logger, log_file

def save_results_to_csv(results: Dict[str, Any], task: str, dataset_name: str, config):
    """Save evaluation results to CSV file."""
    results_dir = f"/export/data/rli/Project/llms/retro/logs/results/results_temp_{config.temperature}"
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, f"{task}_{dataset_name}_bestof{config.best_of}_topk{config.top_k_metrics}_evaluation_results.csv")

    df = pd.DataFrame([results])
    if os.path.exists(csv_file):
        try:
            existing_df = pd.read_csv(csv_file)
            for col in df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
            df = df[existing_df.columns]
            df.to_csv(csv_file, mode='a', header=False, index=False)
            print(f"Results appended to {csv_file}")
            return csv_file
        except Exception as e:
            print(f"Error appending to CSV {csv_file}: {e}")
            csv_file_fallback = os.path.join(results_dir, f"{task}_{dataset_name}_bestof{config.best_of}_topk{config.top_k_metrics}_evaluation_results_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(csv_file_fallback, index=False)
            print(f"Saved results to fallback CSV: {csv_file_fallback}")
            return csv_file_fallback
    else:
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")
        return csv_file