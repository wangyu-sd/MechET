#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.arguments import setup_argument_parser, validate_arguments, parse_tasks
from data.dataset_loader import DatasetLoader
from model.model_loader import ModelLoader
from evaluation.evaluator import Evaluator
from utils.logging import setup_logger

def main():
    """Main evaluation function."""
    # Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Parse tasks
    selected_tasks = parse_tasks(args)
    
    # Load model and tokenizer
    model_loader = ModelLoader(args)
    model, tokenizer = model_loader.load_model_and_tokenizer()
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader(args)
    
    # Run evaluation for each task
    for task in selected_tasks:
        try:
            # Load dataset
            eval_dataset, dataset_name = dataset_loader.load_dataset(task)
            
            # Setup logger
            logger = setup_logger(args, task, args.model_name, dataset_name)
            if logger:
                logger.info(f"Loading model: {args.model_name}")
            
            # Initialize evaluator
            evaluator = Evaluator(model, tokenizer, args, logger)
            
            # Run evaluation
            results = evaluator.evaluate_dataset(
                task=task,
                dataset=eval_dataset,
                dataset_name=dataset_name,
                batch_size=args.batch_size
            )
            
            # Save results
            evaluator.save_results(results, task, dataset_name)
            
            if logger:
                logger.info(f"Evaluation completed for task: {task}")
                
        except Exception as e:
            print(f"Error evaluating task {task}: {e}")
            if logger:
                logger.error(f"Error evaluating task {task}: {e}")

if __name__ == "__main__":
    main()