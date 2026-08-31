# -*- coding: utf-8 -*-
from typing import Dict
from ..config.prompts import get_system_prompt

def format_retrosynthesis_prompt(example: Dict, task: str, prompt_type: str, dataset_file: str) -> Dict:
    """Format prompt for retrosynthesis tasks."""
    # Determine which product field to use
    if 'unmapped' in dataset_file:
        product = example.get("product_unmapped", example.get("product", ""))
        map_note = "- Note: This is an unmapped SMILES representation. If no atom mapping is provided or if atom mapping numbers are all 0 or -1, treat it as unmapped. Mapped and unmapped representations are similar but differ in format, with mapped including explicit atom correspondences. Still includes the atom mapping in the predicted reactants.\n"
    else:
        product = example["product"]
        map_note = ""
    
    extra = ""
    if task == 'retrosynthesis_class':
        rxn_class = example.get("rxn_Class", "Unknown")
        extra = f" and reaction class: \"{rxn_class}\""
    
    prompt = f"""
Given the product SMILES: "{product}"{extra}

Predict the reactants required to synthesize this product.

### Instruction:
- Think step-by-step to identify the reactants based on the product SMILES.
- Consider common retrosynthetic disconnections and reaction types (e.g., amide formation, esterification, nucleophilic substitution).
- Ensure the SMILES string is valid, includes atom mapping if present in the product, and uses '.' to separate multiple reactants.
{map_note}- Return the predicted reactants in SMILES format as a JSON object:
  {{"reactants": "SMILES_string"}}.
"""
    system_prompt = get_system_prompt(prompt_type)
    
    return {
        "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "expected": {"reactants": example["reactants"]}
    }

def format_forward_prediction_prompt(example: Dict, prompt_type: str, dataset_file: str) -> Dict:
    """Format prompt for forward prediction tasks."""
    # Determine which reactants field to use
    if 'unmapped' in dataset_file:
        reactants = example.get("reactants_unmapped", example.get("reactants", ""))
        map_note = "- Note: This is an unmapped SMILES representation. If no atom mapping is provided or if atom mapping numbers are all 0 or -1, treat it as unmapped. Mapped and unmapped representations are similar but differ in format, with mapped including explicit atom correspondences. Still includes the atom mapping in the predicted reactants.\n"
    else:
        reactants = example["reactants"]
        map_note = ""
    
    reaction_centers = example.get("reaction_centers", "Unknown")
    
    prompt = f"""
Given the reactants SMILES: "{reactants}" and reaction_centers: "{reaction_centers}"

Predict the product of this reaction.

### Instruction:
- Think step-by-step to identify the product based on the reactants SMILES.
- Consider common reaction types (e.g., amide formation, esterification, nucleophilic substitution).
- Ensure the SMILES string is valid, includes atom mapping if present.
{map_note}- Return the predicted product in SMILES format as a JSON object:
  {{"product": "SMILES_string"}}.
"""
    system_prompt = get_system_prompt(prompt_type)
    
    return {
        "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "expected": {"product": example["products"]}
    }

# Mapping of tasks to format functions
FORMAT_FUNCTIONS = {
    'retrosynthesis': format_retrosynthesis_prompt,
    'retrosynthesis_class': format_retrosynthesis_prompt,
    'forward_prediction': format_forward_prediction_prompt,
}

def get_prompt_formatter(task: str):
    """Get the appropriate prompt formatter for the task."""
    if task not in FORMAT_FUNCTIONS:
        raise ValueError(f"Unknown task: {task}")
    return FORMAT_FUNCTIONS[task]