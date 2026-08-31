import json
from typing import Dict
from ..config.prompts import get_system_prompt, get_plan_prompt

def format_training_example(example: Dict, prompt_style: str) -> Dict:
    """Format a training example based on task and prompt style."""
    task = example.get("task", "retrosynthesis")
    
    if task in ["retrosynthesis", "retrosynthesis_class"]:
        return _format_retrosynthesis_example(example, prompt_style)
    else:
        return _format_forward_prediction_example(example, prompt_style)

def _format_retrosynthesis_example(example: Dict, prompt_style: str) -> Dict:
    """Format retrosynthesis training example."""
    variant = example.get("variant", "mapped")
    product = (
        example.get("product", "")
        if variant == "mapped"
        else example.get("product_unmapped", example.get("product", ""))
    )
    reactants = example["reactants"]
    
    map_note = _get_map_note(variant)
    extra = f" and reaction class: \"{example.get('rxn_Class', 'Unknown')}\"" if "class" in example.get("task", "") else ""
    
    prompt = f"""
Given the product SMILES: "{product}"{extra}

Predict the reactants required to synthesize this product.

### Instruction:
- Think step-by-step to identify the reactants based on the product SMILES.
- Consider common retrosynthetic disconnections and reaction types.
{map_note}- Return the predicted reactants in SMILES format as a JSON object:
  {{"reactants": "SMILES_string"}}.
"""
    
    response_dict = {"reactants": reactants}
    answer_part = f"<answer>\n{json.dumps(response_dict, ensure_ascii=False)}\n</answer>"
    response = get_plan_prompt(task) + answer_part if prompt_style == "with_plan" else answer_part
    
    return {
        "conversations": [
            {"role": "system", "content": get_system_prompt(prompt_style)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ],
        "task": example.get("task", "retrosynthesis")
    }

def _format_forward_prediction_example(example: Dict, prompt_style: str) -> Dict:
    """Format forward prediction training example."""
    reactants = example["reactants"]
    product = example["products"]
    reaction_centers = example.get("reaction_centers", "Unknown")
    
    prompt = f"""
Given the reactants SMILES: "{reactants}" and reaction centers: "{reaction_centers}"

Predict the product of this reaction.

### Instruction:
- Think step-by-step to identify the product based on the reactants.
- Consider common reaction mechanisms.
- Return the predicted product in SMILES format as a JSON object:
  {{"product": "SMILES_string"}}.
"""
    
    response_dict = {"product": product}
    answer_part = f"<answer>\n{json.dumps(response_dict, ensure_ascii=False)}\n</answer>"
    response = get_plan_prompt("forward_prediction") + answer_part if prompt_style == "with_plan" else answer_part
    
    return {
        "conversations": [
            {"role": "system", "content": get_system_prompt(prompt_style)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ],
        "task": "forward_prediction"
    }

def _get_map_note(variant: str) -> str:
    """Get mapping note based on dataset variant."""
    if variant != "mapped":
        return "- Note: This is an unmapped SMILES representation.\n"
    return ""
