# -*- coding: utf-8 -*-
import json
from typing import Dict

# System prompt dictionary
system_prompts = {
    'with_plan': """
        Respond in the following format:
        <plan>
        ...
        </plan>
        <answer>
        Provide the final answer in JSON format as specified in the instruction.
        </answer>
""",
    'without_plan': """
        Respond in the following format:
        <answer>
        Provide the final answer in JSON format as specified in the instruction.
        </answer>
"""
}

# Plan prompt dictionary (used if with_plan)
plan_prompts = {
    'retrosynthesis': """<plan>
To predict the reactants for the product SMILES:
1. Identify key functional groups and structural features in the product.
2. Propose retrosynthetic disconnections based on common reaction types.
3. Validate that the proposed reactants are chemically feasible.
</plan>
""",
    'retrosynthesis_class': """<plan>
To predict the reactants for the product SMILES with given reaction class:
1. Use the reaction class to guide disconnections.
2. Identify functional groups and propose class-aligned retrosynthetic steps.
3. Validate feasibility.
</plan>
""",
    'forward_prediction': """<plan>
To predict the product from reactants with given reaction class:
1. Analyze reactants' functional groups.
2. Apply forward reaction mechanisms based on the class.
3. Validate the proposed product.
</plan>
"""
}

def format_retrosynthesis_prompt(example: Dict) -> Dict:
    variant = example.get("variant", "mapped")
    product = (
        example["product"]
        if variant == "mapped"
        else example.get("product_unmapped", example["product"])
    )
    reactants = example["reactants"]
    
    map_note = (
        "- Note: This is an unmapped SMILES representation...\n"
        if variant != "mapped"
        else ""
    )
    
    prompt = f"""
Task: Retrosynthesis
Given the product SMILES: "{product}"
Predict the reactants required to synthesize this product.
### Instruction:
- Think step-by-step to identify the reactants based on the product SMILES.
{map_note}- Return the predicted reactants in SMILES format as a JSON object:
  {{"reactants": "SMILES_string"}}.
"""
    response_dict = {"reactants": reactants}
    answer_part = f"<answer>\n{json.dumps(response_dict, ensure_ascii=False)}\n</answer>"
    response = plan_prompts['retrosynthesis'] + answer_part if example.get("prompt_style", "with_plan") == 'with_plan' else answer_part
    
    return {
        "conversations": [
            {"role": "system", "content": system_prompts[example.get("prompt_style", "with_plan")]},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ],
        "task": "retrosynthesis"
    }

def format_retrosynthesis_class_prompt(example: Dict) -> Dict:
    variant = example.get("variant", "mapped")
    product = (
        example["product"]
        if variant == "mapped"
        else example.get("product_unmapped", example["product"])
    )
    rxn_class = example.get("rxn_Class", "Unknown")
    reactants = example["reactants"]
    
    map_note = (
        "- Note: This is an unmapped SMILES representation...\n"
        if variant != "mapped"
        else ""
    )
    
    prompt = f"""
Task: Retrosynthesis with Reaction Class
Given the product SMILES: "{product}" and reaction class: "{rxn_class}"
Predict the reactants required to synthesize this product, guided by the reaction class.
### Instruction:
- Think step-by-step, using the reaction class to narrow down possible disconnections.
{map_note}- Return the predicted reactants in SMILES format as a JSON object:
  {{"reactants": "SMILES_string"}}.
"""
    response_dict = {"reactants": reactants}
    answer_part = f"<answer>\n{json.dumps(response_dict, ensure_ascii=False)}\n</answer>"
    response = plan_prompts['retrosynthesis_class'] + answer_part if example.get("prompt_style", "with_plan") == 'with_plan' else answer_part
    
    return {
        "conversations": [
            {"role": "system", "content": system_prompts[example.get("prompt_style", "with_plan")]},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ],
        "task": "retrosynthesis_class"
    }

def format_forward_prediction_prompt(example: Dict) -> Dict:
    reactants = example["reactants"]
    reaction_centers = example.get("reaction_centers", "Unknown")
    product = example["products"]
    variant = example.get("variant", "mapped")
    
    prompt = f"""
Task: Forward Reaction Prediction with Class
Given the reactants SMILES: "{reactants}" and reaction class: "{reaction_centers}"
Predict the product of this reaction, guided by the reaction class.
### Instruction:
- Think step-by-step to identify the product based on the reactants and class.
- Return the predicted product in SMILES format as a JSON object:
  {{"product": "SMILES_string"}}.
"""
    response_dict = {"product": product}
    answer_part = f"<answer>\n{json.dumps(response_dict, ensure_ascii=False)}\n</answer>"
    response = plan_prompts['forward_prediction'] + answer_part if example.get("prompt_style", "with_plan") == 'with_plan' else answer_part
    
    return {
        "conversations": [
            {"role": "system", "content": system_prompts[example.get("prompt_style", "with_plan")]},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ],
        "task": "forward_prediction"
    }


format_functions = {
    "retrosynthesis": format_retrosynthesis_prompt,
    "retrosynthesis_class": format_retrosynthesis_class_prompt,
    "forward_prediction": format_forward_prediction_prompt,
}
