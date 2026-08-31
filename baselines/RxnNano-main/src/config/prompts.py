# -*- coding: utf-8 -*-
from typing import Dict

SYSTEM_PROMPT_DICT = {
    "0": """
        Respond in the following format:
        <answer>
        Provide the final answer in JSON format as specified in the instruction.
        </answer>
        """,
    'with_plan': """
You are an expert in chemical reaction prediction. Respond in the following format:
<plan>
Provide step-by-step plan.
</plan>
<answer>
Provide the final answer in JSON format as specified in the instruction.
</answer>
""",
    "1-plan": """
        Respond in the following format:
        <plan>
        ...
        </plan>
        <answer>
        Provide the final answer in JSON format as specified in the instruction.
        </answer>
        """,
    "1-reason": """
        Respond in the following format:
        <reason>
        Provide step-by-step reasoning.
        </reason>
        <answer>
        Provide the final answer in JSON format as specified in the instruction.
        </answer>
        """,
    'plan-reason':"""
        Respond in the following format:
        <plan>
        Provide step-by-step plan to solve the task based on the given instructions and product SMILES.
        </plan>
        <reason>
        Conduct your detail reasoning.
        </reason>
        <answer>
        Provide the reactants in SMILES format as a JSON object: {"reactants": "SMILES_string"}
        </answer>
        """,
    "reason-think": """
        Respond in the following format:
        <plan>
        Provide step-by-step reasoning to predict the reactants from the given product SMILES.
        </plan>
        <think>
        Explain the key chemical transformations or patterns identified in the product that lead to the predicted reactants.
        </think>
        <answer>
        Provide the reactants in SMILES format as a JSON object: {"reactants": "SMILES_string"}
        </answer>

        Example:
        Product SMILES: CC(=O)Nc1ccccc1
        <plan>
        1. The product is an amide, suggesting an amide bond formation.
        2. A common retrosynthetic disconnection for amides involves an amine and an acyl chloride.
        3. The phenyl group (c1ccccc1) and the acetyl group (CC(=O)) suggest aniline (c1ccccc1NH2) and acetyl chloride (CC(=O)Cl) as reactants.
        </plan>
        <think>
        The amide bond is formed via nucleophilic acyl substitution, where the amine (aniline) attacks the acyl chloride (acetyl chloride), releasing HCl.
        </think>
        <answer>
        {"reactants": "c1ccccc1NH2.CC(=O)Cl"}
        </answer>
        """
}

def get_system_prompt(prompt_type: str) -> str:
    if prompt_type not in SYSTEM_PROMPT_DICT:
        raise ValueError(f"Invalid prompt type. Available types: {list(SYSTEM_PROMPT_DICT.keys())}")
    return SYSTEM_PROMPT_DICT[prompt_type]