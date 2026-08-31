# -*- coding: utf-8 -*-
import re
from typing import Dict

def extract_smiles_from_prompt(content: str, task: str) -> str:
    """Extract SMILES string from prompt content."""
    if 'retrosynthesis' in task:
        sentence_match = re.search(r"Given the product SMILES: \"(.*?)\"\s*Predict", content, re.DOTALL)
    else:
        sentence_match = re.search(r"Given the reactants SMILES: \"(.*?)\"\s*Predict", content, re.DOTALL)
    return sentence_match.group(1).strip() if sentence_match else "N/A"