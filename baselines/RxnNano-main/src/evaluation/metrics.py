# -*- coding: utf-8 -*-
import re
import json
from typing import List, Dict, Union
from rdkit import Chem, rdBase

def extract_xml_answer(text: str) -> str:
    """Extract content from <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_xml_think(text: str, prompt_type: str) -> str:
    """Extract thinking/reasoning content based on prompt type."""
    output = ""
    if prompt_type in ["1-plan", "plan-reason"]:
        plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
        reason_match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        if plan_match:
            output += f"Plan: {plan_match.group(1).strip()}"
        if reason_match:
            output += f"\nReason: {reason_match.group(1).strip()}" if output else f"Reason: {reason_match.group(1).strip()}"
    elif prompt_type == "1-reason":
        reason_match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        if reason_match:
            output += f"Reason: {reason_match.group(1).strip()}"
    elif prompt_type == "reason-think":
        plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if plan_match:
            output += f"Plan: {plan_match.group(1).strip()}"
        if think_match:
            output += f"\nThink: {think_match.group(1).strip()}" if output else f"Think: {think_match.group(1).strip()}"
    return output

def normalize_smiles(smiles: str) -> str:
    """Normalize SMILES by sorting components to make order invariant."""
    if not smiles:
        return ""
    components = smiles.split('.')
    components = [c.strip() for c in components if c.strip()]
    components.sort()
    return '.'.join(components)

def is_valid_smiles(smiles: Union[str, list]) -> bool:
    """Validate one or more SMILES with RDKit."""
    if isinstance(smiles, list):
        return any(is_valid_smiles(s) for s in smiles if isinstance(s, str))
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    try:
        # Invalid sampled strings are expected here; keep the evaluation log readable.
        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False

def compute_top_k_accuracy(expected: str, predicted_list: List[str], k: int) -> bool:
    """Compute top-k accuracy with normalized SMILES."""
    norm_expected = normalize_smiles(expected)
    norm_predictions = [normalize_smiles(p) for p in predicted_list]
    return norm_expected in norm_predictions[:k]

def parse_predicted_answer(answer_text: str, expected_key: str) -> str:
    """Parse the predicted answer from JSON format."""
    if not answer_text:
        return ""
    try:
        predicted_json = json.loads(answer_text) if answer_text else {}
        return predicted_json.get(expected_key, "") if isinstance(predicted_json, dict) else ""
    except json.JSONDecodeError:
        return ""

def deduplicate_and_rerank(predictions: List[str], scores: List[float], n: int) -> List[str]:
    """Deduplicate predictions and rerank by score."""
    return [pred for pred, _ in deduplicate_and_rerank_with_scores(predictions, scores, n)]

def deduplicate_and_rerank_with_scores(
    predictions: List[str], scores: List[float], n: int
) -> List[tuple[str, float]]:
    """Deduplicate predictions and return the retained score with each prediction."""
    norm_to_data = {}
    for pred, score in zip(predictions, scores):
        norm = normalize_smiles(pred)
        if norm and (norm not in norm_to_data or score > norm_to_data[norm][0]):
            norm_to_data[norm] = (score, pred)
    
    sorted_unique = sorted(norm_to_data.items(), key=lambda x: x[1][0], reverse=True)
    return [(data[1], data[0]) for _, data in sorted_unique][:n]
