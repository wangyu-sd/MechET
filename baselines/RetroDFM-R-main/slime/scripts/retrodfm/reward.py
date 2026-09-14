import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

_END_TOKENS = ("<|im_end|>", "<|endoftext|>")
_MIN_THINK_CHARS = 100
_GRM_MODEL = os.getenv("GRM_MODEL", "Qwen3.6-35B-A3B")
_GRM_SEMAPHORE = asyncio.Semaphore(int(os.getenv("GRM_CONCURRENCY", "512")))
_GRM_CLIENT = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "None"),
    max_retries=5,
    timeout=3600,
)


def _strip_response(text: str) -> str:
    text = (text or "").strip()
    for token in _END_TOKENS:
        text = text.removesuffix(token).strip()
    return text


def _extract_tag(text: str, left_tag: str, right_tag: str) -> str:
    left = text.find(left_tag)
    if left == -1:
        return ""
    right = text.find(right_tag, left + len(left_tag))
    if right == -1:
        return ""
    return text[left + len(left_tag) : right].strip()


def _label_to_reactants(label: Any) -> str:
    if isinstance(label, dict):
        return str(label.get("reactants", ""))
    label = str(label or "").strip()
    return _extract_tag(label, "<answer>", "</answer>") or label


def _judge_prompt(product: str, cot: str, reactants: str) -> str:
    return f"""You are an expert reviewer for single-step organic retrosynthesis reasoning.

Judge whether the chain of thought and final reactants are internally consistent for a retrosynthesis task.
You will receive:
- product: the target product SMILES extracted from <SMILES>.
- cot: the model's reasoning extracted from <think>.
- reactants: the model's final answer extracted from <answer>.

Use exactly these 0-1 integer dimensions:
- soundness: Chemical Soundness. Are the chemical concepts, bond identifications, functional group reasoning, and reaction claims inside cot factually correct? Score 1 only if the reasoning is chemically plausible and contains no material false chemistry. Otherwise score 0.
- faithfulness: Does reactants logically and directly execute the exact strategy proposed in cot without contradictions? Score 1 only if the final reactants match the disconnection/reaction strategy described in cot. Otherwise score 0.

Return exactly one JSON object and no prose outside JSON. The JSON schema is:
{{
  "soundness": 1,
  "faithfulness": 1,
  "rationale": "2-4 sentence justification grounded in the product, reasoning, and reactants"
}}

soundness and faithfulness must be integers, each either 0 or 1.

Product SMILES:
{product}

Chain of thought:
{cot}

Final reactants:
{reactants}

Assess chemical soundness and faithfulness. Return only the required JSON object."""


def _parse_judgment(text: str) -> tuple[int, int]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    judgment = json.loads(text)
    if set(judgment) != {"soundness", "faithfulness", "rationale"}:
        raise ValueError("unexpected GRM response schema")
    soundness = judgment["soundness"]
    faithfulness = judgment["faithfulness"]
    if type(soundness) is not int or soundness not in (0, 1):
        raise ValueError("soundness must be 0 or 1")
    if type(faithfulness) is not int or faithfulness not in (0, 1):
        raise ValueError("faithfulness must be 0 or 1")
    if not isinstance(judgment["rationale"], str) or not judgment["rationale"].strip():
        raise ValueError("rationale must be a non-empty string")
    return soundness, faithfulness


async def _grm_reward(product: str, cot: str, reactants: str) -> tuple[int, int]:
    if not product or not cot or not reactants:
        return 0, 0

    messages = [{"role": "user", "content": _judge_prompt(product, cot, reactants)}]
    async with _GRM_SEMAPHORE:
        for attempt in range(4):
            completion = await _GRM_CLIENT.chat.completions.create(
                model=_GRM_MODEL,
                messages=messages,
                temperature=1.0,
                max_tokens=32768,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            response = completion.choices[0].message.content or ""
            try:
                return _parse_judgment(response)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                if attempt == 3:
                    logger.warning("GRM output parsing failed: %s", error)
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": "Return only valid JSON with exactly soundness, faithfulness, and rationale.",
                        },
                    ]
                )
    return 0, 0


def _format_reward(response: str) -> float:
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    if any(response.count(tag) != 1 for tag in tags):
        return 0.0
    match = re.match(
        r"^<think>\n(?P<think>.*?)\n</think>\s*<answer>\n.*?\n</answer>$",
        response,
        re.DOTALL | re.MULTILINE,
    )
    return float(bool(match and len(match.group("think").strip()) >= _MIN_THINK_CHARS))


@lru_cache(maxsize=100_000)
def _canonicalize(smiles: str) -> str:
    smiles = (smiles or "").strip()
    if len(smiles) > 200:
        return ""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True)


def _accuracy_reward(response: str, label: str) -> float:
    try:
        prediction = _canonicalize(_extract_tag(response, "<answer>", "</answer>"))
        reference = _canonicalize(label)
        return float(bool(prediction and reference and prediction == reference))
    except Exception:
        return 0.0


async def custom_rm(args, sample):
    response = _strip_response(sample.response)
    accuracy = _accuracy_reward(response, _label_to_reactants(sample.label))
    format_reward = _format_reward(response)
    return {
        "rewards": accuracy + 0.5 * format_reward,
        "scores": accuracy,
        "passk_reward": float(accuracy == 1.0 and format_reward == 1.0),
        "format_reward": format_reward,
    }


async def custom_grm(args, sample):
    result = await custom_rm(args, sample)
    if result["format_reward"] == 0.0:
        result.update(sound_reward=0, faith_reward=0)
        return result

    response = _strip_response(sample.response)
    soundness, faithfulness = await _grm_reward(
        _extract_tag(str(sample.prompt), "<SMILES>", "</SMILES>"),
        _extract_tag(response, "<think>", "</think>"),
        _extract_tag(response, "<answer>", "</answer>"),
    )
    result.update(
        rewards=(
            result["scores"]
            + 0.2 * result["format_reward"]
            + 0.2 * soundness
            + 0.6 * faithfulness
        ),
        sound_reward=soundness,
        faith_reward=faithfulness,
    )
    return result


def _log_mean(samples, metrics, name: str, key: str) -> None:
    values = [
        float(sample.reward[key])
        for sample in samples
        if isinstance(sample.reward, dict) and key in sample.reward
    ]
    if values:
        metrics[name] = sum(values) / len(values)


def _log_metrics(samples, metrics, prefix: str) -> None:
    names = {
        "scores": "score_mean",
        "format_reward": "format_reward_mean",
        "passk_reward": "passk_reward_mean",
        "sound_reward": "sound_reward_mean",
        "faith_reward": "faith_reward_mean",
    }
    for key, name in names.items():
        _log_mean(samples, metrics, f"{prefix}/{name}", key)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    if rollout_extra_metrics is not None:
        _log_metrics(samples, rollout_extra_metrics, "rollout")
    return False


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    if extra_metrics is not None:
        for dataset_name, dataset_data in data.items():
            _log_metrics(dataset_data.get("samples") or [], extra_metrics, f"eval/{dataset_name}")
    return False
