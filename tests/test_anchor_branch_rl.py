import json
import hashlib
from pathlib import Path
import random
from types import ModuleType, SimpleNamespace

import pytest

from mechet.anchor_branch_rl import (
    ANCHOR_BRANCH_VERSION,
    AnchorTask,
    anchor_messages,
    assign_anchor_advantages,
    build_anchor_task,
    choose_horizon,
    first_step_token_mask,
    score_completion,
    update_frontier,
)


TARGET = "[CH3:1][OH:2]"
GOLD = "[step(move(bond(1,2),atom(2)),move(lp(3),bond(1,3)),imports=['[Br-:3]'])]"
EXPECTED = "[CH3:1][Br:3].[OH-:2]"
ROW = {
    "id": "unit",
    "target_smiles": TARGET,
    "expected_precursor": EXPECTED,
    "messages": [
        {"role": "system", "content": "Output STEPS"},
        {"role": "user", "content": "TARGET=" + TARGET},
        {"role": "assistant", "content": GOLD},
    ],
}


class CharacterTokenizer:
    eos_token_id = 0
    unk_token_id = -1

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(item) for item in ids if not (skip_special_tokens and item == 0))

    def convert_tokens_to_ids(self, token):
        return {"<|endoftext|>": 0, "<|im_end|>": 0, "<|im_start|>": 1}.get(
            token, -1
        )

    def apply_chat_template(
        self, messages=None, conversation=None, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        messages = messages if messages is not None else conversation
        rendered = "\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in messages
        )
        return rendered + ("\nASSISTANT: " if add_generation_prompt else "")


def test_reference_resets_are_exact_and_gold_free():
    task = build_anchor_task(ROW, 1)
    assert task.horizon == task.total_steps == 1
    assert task.is_full_episode
    prompt = json.dumps(anchor_messages(task))
    assert GOLD not in prompt and EXPECTED not in prompt
    score = score_completion(task, task.gold_suffix, terminated=True)
    assert score["correct"] and score["reward"] == 1
    assert task.state_hash and ANCHOR_BRANCH_VERSION


def test_real_multistep_reference_suffixes_reach_same_endpoint():
    path = Path("data/flower_python_template_slots_v1/train.jsonl")
    if not path.is_file():
        pytest.skip("large FlowER training artifact is not part of source checkout")
    with path.open() as handle:
        rows = [json.loads(next(handle)) for _ in range(8)]
    checked = 0
    for row in rows:
        total = int(row["metadata"]["n_program_steps"])
        for horizon in range(1, total + 1):
            task = build_anchor_task(row, horizon)
            score = score_completion(task, task.gold_suffix, terminated=True)
            assert score["formal_execute"] and score["correct"]
            checked += 1
    assert checked >= 8


def test_adaptive_horizon_has_full_rehearsal_and_local_band():
    assert choose_horizon(7, 2, random.Random(1), full_episode_fraction=1) == 7
    values = {
        choose_horizon(7, 3, random.Random(seed), full_episode_fraction=0)
        for seed in range(100)
    }
    assert values == {2, 3, 4}


def test_only_first_step_tokens_receive_local_credit():
    tokenizer = CharacterTokenizer()
    text = "[step(move(bond(1,2),atom(2))),step(move(lp(2),bond(1,2)))]"
    ids = [ord(char) for char in text] + [0]
    mask = first_step_token_mask(tokenizer, ids, text)
    first_end = text.index("),step") + 1
    assert mask[:first_end] == [1] * first_end
    assert not any(mask[first_end:])
    invalid = "not a program"
    assert all(first_step_token_mask(tokenizer, [ord(c) for c in invalid], invalid))


def test_same_action_is_pooled_before_counterfactual_advantage():
    def record(action, reward, correct=False):
        return {
            "id": "unit",
            "anchor": {"state_hash": "abc"},
            "action_fingerprint": action,
            "reward": reward,
            "score": {"correct": correct},
        }

    records = [record("a", 1, True), record("a", 0), record("b", 0), record("c", -0.1)]
    summary = assign_anchor_advantages(records)
    assert summary["effective"] and summary["unique_actions"] == 3
    assert records[0]["anchor_action_q"] == records[1]["anchor_action_q"] == 0.5
    assert records[0]["advantage"] == records[1]["advantage"]
    assert records[0]["advantage"] > records[2]["advantage"] > records[3]["advantage"]


def test_frontier_promotes_only_with_informative_successful_groups():
    groups = [
        {"effective": True, "endpoint_success": index < 7, "is_full_episode": False}
        for index in range(10)
    ]
    frontier, report = update_frontier(
        2, groups, promote_pass_rate=0.6, min_effective_groups=8, maximum=5
    )
    assert frontier == 3 and report["promoted"]
    frontier, report = update_frontier(
        2, groups[:3], promote_pass_rate=0.6, min_effective_groups=8, maximum=5
    )
    assert frontier == 2 and not report["promoted"]


def test_reward_is_endpoint_primary_with_only_negative_invalid_constraint():
    task = build_anchor_task(ROW, 1)
    wrong = score_completion(
        task,
        "[step(move(bond(1,2),atom(2)))]",
        terminated=True,
    )
    invalid = score_completion(task, "[]", terminated=True)
    assert wrong["reward"] <= 0
    assert invalid["reward"] == pytest.approx(-0.1)


def test_structural_noop_reference_uses_full_endpoint_discriminator():
    program = (
        "[step(move(bond(1,2),atom(2)),imports=['[Na+:3]']),"
        "step(move(lp(2),bond(1,2)))]"
    )
    task = AnchorTask(
        reaction_id="aux-only",
        original_target=TARGET,
        anchor_state=TARGET,
        expected_precursor=TARGET + ".[Na+:3]",
        horizon=2,
        total_steps=2,
        prefix_steps=0,
        gold_suffix=program,
        state_hash=hashlib.sha256(TARGET.encode()).hexdigest(),
    )
    score = score_completion(task, program, terminated=True)
    assert score["formal_execute"] and score["correct"] and not score["noop"]


def test_collector_builds_same_state_group_and_first_action_masks(tmp_path, monkeypatch):
    import sys
    from anchor_branch_stage import collect

    tokenizer = CharacterTokenizer()

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            return tokenizer

        def generate(self, prompts, params, **kwargs):
            results = []
            for _ in prompts:
                outputs = []
                for text in (GOLD, "[]")[: params.n]:
                    ids = tokenizer.encode(text, add_special_tokens=False) + [0]
                    outputs.append(
                        SimpleNamespace(
                            token_ids=ids,
                            finish_reason="stop",
                            logprobs=[
                                {token: SimpleNamespace(logprob=-1.0)} for token in ids
                            ],
                        )
                    )
                results.append(SimpleNamespace(outputs=outputs))
            return results

    vllm = ModuleType("vllm")
    vllm.__version__ = "0.8.5"
    vllm.LLM = FakeLLM
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    request = ModuleType("vllm.lora.request")
    request.LoRARequest = lambda *args: None
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(ROW) + "\n")
    output = tmp_path / "rollout.jsonl"
    collect(
        SimpleNamespace(
            data=str(source),
            output=str(output),
            model="unused",
            adapter="unused",
            rank=0,
            world_size=1,
            k=2,
            seed=17,
            round_index=0,
            frontier=1,
            full_episode_fraction=0.0,
            replay_fraction=0.0,
            invalid_penalty=0.1,
            temperature=1.0,
            batch_products=1,
            max_new_tokens=512,
            max_context=4096,
            evaluation=False,
            full_only=False,
        )
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 2
    assert len({row["anchor"]["state_hash"] for row in records}) == 1
    assert records[0]["advantage"] > records[1]["advantage"]
    assert 0 < sum(records[0]["loss_mask"]) < len(records[0]["prediction"]) + 2
    assert len(records[0]["input_ids"]) == len(records[0]["old_logps"]) == len(
        records[0]["loss_mask"]
    )
