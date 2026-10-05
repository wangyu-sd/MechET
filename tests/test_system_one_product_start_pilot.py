from types import SimpleNamespace

import torch

from mechet.electron_pointer import parse_pointer_observation
from mechet.jev_style_decision import FactorizedTypedElectronFlowHead
from mechet.natural_language_electron_flow import build_inventory
from scripts.audit_system_one_observation_parity import mapped_from_visible
from scripts.eval_system_one_product_start_pilot import (
    HybridPolicy,
    ProductInput,
    ReactionTask,
    canonical_visible,
    execute_ranked_electron_action,
    rollout,
    select_tasks,
    typed_runtime_encoding,
)
from scripts.train_jev_style_electron_flow import prepare_typed


class FakeTokenizer:
    unk_token_id = -1
    controls = {
        "<|fim_prefix|>": 1001,
        "<|fim_middle|>": 1002,
        "<|box_start|>": 1003,
        "<|box_end|>": 1004,
        "<|fim_suffix|>": 1005,
    }

    def convert_tokens_to_ids(self, token):
        return self.controls.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        if text in self.controls:
            return {"input_ids": [self.controls[text]]}
        return {"input_ids": [10 + ord(char) % 200 for char in text]}


def test_runtime_typed_input_is_identical_to_teacher_forced_encoding():
    messages = [
        {"role": "system", "content": "infer a retrosynthetic electron flow"},
        {"role": "user", "content": "ANNOTATED CURRENT STATE: <A01>C<A02>O"},
    ]
    observation = parse_pointer_observation(messages[-1]["content"])
    example = SimpleNamespace(
        row_id="r::event", messages=messages,
        atom_names=observation.atom_names, bonds=observation.bonds,
        source_targets=(("atom", 0, 0),), sink_targets=(("atom", 1, 1),),
    )
    tokenizer = FakeTokenizer()
    assert typed_runtime_encoding(tokenizer, messages, observation) == (
        prepare_typed(example, tokenizer).encoding
    )


def test_runtime_typed_forward_uses_the_trained_head_signature():
    messages = [
        {"role": "system", "content": "infer a retrosynthetic electron flow"},
        {"role": "user", "content": "ANNOTATED CURRENT STATE: <A01>C<A02>O"},
    ]
    observation = parse_pointer_observation(messages[-1]["content"])

    class FakePolicy:
        def __call__(self, *, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.randn(1, input_ids.shape[1], 16))

    runtime = SimpleNamespace(
        torch=torch, device=torch.device("cpu"), dtype=torch.float32,
        tokenizer=FakeTokenizer(), typed_cap=8192, typed_policy=FakePolicy(),
        typed_head=FactorizedTypedElectronFlowHead(16, 8).eval(),
    )
    ranked, length = HybridPolicy.electrons(runtime, messages, observation)
    assert len(ranked) == len(set(ranked)) == 8
    assert length > 0


def test_hash_selection_does_not_depend_on_expected_precursor():
    tasks = [ReactionTask(ProductInput(str(index), "CO", "system", []), "A", 3)
             for index in range(12)]
    changed = [ReactionTask(task.policy_input, "different", 999) for task in tasks]
    assert [task.policy_input.reaction_id for task in select_tasks(tasks, seed=17, limit=5)] == [
        task.policy_input.reaction_id for task in select_tasks(changed, seed=17, limit=5)
    ]


def test_strict_endpoint_canonicalization_keeps_explicit_hydrogen():
    assert canonical_visible("[H]O") != canonical_visible("O")
    assert canonical_visible("CO.[H][H]") == canonical_visible("[H][H].CO")


def test_runtime_inventory_keeps_explicit_hydrogen_atom_handles():
    prompt = build_inventory(mapped_from_visible("CO.[H]O")).prompt
    observation = parse_pointer_observation(prompt)
    assert len(observation.atom_names) == 4


def test_premature_finish_is_not_credited_as_executable_endpoint():
    class AlwaysFinish:
        def route(self, messages, tools, observation):
            return "finish_trace", [0.0, 0.0, 1.0], 10

    class UnusedRetriever:
        def propose(self, target, current):
            raise AssertionError("retriever should not be called")

    task = ProductInput("example", "CO", "Infer an electron flow.", [])
    result = rollout(task, AlwaysFinish(), UnusedRetriever(), max_actions=4)
    assert result["terminal"] == "PREMATURE_OR_PENDING_FINISH"
    assert result["completed"] is False
    assert result["predicted_precursor"] is None


def test_legality_backoff_preserves_baseline_then_uses_ranked_singleton():
    calls = []

    def executor(mapped, observation, selected):
        assert mapped == "mapped" and observation == "observation"
        calls.append(tuple(selected))
        return {"ok": list(selected) == [20], "code": "PASS" if list(selected) == [20]
                else "CHEMICAL_STATE_INVALID"}

    selected, result, rank, attempts = execute_ranked_electron_action(
        "mapped", "observation", [10, 20, 30], executor=executor
    )
    assert selected == [10] and result["ok"] is False
    assert rank == 1 and attempts == 2
    assert calls == [(10, 20), (10,)]

    calls.clear()
    selected, result, rank, attempts = execute_ranked_electron_action(
        "mapped", "observation", [10, 20, 30],
        legality_backoff=True, executor=executor,
    )
    assert selected == [20] and result["ok"] is True
    assert rank == 2 and attempts == 3
    assert calls == [(10, 20), (10,), (20,)]
