from types import SimpleNamespace

import torch

from mechet.electron_pointer import parse_pointer_observation
from mechet.electron_pointer import candidate_keys
from mechet.jev_style_decision import FactorizedTypedElectronFlowHead
from mechet.natural_language_electron_flow import build_inventory
from scripts.audit_system_one_observation_parity import mapped_from_visible
from scripts.eval_system_one_product_start_pilot import (
    HybridPolicy,
    ProductInput,
    ReactionTask,
    canonical_visible,
    execute_ranked_electron_action,
    execute_first_event_target_focus,
    pair_touches_atoms,
    principal_component_atom_indices,
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


def test_first_event_focus_replaces_only_legal_context_only_action():
    state = "CC.CO"
    observation = parse_pointer_observation(build_inventory(mapped_from_visible(state)).prompt)
    principal = principal_component_atom_indices(state, "CC")
    sources = candidate_keys(len(observation.atom_names), observation.bonds, source=True)
    sinks = candidate_keys(len(observation.atom_names), observation.bonds, source=False)

    def flat(src, dst):
        return sources.index(src) * len(sinks) + sinks.index(dst)

    context1 = flat(("atom", 2, 2), ("atom", 3, 3))
    context2 = flat(("atom", 3, 3), ("atom", 2, 2))
    target = flat(("atom", 0, 0), ("atom", 1, 1))
    assert not pair_touches_atoms(observation, context1, principal)
    assert pair_touches_atoms(observation, target, principal)
    calls = []

    def executor(mapped, obs, selected):
        calls.append(tuple(selected))
        return {"ok": True, "code": "PASS"}

    selected, result, rank, attempts, metadata = execute_first_event_target_focus(
        "mapped", observation, [context1, context2, target], principal,
        executor=executor,
    )
    assert result["ok"] is True
    assert selected == [target, context1] and rank is None
    assert attempts == 2 and calls == [(context1, context2), (target, context1)]
    assert metadata["overrode_baseline"] is True
    assert metadata["target_pair_rank"] == 3


def test_first_event_focus_keeps_already_target_localized_baseline():
    state = "CC.CO"
    observation = parse_pointer_observation(build_inventory(mapped_from_visible(state)).prompt)
    principal = principal_component_atom_indices(state, "CC")
    sources = candidate_keys(len(observation.atom_names), observation.bonds, source=True)
    sinks = candidate_keys(len(observation.atom_names), observation.bonds, source=False)
    target = sources.index(("atom", 0, 0)) * len(sinks) + sinks.index(("atom", 1, 1))
    context = sources.index(("atom", 2, 2)) * len(sinks) + sinks.index(("atom", 3, 3))
    selected, result, _, attempts, metadata = execute_first_event_target_focus(
        "mapped", observation, [context, target], principal,
        executor=lambda *_: {"ok": True, "code": "PASS"},
    )
    assert selected == [context, target] and result["ok"] and attempts == 1
    assert metadata["overrode_baseline"] is False


def test_principal_localization_ignores_stereotag_only():
    assert principal_component_atom_indices("C[C@H](O)F.[Cl-]", "CC(O)F") == {0, 1, 2, 3}


def test_principal_target_prompt_changes_only_visible_target_line():
    captured = []

    class CaptureThenFinish:
        def route(self, messages, tools, observation):
            captured.append(messages[-1]["content"])
            return "finish_trace", [0.0, 0.0, 1.0], 10

    class UnusedRetriever:
        def propose(self, target, current):
            raise AssertionError("retriever should not be called")

    task = ProductInput("example", "CO.[Cl-]", "Infer an electron flow.", [])
    rollout(task, CaptureThenFinish(), UnusedRetriever(), max_actions=1,
            principal_product="CO", principal_target_prompt=True)
    assert captured[0].startswith("TARGET PRODUCT SMILES: CO\nCURRENT STATE SMILES: CO.[Cl-]")
    captured.clear()
    rollout(task, CaptureThenFinish(), UnusedRetriever(), max_actions=1)
    assert captured[0].startswith(
        "TARGET PRODUCT SMILES: CO.[Cl-]\nCURRENT STATE SMILES: CO.[Cl-]"
    )


def test_principal_trained_import_retrieval_queries_input_product():
    queries = []

    class ImportOnce:
        def route(self, messages, tools, observation):
            return "import_fragments", [1.0, 0.0, 0.0], 10

    class CaptureRetriever:
        target_is_principal = True

        def propose(self, target, current):
            queries.append((target, current))
            return (("O", 1),)

    task = ProductInput("example", "CO.[Cl-]", "Infer an electron flow.", [])
    rollout(task, ImportOnce(), CaptureRetriever(), max_actions=1,
            principal_product="CO", principal_target_prompt=True)
    assert queries == [("CO", "CO.[Cl-]")]
