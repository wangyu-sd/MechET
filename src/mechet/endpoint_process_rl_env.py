"""Private reward environment for endpoint-grounded event-level RLVR."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .endpoint_progress import (
    EndpointDistance,
    EndpointDistanceWeights,
    capped_progress_increment,
    endpoint_distance,
    potential_progress_reward,
)
from .grounded_event_search import (
    GroundedProposal,
    GroundedSearchNode,
    advance_grounded_beam,
    append_rejection_feedback,
)
from .in_place_grounded_flow import deterministic_unmapped_state, mapped_atom_numbers


@dataclass(frozen=True)
class ProcessRewardConfig:
    exact_endpoint: float = 4.0
    wrong_finish: float = -1.0
    invalid_proposal: float = -0.25
    state_cycle: float = -0.50
    retry_exhausted: float = -0.75
    unfinished: float = -1.0
    progress_event_cap: float = 0.25
    progress_total_cap: float = 1.0
    same_state_retry_limit: int = 2
    max_committed_events: int = 8
    max_import_fragments: int = 12
    max_import_atoms: int = 96
    max_fragment_heavy_atoms: int = 48
    distance_weights: EndpointDistanceWeights = EndpointDistanceWeights()

    def __post_init__(self) -> None:
        if self.same_state_retry_limit < 1:
            raise ValueError("same_state_retry_limit must be positive")
        if self.max_committed_events < 1:
            raise ValueError("max_committed_events must be positive")
        if self.exact_endpoint <= self.progress_total_cap:
            raise ValueError("exact endpoint reward must dominate progress shaping")


@dataclass(frozen=True)
class EventCredit:
    index: int
    raw_response: str
    action_name: str
    accepted: bool
    code: str
    process_reward: float
    progress_reward: float
    terminal_reward: float
    total_reward: float
    distance_before: float
    distance_after: float
    state_unchanged: bool
    terminated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action_name": self.action_name,
            "accepted": self.accepted,
            "code": self.code,
            "process_reward": self.process_reward,
            "progress_reward": self.progress_reward,
            "terminal_reward": self.terminal_reward,
            "total_reward": self.total_reward,
            "distance_before": self.distance_before,
            "distance_after": self.distance_after,
            "state_unchanged": self.state_unchanged,
            "terminated": self.terminated,
        }


@dataclass(frozen=True)
class EpisodeSummary:
    identifier: str
    start_horizon: str
    start_prefix_events: int
    endpoint_exact: bool
    explicit_finish: bool
    formal_terminal: bool
    termination_code: str
    generated_committed_events: int
    rejected_proposals: int
    retry_exhausted: bool
    cycle: bool
    start_distance: float
    terminal_distance: float
    cumulative_progress: float
    total_reward: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _unmapped_mol(mapped_smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(mapped_smiles or ""), params)
    if mol is None:
        raise ValueError("cannot align invalid imported fragment")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def align_isomorphic_fragment(
    current_fragment: str, reference_fragment: str
) -> dict[int, int] | None:
    """Return deterministic current-map -> reference-map alignment if isomorphic."""

    current_params = Chem.SmilesParserParams()
    current_params.removeHs = False
    reference_params = Chem.SmilesParserParams()
    reference_params.removeHs = False
    current = Chem.MolFromSmiles(str(current_fragment or ""), current_params)
    reference = Chem.MolFromSmiles(str(reference_fragment or ""), reference_params)
    if current is None or reference is None:
        return None
    if current.GetNumAtoms() != reference.GetNumAtoms():
        return None
    current_maps = [int(atom.GetAtomMapNum()) for atom in current.GetAtoms()]
    reference_maps = [int(atom.GetAtomMapNum()) for atom in reference.GetAtoms()]
    if any(value <= 0 for value in current_maps + reference_maps):
        return None
    query = _unmapped_mol(current_fragment)
    target = _unmapped_mol(reference_fragment)
    matches = target.GetSubstructMatches(
        query, uniquify=False, useChirality=True, maxMatches=256
    )
    full = [match for match in matches if len(match) == current.GetNumAtoms()]
    if not full:
        return None
    match = min(full, key=lambda item: tuple(reference_maps[index] for index in item))
    return {
        current_maps[index]: reference_maps[reference_index]
        for index, reference_index in enumerate(match)
    }


class EndpointProcessRLEnv:
    """Execute one sampled event at a time while keeping reward gold private."""

    def __init__(
        self,
        row: Mapping[str, Any],
        *,
        prefix_events: int = 0,
        start_horizon: str = "product",
        reward_config: ProcessRewardConfig | None = None,
    ) -> None:
        public = dict(row.get("public") or {})
        private = dict(row.get("private_reward") or {})
        identifier = str(row.get("id") or "")
        target = str(private.get("target_mapped_state") or "")
        expected = str(private.get("expected_precursor_mapped") or "")
        states = [str(value) for value in private.get("prefix_states") or []]
        transitions = [dict(value) for value in private.get("transitions") or []]
        if not identifier or not target or not expected:
            raise ValueError("endpoint RL row is missing private executor state")
        if len(states) != len(transitions) + 1:
            raise ValueError("prefix_states must contain one more state than transitions")
        if not 0 <= prefix_events < len(states):
            raise ValueError("prefix_events is outside the gold trace")
        self.identifier = identifier
        self.start_horizon = str(start_horizon)
        self.start_prefix_events = int(prefix_events)
        self.reward_config = reward_config or ProcessRewardConfig()
        self.expected_precursor = expected
        self.target_atom_maps = tuple(int(value) for value in private.get("target_atom_maps") or [])
        self.contributing_atom_maps = tuple(
            int(value) for value in private.get("contributing_atom_maps") or []
        )
        self.gold_import_fragments = tuple(
            str(value) for value in private.get("gold_import_fragments") or []
        )
        current = states[prefix_events]
        present_maps = mapped_atom_numbers(current)
        expected_maps = mapped_atom_numbers(expected)
        self.current_to_reference_maps: dict[int, int] = {
            value: value for value in present_maps if value in expected_maps
        }
        self._used_gold_imports = {
            index
            for index, fragment in enumerate(self.gold_import_fragments)
            if mapped_atom_numbers(fragment) <= present_maps
        }
        system_prompt = str(public.get("system_prompt") or "")
        product = str(public.get("product") or "")
        if not system_prompt or not product:
            raise ValueError("endpoint RL row is missing public product prompt")
        messages = (
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"TARGET: {product}\nCURRENT STATE: "
                    f"{deterministic_unmapped_state(current).text}"
                ),
            },
        )
        visible_prefix = tuple(
            deterministic_unmapped_state(value).text for value in states[: prefix_events + 1]
        )
        self.node = GroundedSearchNode(
            target_mapped_state=target,
            current_mapped_state=current,
            next_private_map=max(expected_maps | present_maps, default=0) + 1,
            messages=messages,
            transitions=tuple(_copy_json(value) for value in transitions[:prefix_events]),
            visited_visible_states=visible_prefix,
            trace_labels=tuple(f"gold_prefix_{index:03d}" for index in range(prefix_events)),
        )
        self.tools = _copy_json(public.get("tools") or [])
        self.credits: list[EventCredit] = []
        self.generated_committed_events = 0
        self.rejected_proposals = 0
        self.consecutive_rejections = 0
        self.cumulative_progress = 0.0
        self.done = False
        self.endpoint_exact = False
        self.explicit_finish = False
        self.formal_terminal = False
        self.termination_code = "ACTIVE"
        self.start_distance = self._distance(current)
        self.current_distance = self.start_distance

    def _distance(self, state: str) -> EndpointDistance:
        return endpoint_distance(
            state,
            self.expected_precursor,
            target_atom_maps=self.target_atom_maps,
            contributing_atom_maps=self.contributing_atom_maps,
            current_to_reference_maps=self.current_to_reference_maps,
            weights=self.reward_config.distance_weights,
        )

    def public_messages(self) -> list[dict[str, Any]]:
        return self.node.transcript()

    def public_tools(self) -> list[dict[str, Any]]:
        return _copy_json(self.tools)

    def _align_new_imports(self, fragments: Sequence[str]) -> None:
        for fragment in fragments:
            candidates: list[tuple[tuple[int, ...], int, dict[int, int]]] = []
            for index, gold in enumerate(self.gold_import_fragments):
                if index in self._used_gold_imports:
                    continue
                alignment = align_isomorphic_fragment(fragment, gold)
                if alignment is None:
                    continue
                candidates.append((tuple(sorted(alignment.values())), index, alignment))
            if not candidates:
                continue
            _, index, alignment = min(candidates)
            self._used_gold_imports.add(index)
            self.current_to_reference_maps.update(alignment)

    def _record(
        self,
        proposal: GroundedProposal,
        *,
        accepted: bool,
        code: str,
        process_reward: float = 0.0,
        progress_reward: float = 0.0,
        terminal_reward: float = 0.0,
        distance_before: float,
        distance_after: float,
        state_unchanged: bool,
    ) -> EventCredit:
        credit = EventCredit(
            index=len(self.credits),
            raw_response=proposal.raw_response,
            action_name=proposal.name,
            accepted=accepted,
            code=code,
            process_reward=float(process_reward),
            progress_reward=float(progress_reward),
            terminal_reward=float(terminal_reward),
            total_reward=float(process_reward + progress_reward + terminal_reward),
            distance_before=float(distance_before),
            distance_after=float(distance_after),
            state_unchanged=state_unchanged,
            terminated=self.done,
        )
        self.credits.append(credit)
        return credit

    def step(self, proposal: GroundedProposal) -> EventCredit:
        if self.done:
            raise RuntimeError("cannot step a terminated endpoint RL episode")
        before_state = self.node.current_mapped_state
        before_distance = self.current_distance.total
        result = advance_grounded_beam(
            [(self.node, [proposal])],
            beam_width=1,
            expected_precursor=self.expected_precursor,
            max_import_fragments=self.reward_config.max_import_fragments,
            max_import_atoms=self.reward_config.max_import_atoms,
            max_fragment_heavy_atoms=self.reward_config.max_fragment_heavy_atoms,
        )
        if result.terminals:
            terminal = result.terminals[0]
            self.node = terminal.node
            self.done = True
            self.explicit_finish = True
            self.formal_terminal = bool(terminal.result.get("formal_execute"))
            self.endpoint_exact = bool(terminal.result.get("endpoint_exact"))
            self.termination_code = (
                "EXACT_ENDPOINT" if self.endpoint_exact else "WRONG_FINISH"
            )
            terminal_reward = (
                self.reward_config.exact_endpoint
                if self.endpoint_exact
                else self.reward_config.wrong_finish
            )
            return self._record(
                proposal,
                accepted=True,
                code=self.termination_code,
                terminal_reward=terminal_reward,
                distance_before=before_distance,
                distance_after=before_distance,
                state_unchanged=True,
            )
        if result.selected:
            child = result.selected[0]
            imported = tuple(child.transitions[-1].get("imports") or ())
            self._align_new_imports(imported)
            after_distance = self._distance(child.current_mapped_state)
            proposed = potential_progress_reward(
                self.current_distance,
                after_distance,
                self.start_distance,
                event_cap=self.reward_config.progress_event_cap,
            )
            applied, self.cumulative_progress = capped_progress_increment(
                self.cumulative_progress,
                proposed,
                total_cap=self.reward_config.progress_total_cap,
            )
            self.node = child
            self.current_distance = after_distance
            self.generated_committed_events += 1
            self.consecutive_rejections = 0
            terminal_reward = 0.0
            if self.generated_committed_events >= self.reward_config.max_committed_events:
                self.done = True
                self.termination_code = "EVENT_BUDGET_EXHAUSTED"
                terminal_reward = self.reward_config.unfinished
            return self._record(
                proposal,
                accepted=True,
                code="PASS" if not self.done else self.termination_code,
                progress_reward=applied,
                terminal_reward=terminal_reward,
                distance_before=before_distance,
                distance_after=after_distance.total,
                state_unchanged=before_state == child.current_mapped_state,
            )

        rejection = result.rejected[0] if result.rejected else None
        code = rejection.code if rejection is not None else "EVENT_REJECTED"
        message = rejection.message if rejection is not None else "executor rejected event"
        self.rejected_proposals += 1
        process_reward = self.reward_config.invalid_proposal
        terminal_reward = 0.0
        if code == "STATE_CYCLE":
            process_reward = self.reward_config.state_cycle
            self.done = True
            self.termination_code = "STATE_CYCLE"
        else:
            self.consecutive_rejections += 1
            if self.consecutive_rejections >= self.reward_config.same_state_retry_limit:
                self.done = True
                self.termination_code = "RETRY_EXHAUSTED"
                terminal_reward = self.reward_config.retry_exhausted
        if not self.done and rejection is not None:
            self.node = append_rejection_feedback(self.node, proposal, rejection)
        return self._record(
            proposal,
            accepted=False,
            code=code if not self.done else self.termination_code,
            process_reward=process_reward,
            terminal_reward=terminal_reward,
            distance_before=before_distance,
            distance_after=before_distance,
            state_unchanged=True,
        )

    def reject_unparsed(
        self,
        *,
        raw_response: str,
        token_count: int,
        logprob_sum: float = 0.0,
        code: str = "PARSE_FAILED",
        message: str = "response did not contain exactly one valid tool call",
    ) -> EventCredit:
        proposal = GroundedProposal(
            name="invalid_response",
            arguments={},
            raw_response=raw_response,
            logprob_sum=logprob_sum,
            token_count=max(int(token_count), 1),
            seed=0,
        )
        if self.done:
            raise RuntimeError("cannot reject a response after termination")
        before = self.current_distance.total
        self.rejected_proposals += 1
        self.consecutive_rejections += 1
        terminal_reward = 0.0
        if self.consecutive_rejections >= self.reward_config.same_state_retry_limit:
            self.done = True
            self.termination_code = "RETRY_EXHAUSTED"
            terminal_reward = self.reward_config.retry_exhausted
        else:
            observation = {
                "ok": False,
                "code": code,
                "message": message,
                "current_state": deterministic_unmapped_state(
                    self.node.current_mapped_state
                ).text,
            }
            transcript = self.node.transcript()
            transcript.extend(
                [
                    {"role": "assistant", "content": raw_response},
                    {
                        "role": "tool",
                        "tool_call_id": "invalid_response",
                        "name": "invalid_response",
                        "content": json.dumps(observation, separators=(",", ":")),
                    },
                ]
            )
            self.node = GroundedSearchNode(
                target_mapped_state=self.node.target_mapped_state,
                current_mapped_state=self.node.current_mapped_state,
                next_private_map=self.node.next_private_map,
                messages=tuple(transcript),
                transitions=self.node.transitions,
                visited_visible_states=self.node.visited_visible_states,
                trace_labels=self.node.trace_labels,
                logprob_sum=self.node.logprob_sum + logprob_sum,
                token_count=self.node.token_count + max(int(token_count), 1),
            )
        return self._record(
            proposal,
            accepted=False,
            code=code if not self.done else self.termination_code,
            process_reward=self.reward_config.invalid_proposal,
            terminal_reward=terminal_reward,
            distance_before=before,
            distance_after=before,
            state_unchanged=True,
        )

    def summary(self) -> EpisodeSummary:
        if not self.done:
            raise RuntimeError("episode summary requested before termination")
        return EpisodeSummary(
            identifier=self.identifier,
            start_horizon=self.start_horizon,
            start_prefix_events=self.start_prefix_events,
            endpoint_exact=self.endpoint_exact,
            explicit_finish=self.explicit_finish,
            formal_terminal=self.formal_terminal,
            termination_code=self.termination_code,
            generated_committed_events=self.generated_committed_events,
            rejected_proposals=self.rejected_proposals,
            retry_exhausted=self.termination_code == "RETRY_EXHAUSTED",
            cycle=self.termination_code == "STATE_CYCLE",
            start_distance=self.start_distance.total,
            terminal_distance=self.current_distance.total,
            cumulative_progress=self.cumulative_progress,
            total_reward=sum(item.total_reward for item in self.credits),
        )

    def terminate_unfinished(self, code: str = "PROPOSAL_BUDGET_EXHAUSTED") -> None:
        """Attach an unfinished penalty to the last sampled event span."""

        if self.done:
            return
        if not self.credits:
            raise RuntimeError("cannot terminate an episode with no sampled action")
        last = self.credits[-1]
        terminal_reward = last.terminal_reward + self.reward_config.unfinished
        self.credits[-1] = replace(
            last,
            code=code,
            terminal_reward=terminal_reward,
            total_reward=last.process_reward + last.progress_reward + terminal_reward,
            terminated=True,
        )
        self.done = True
        self.termination_code = code

    def assert_no_reward_leakage(self) -> None:
        public_messages = self.public_messages()
        payload = json.dumps(
            {"messages": public_messages, "tools": self.public_tools()},
            ensure_ascii=False,
            sort_keys=True,
        )
        initial_payload = json.dumps(public_messages[:2], ensure_ascii=False, sort_keys=True)
        forbidden = {
            self.expected_precursor,
            deterministic_unmapped_state(self.expected_precursor).text,
        }
        for value in forbidden:
            if value and value in initial_payload:
                raise AssertionError("gold endpoint leaked into the initial model prompt")
        lowered = payload.lower()
        for key in ("gold_endpoint", "reference_endpoint", "endpoint_distance"):
            if key in lowered:
                raise AssertionError(f"reward-side field leaked into public payload: {key}")
