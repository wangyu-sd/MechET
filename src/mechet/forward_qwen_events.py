"""Blind forward next-event supervision from recorded FlowER transitions.

No final target, reverse proposal, or teacher product is used by the policy.
All source hydrogens are retained: public inventory must not depend on the
gold reaction centre. The legacy retrosynthetic compiler remains the default.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .forward_expert import verify_electron_step
from .mech_graph import parse_flower_line
from .natural_language_electron_flow import build_inventory, compile_event_arguments, render_event_arguments
from .proof_program import ProofEdge, _canonical_mapped, _get_be_delta, sides_equal
from .proof_to_trace import infer_moves_from_edge

SYSTEM = (
    "Predict the next FORWARD electron-flow event from the current reactant or "
    "intermediate mixture. This is forward synthesis, not retrosynthesis. "
    "No intended final product is provided. Temporary atom names refer only to "
    "the current annotated SMILES; they are not atom-map numbers. All reagents "
    "in this benchmark are present in the initial mixture. Move electrons from "
    "a lone pair, bond, or radical pair to an atom or bond using "
    "apply_forward_electron_flow. A coupled event is atomic. The tool owns "
    "the next state; do not predict its SMILES. Use finish_forward only when "
    "the mixture has reached a terminal reaction state."
)


def initialize_forward_state(smiles: str) -> str:
    """Create private addresses and explicit H from ONLY the proposed mixture."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError('invalid forward reactant mixture')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True), params)
    mol = Chem.AddHs(mol)
    for index, atom in enumerate(mol.GetAtoms(), 1):
        atom.SetAtomMapNum(index)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def forward_tools() -> list[dict[str, Any]]:
    # Reuse the established payload schema but never modify the inverse schema.
    from scripts.build_natural_language_event_sft import TOOLS
    event = copy.deepcopy(next(t for t in TOOLS if t['function']['name'] == 'apply_electron_flow'))
    event['function']['name'] = 'apply_forward_electron_flow'
    event['function']['description'] = 'Execute one coupled FORWARD electron-transfer event from the current mixture.'
    event['function']['parameters']['properties']['direction']['enum'] = ['forward']
    finish = {'type': 'function', 'function': {
        'name': 'finish_forward', 'description': 'Finish and return the current forward product mixture.',
        'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False},
    }}
    return [event, finish]


def forward_prompt(mapped_state: str, history: Sequence[Mapping[str, Any]] = ()) -> str:
    inventory = build_inventory(_canonical_mapped(mapped_state))
    text = 'CURRENT MIXTURE: ' + inventory.visible_smiles + '\n' + inventory.prompt
    if history:
        text += ('\nACCEPTED FORWARD HISTORY (temporary names belong to their own past states):\n'
                 + json.dumps(list(history), ensure_ascii=False, separators=(',', ':')))
    return text


def execute_forward_call(state: str, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    # Canonicalize the private mapped graph BEFORE deriving mapless aliases.
    # Otherwise interchangeable reagent copies can swap aliases after replay.
    # No future state or target participates in this normalization.
    state = _canonical_mapped(state)
    if name == 'finish_forward':
        if arguments:
            raise ValueError('finish_forward takes no arguments')
        return {'ok': True, 'state_smiles': state, 'terminal': True}
    if name != 'apply_forward_electron_flow':
        raise ValueError(f'unknown forward tool: {name}')
    moves = compile_event_arguments(state, arguments, direction='forward')
    return verify_electron_step(state, moves)


def build_transition_row(line: str, *, reaction_id: str, decision_index: int,
                         history: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    parsed = parse_flower_line(line)
    if parsed is None:
        raise ValueError('malformed FlowER transition')
    source, destination, trajectory_id = parsed
    source, destination = _canonical_mapped(source), _canonical_mapped(destination)
    # Same atom-map namespace, not mapless equality: preserve the actual pairing.
    terminal = sides_equal(source, destination, ignore_maps=False)
    if terminal:
        name, arguments = 'finish_forward', {}
    else:
        bonds, lone_pairs, charges = _get_be_delta(source, destination)
        edge = ProofEdge('before', 'after', bonds=bonds, lone_pairs=lone_pairs, charges=charges)
        moves = infer_moves_from_edge(edge)
        # Some valid deltas have more than one arrow pairing. Use a replay-exact
        # canonical witness, with explicitly labelled BE_DELTA fallback if needed.
        result = verify_electron_step(source, moves)
        if not result.get('ok') or not sides_equal(result.get('state_smiles', ''), destination, ignore_maps=False):
            moves = ({'mode': 'BE_DELTA',
                      'bond_deltas': [{'atoms': [a, b], 'delta': d} for a, b, d in bonds],
                      'charge_actions': [{'atom_map': c.atom_map, 'q0': c.q0, 'q1': c.q1} for c in charges]},)
        name = 'apply_forward_electron_flow'
        arguments = render_event_arguments(source, moves, direction='forward')
    result = execute_forward_call(source, name, arguments)
    if not result.get('ok') or not sides_equal(result.get('state_smiles', ''), destination, ignore_maps=False):
        raise ValueError(f'FORWARD_REPLAY_MISMATCH:{trajectory_id}:{decision_index}:{result}')
    identifier = f'{reaction_id}:forward:{decision_index}'
    call_id = f'forward-{decision_index}'
    return {
        'id': identifier, 'reaction_id': reaction_id, 'tools': forward_tools(),
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': forward_prompt(source, history)},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}]},
            {'role': 'tool', 'tool_call_id': call_id, 'name': name,
             'content': json.dumps({'ok': True, 'current_state_smiles': build_inventory(result['state_smiles']).visible_smiles,
                                    'terminal': terminal}, ensure_ascii=False)},
        ],
        'metadata': {
            'decision_contract': 'forward_inventory_tool_decision_v1',
            'direction': 'forward', 'source': 'FlowER flower_new_dataset',
            'source_trajectory_id': trajectory_id, 'decision_index': decision_index,
            'executor_replayed': True, 'final_product_model_visible': False,
            'source_explicit_hydrogens_retained': True,
            'history_contract': 'recorded_forward_path_actions_v1',
            'forward_state_normalization': 'canonical_private_mapped_before_alias_v1',
            'be_delta_fallback': bool(arguments.get('bond_order_changes') or arguments.get('charge_changes')),
        },
    }


def check_adapter_isolation(reverse_dir: Path, forward_dir: Path, expected_sha: str) -> None:
    reverse_dir, forward_dir = reverse_dir.resolve(), forward_dir.resolve()
    if reverse_dir == forward_dir or reverse_dir in forward_dir.parents or forward_dir in reverse_dir.parents:
        raise ValueError('forward/reverse output overlap')
    digest = hashlib.sha256((reverse_dir / 'adapter_model.safetensors').read_bytes()).hexdigest()
    if digest != expected_sha:
        raise ValueError(f'reverse adapter SHA mismatch: {digest}')


def validate_forward_token_cache(audit: Mapping[str, Any], expected_rows: int) -> None:
    if int(audit['max_raw_input_tokens']) > int(audit['configured_max_length']):
        raise ValueError('forward conditioning would be cropped; increase context without dropping rows')
    if audit['n_windowed_source_rows'] or audit['n_rows'] != expected_rows or audit['n_training_windows'] != expected_rows:
        raise ValueError('forward token cache must retain one intact decision per original row')
