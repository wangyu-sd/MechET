from types import SimpleNamespace

import pytest

from scripts.audit_reliable_uspto31k_transfer import audit_beam_tree_row
from scripts.run_natural_language_value_search import Action, search_unlabeled


def test_beam_audit_accepts_complete_tree_and_catches_missing_branch():
    class Runtime:
        pointer_invalid_handles = 0

        def proposals(self, node, **_kwargs):
            if not node.actions:
                return [
                    Action('import_fragments', {'fragments': [{'smiles': 'O', 'count': 1}]}, 'O', -0.1, 1),
                    Action('import_fragments', {'fragments': [{'smiles': 'N', 'count': 1}]}, 'N', -0.2, 1),
                ]
            return [Action('finish_trace', {}, 'finish', -0.1, 1)]

        def values(self, _target, states, **_kwargs):
            return [0.0] * len(states)

    args = SimpleNamespace(
        max_decisions=2, branching=2, max_new_tokens=8, compact_history=False,
        max_imports=2, reject_target_retained_finish=False,
        early_beam=2, late_beam=2, early_depth=2,
        value_weight=0.0, pointer_weight=0.0, product_only_remap=True,
    )
    searched = search_unlabeled(Runtime(), 'C', args)
    row = {
        'search_config': {'mode': 'deterministic_beam'},
        'search_tree': searched['search_tree'],
        'attempts': searched['attempts'],
        'terminal_node_ids': [node.node_id for node in searched['terminals']],
        'n_terminals': len(searched['terminals']),
        'top_node_id': searched['top'].node_id,
        'top_terminal': searched['top'].terminal,
    }
    audit_beam_tree_row(row)
    row['search_tree'] = row['search_tree'][:-1]
    with pytest.raises(ValueError, match='child node|terminal branches|top candidate'):
        audit_beam_tree_row(row)
