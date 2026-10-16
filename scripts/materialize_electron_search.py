#!/usr/bin/env python3
"""Read an unfinished incremental graph WITHOUT model/tool calls or sealing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))
from mechet.shared_electron_search import SearchJournal, read_record, fingerprint
from scripts.infer_dynamic_electron_trees import dump, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True, help='one hashed per-case journal directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stock', type=Path, help='original canonical stock for route graphs')
    parser.add_argument('--mcts-checkpoint',type=Path,help='optional exact per-case MCTS checkpoint')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new path, do not overwrite evidence')
    contract = read_record(args.journal / 'contract.json')
    journal = SearchJournal(args.journal.parent, contract['case_id'], contract)
    if journal.root.resolve() != args.journal.resolve():
        parser.error('case ID does not match journal directory')
    stock = None
    if args.stock is not None:
        expected = contract.get('case_stock_sha')
        if expected is None or sha256(args.stock) != expected:
            parser.error('stock SHA differs from this CASE cohort stock contract')
        with args.stock.open() as handle:
            stock = {line.strip() for line in handle if line.strip()}
    controller = journal.restore(stock=stock)
    graph = controller.to_dict()
    checkpoint=args.mcts_checkpoint or args.journal.parent.parent/'mcts'/(fingerprint(contract['case_id'])+'.json')
    if checkpoint.exists():
        saved=read_record(checkpoint)
        planning=saved.get('planning_contract',{})
        shared=Path(saved['metrics']['shared_statistics_root'])
        if (saved['case_id']!=contract['case_id'] or planning.get('stock_sha')!=contract.get('case_stock_sha')
                or shared.name!=fingerprint(planning)
                or (contract.get('shared_cache_namespace') is not None and
                    planning.get('policy_executor_namespace')!=contract['shared_cache_namespace'])):
            parser.error('MCTS checkpoint differs from case/stock/policy contract')
        graph['mcts']=saved['metrics']
        graph['mcts_checkpoint']=saved
        graph['mcts_graph']={sid:read_record(shared/'nodes'/sid[:2]/(sid+'.json'))
                            for sid in saved['seen_states']}
    graph['persistence'] = {'search_complete': False, 'journal_commits': journal.sequence,
        'journal_path': str(args.journal.resolve()), 'contract': contract,
        'endpoint_labels_used': False,
        'note': 'observed partial graph, not a complete full-cohort evaluation'}
    if hasattr(controller, '_certificate'):
        graph['observed_best_route'] = controller._certificate()
    dump(args.output, graph)
    print(json.dumps({'phase': 'materialized-partial-search', 'case_id': controller.case_id,
                      'journal_commits': journal.sequence, 'output': str(args.output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
