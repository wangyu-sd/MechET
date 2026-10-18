"""Freeze actual complete generated reactions, never reference precursors.

Occurrence identity retains report/pointer and route linkage. Structural keys
deduplicate retrieval, not targets, programs or routes. Source bytes are checked
before any chemistry operation so a growing run cannot change the snapshot.
"""
from collections import Counter
import csv
import json
from pathlib import Path

from .literature_reaction_keys import canonical_components, make_reaction, stable_hash
from .literature_sources import file_sha256


def resolve_tree_artifact(report, source):
    """Resolve the two actual serializers, never guess a rank from row order."""
    path = Path(report['tree_artifact'])
    if path.is_absolute():
        candidates = [path] if path.is_file() else []
    elif 'report_row_index' in source:
        candidates = list(Path(source['report_path']).parent.glob('rank*/'+path.as_posix()))
    else:
        candidate = Path(source['report_path']).parent.parent/path
        candidates = [candidate] if candidate.is_file() else []
    if len(candidates)!=1:
        raise ValueError('missing or ambiguous recorded tree artifact: '+str(path))
    return candidates[0]


def frozen_reports(root, csv_manifest):
    root = Path(root).resolve()
    result = []
    ids = set()
    with Path(csv_manifest).open() as handle:
        for row in csv.DictReader(handle):
            path = (root/row['report_relative_path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('report path outside frozen root')
            sha = file_sha256(path)
            if sha != row['report_sha256']:
                raise ValueError('frozen report checksum mismatch: '+str(path))
            report = json.loads(path.read_bytes())
            if report['id'] != row['stable_id'] or report['id'] in ids:
                raise ValueError('snapshot target identity differs or repeated')
            ids.add(report['id'])
            result.append((report,{'report_path':str(path),'report_sha256':sha}))
    return result


def _record(product, precursors, *, report, run_id, source, pointer, program=None, terminal_id=None):
    program = program or {}
    actions = program.get('actions',[])
    mapped_product = actions[0].get('state_before') if actions else None
    auxiliary = program.get('auxiliary_fragments',[])
    reaction = make_reaction(product,precursors,auxiliary=auxiliary,
        mapped_product=mapped_product, mapped_precursors=program.get('structural_terminal_mapped'))
    return {**reaction,'run_id':run_id,'target_id':report['id'],'cohort':report.get('cohort'),
            'reaction_product':product,'structural_precursors_raw':precursors,
            'reaction_edge_id':stable_hash([product,precursors]),
            'occurrence_id':stable_hash([run_id,report['id'],source['report_sha256'],pointer]),
            'source':source,'source_pointer':pointer,'terminal_record_id':terminal_id or program.get('terminal_context'),
            'electron_program_source_pointer':pointer+'/program' if program else None,
            'formal_status':'recorded_executor_accepted_complete_reaction',
            'chemical_evidence_status':'unresolved'}


def single_reactions(report, *, run_id, source):
    for i, candidate in enumerate(report.get('ranked_candidates',[])):
        # These are ALL saved candidate records, not a selected rank prefix.
        structural = candidate['structural_precursor']
        reaction = _record(report['target'],structural.split('.'),report=report,run_id=run_id,source=source,
            pointer=f'/ranked_candidates/{i}',terminal_id=candidate.get('context'))
        reaction['structural_precursor_mapped'] = candidate.get('structural_precursor_mapped')
        full = candidate.get('full_mixture')
        if full:
            remainder = Counter(canonical_components(full))-Counter(reaction['structural_components'])
            reaction['auxiliary_components'] = sorted(remainder.elements())
            reaction['auxiliary_role_basis'] = 'recorded_structural_projection_full_mixture_difference'
        yield reaction


def route_reactions(report, *, run_id, source):
    known = {}
    for i, edge in enumerate(report.get('complete_reaction_edges',[])):
        known[stable_hash([edge['product'],sorted(edge['precursors'])])] = (edge,f'/complete_reaction_edges/{i}/program')
    for r, route in enumerate(report.get('enumerated_stock_connected_routes',[])):
        for i, edge in enumerate(route['complete_reaction_edges']):
            known.setdefault(stable_hash([edge['product'],sorted(edge['precursors'])]),
                (edge,f'/enumerated_stock_connected_routes/{r}/complete_reaction_edges/{i}/program'))
    for i, edge in enumerate(report.get('generated_reaction_graph',[])):
        details, program_pointer = known.get(stable_hash([edge['product'],sorted(edge['precursors'])]),(edge,None))
        reaction = _record(edge['product'],edge['precursors'],report=report,run_id=run_id,source=source,
            pointer=f'/generated_reaction_graph/{i}',program=details.get('program'))
        # Program details may come from a route witness, not the graph summary.
        if details.get('program'):
            reaction['electron_program_source_pointer'] = program_pointer
        yield reaction
