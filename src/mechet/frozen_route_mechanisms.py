"""Exact frozen-record correspondence, never electronic-program inference.

Callers must authenticate the complete source artifact and audit manifest
before supplying records. Matching removes maps/component order only, retaining
charge, isotope, stereo and precursor multiplicity. Source programs are
executor-verified corpus witnesses, not independent experimental certificates.
"""
from __future__ import annotations
from copy import deepcopy
from .frozen_route_assets import normalize_stock_smiles


def reaction_chemistry_key(product: str, precursors: list[str]) -> tuple:
    # Outer precursor entries can themselves be salt mixtures. Preserve their
    # component multiplicities for correspondence; this is NOT a stock lookup.
    components=[part for precursor in precursors for part in precursor.split('.')]
    return normalize_stock_smiles(product),tuple(sorted(normalize_stock_smiles(x) for x in components))


def resolve_reference_mechanism(reference: dict, records: list[dict]) -> dict:
    result=deepcopy(reference)
    result.update(program_signature=None,primitive_signatures=None,
                  electronic_programs=[],matched_source_ids=[],matched_source_splits=[],
                  signature_version=None,action_vocabulary_covered=None)
    if reference.get('mapping_valid') is not True:
        result.update(annotation_status='no_verified_match',
                      mechanism_match_status='REFERENCE_MAPPING_UNAVAILABLE')
        return result
    key=reaction_chemistry_key(reference['source_product_smiles'],reference['source_precursor_smiles'])
    matches=[row for row in records if reaction_chemistry_key(row['target_smiles'],
             [row['structural_precursor']])==key]
    if not matches:
        result.update(annotation_status='no_verified_match',mechanism_match_status='NO_EXACT_CORPUS_CHEMISTRY')
        return result
    matches.sort(key=lambda row:(row['source_split'],row['source_id']))
    result.update(matched_source_ids=sorted({row['source_id'] for row in matches}),
                  matched_source_splits=sorted({row['source_split'] for row in matches}))
    supported=[row for row in matches if row.get('executor_replayed') is True
               and row.get('ordinary_source_sink_program') is True
               and row.get('program_signature') and row.get('primitive_signatures')
               and row.get('electronic_program',{}).get('steps')
               and row.get('signature_version')=='source_to_sink_execution_moves_v1']
    # Do not choose a convenient supported witness while hiding conflicting or
    # unsupported source records for precisely the same reference chemistry.
    if len(supported)!=len(matches):
        result.update(annotation_status='unsupported_mechanism',
                      mechanism_match_status='UNVERIFIED_OR_NON_SOURCE_SINK_CORPUS_RECORD')
        return result
    identities={(row['program_signature'],tuple(sorted(row['primitive_signatures']))) for row in supported}
    if len(identities)!=1:
        result.update(annotation_status='unsupported_mechanism',mechanism_match_status='AMBIGUOUS_RECORDED_PROGRAMS')
        return result
    first=supported[0]
    result.update(annotation_status='verified_mechanism',mechanism_match_status='EXACT_FROZEN_CORPUS_CORRESPONDENCE',
                  program_signature=first['program_signature'],primitive_signatures=list(first['primitive_signatures']),
                  signature_version=first['signature_version'],action_vocabulary_covered=True,
                  electronic_programs=[deepcopy(row['electronic_program']) for row in supported],
                  mechanism_evidence='executor_verified_frozen_corpus_witness_not_experimental_certificate')
    return result
