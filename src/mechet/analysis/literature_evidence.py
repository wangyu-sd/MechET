"""Primary-source grading, separate from indexed or formal-execution evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .literature_reaction_keys import make_reaction


def _passage_checked(source):
    text = source.get('checked_passage', '')
    try:
        receipt_path=Path(source['source_receipt_path'])
        receipt=json.loads(receipt_path.read_bytes())
        if receipt_path.name!=hashlib.sha256(source['source_url'].encode()).hexdigest()+'.json':
            return False
        for key in ['source_url','raw_document_path','raw_document_sha256','document_path','document_sha256']:
            if not source.get(key) or source[key]!=receipt.get(key):
                return False
        raw=Path(receipt['raw_document_path']).read_bytes()
        document=Path(receipt['document_path']).read_text()
        if hashlib.sha256(raw).hexdigest()!=receipt['raw_document_sha256']:
            return False
        if hashlib.sha256(document.encode()).hexdigest()!=receipt['document_sha256']:
            return False
        from .literature_sources import normalize_primary_text
        if receipt.get('text_normalization')!='html_data_join_unicode_whitespace_v1':
            return False
        if normalize_primary_text(raw,receipt.get('source_encoding','utf-8'))!=document:
            return False
        if source.get('document_text',document)!=document:
            return False
    except (OSError,UnicodeError,KeyError,ValueError,TypeError,LookupError):
        return False
    return bool(source.get('source_url') and source.get('locator') and source.get('document_sha256')
                and len(source['document_sha256']) == 64 and source.get('checker')
                and source.get('passage_checked') is True and text and text in document
                and hashlib.sha256(text.encode()).hexdigest() == source.get('checked_passage_sha256'))


def assess(reaction, hits, primary_sources=()):
    base = {'reaction_key':reaction['reaction_key'], 'tier':'C', 'supported':False,
            'retrieval_status':'indexed_candidate' if hits else 'unresolved',
            'verification_status':'machine_structural_match' if hits else 'unresolved',
            'source_candidates':[h['source_id'] for h in hits], 'selected_source':None,
            'physical_mechanism_verified':False, 'training_source_overlap':'unknown',
            'reason':'no_checked_structurally_reconciled_performed_source'}
    supported = []
    for source in primary_sources:
        if not _passage_checked(source) or source.get('performed_example') is not True:
            continue
        try:
            candidate = make_reaction(source['source_product'], source['source_precursors'])
        except (KeyError, ValueError):
            continue
        if candidate['reaction_key'] == reaction['reaction_key']:
            supported.append(('A',source))
        elif (reaction.get('reaction_family_key') and candidate.get('reaction_family_key') == reaction['reaction_family_key']
              and source.get('analogue_compatibility_checked') is True
              and source.get('substrate_differences') and source.get('functional_group_compatibility')
              and source.get('stereo_compatibility_checked') is True):
            supported.append(('B',source))
    if supported:
        tier, source = sorted(supported,key=lambda item:(item[0],item[1]['source_id']))[0]
        return {**base, 'tier':tier, 'supported':True, 'selected_source':source['source_id'],
                'retrieval_status':'experimental_example_checked', 'verification_status':'source_passage_checked',
                'reason':'exact_performed_structural_reaction' if tier=='A' else 'checked_experimental_analogue',
                'source_url':source['source_url'], 'locator':source['locator'],
                'document_sha256':source['document_sha256'],
                'raw_document_sha256':source['raw_document_sha256'],
                'conditions_match':source.get('conditions_match','unknown')}
    # A contradiction needs an explicit checked claim about THIS reaction, not
    # missing retrieval, another endpoint, or an unrelated hazardous reagent.
    for source in primary_sources:
        if (_passage_checked(source) and source.get('contradicted_reaction_key') == reaction['reaction_key']
                and source.get('contradiction_class') and source.get('contradiction_evidence')):
            return {**base,'tier':'D','reason':'documented_direct_conflict',
                    'selected_source':source['source_id'],'verification_status':'source_passage_checked',
                    'retrieval_status':'source_retrieved','source_url':source['source_url'],
                    'locator':source['locator'],'contradiction_class':source['contradiction_class']}
    return base
