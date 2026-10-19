"""Bind explicitly reviewed preparations to immutable cached primary bytes.

This tool does not discover evidence, assign chemistry compatibility, or turn
an extracted reaction into an experiment. A reviewer supplies named structures,
the exact checked passage, locator and preparation-specific limitations first.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .literature_evidence import _passage_checked
from .literature_reaction_keys import make_reaction
from .literature_sources import file_sha256, normalize_primary_text


def bind_preparation(spec, cache_root):
    required = ('source_id', 'source_url', 'document_sha256', 'checked_passage',
                'locator', 'source_product', 'source_precursors', 'structural_reconciliation',
                'checker', 'expected_exact_reaction_key')
    if any(not spec.get(key) for key in required):
        raise ValueError('checked preparation is missing source, chemistry or review fields')
    if spec.get('performed_example') is not True or spec.get('passage_checked') is not True:
        raise ValueError('preparation must already have been explicitly reviewed')
    if len(spec['checked_passage'].split()) > 25:
        raise ValueError('use a short primary quotation, not the whole preparation')
    receipt_path = Path(cache_root)/(hashlib.sha256(spec['source_url'].encode()).hexdigest()+'.json')
    receipt = json.loads(receipt_path.read_bytes())
    if receipt['source_url'] != spec['source_url'] or receipt['document_sha256'] != spec['document_sha256']:
        raise ValueError('reviewed source identity changed')
    raw, text = Path(receipt['raw_document_path']), Path(receipt['document_path'])
    if (file_sha256(raw) != receipt['raw_document_sha256'] or
            file_sha256(text) != receipt['document_sha256']):
        raise ValueError('cached source checksum differs')
    document = text.read_text(encoding='utf-8')
    if (receipt.get('text_normalization') != 'html_data_join_unicode_whitespace_v1' or
            normalize_primary_text(raw.read_bytes(), receipt.get('source_encoding', 'utf-8')) != document):
        raise ValueError('primary text is not the recorded normalization of original bytes')
    quote = spec['checked_passage']
    if document.count(quote) != 1:
        raise ValueError('checked passage is absent or not unique; select an unambiguous short witness')
    chemistry = make_reaction(spec['source_product'], spec['source_precursors'])
    if chemistry['reaction_key'] != spec['expected_exact_reaction_key']:
        raise ValueError('named source structures differ from reviewed complete reaction')
    record = {**spec, 'source_receipt_path': str(receipt_path.resolve()),
        **{key: receipt[key] for key in ('raw_document_path', 'raw_document_sha256',
                                         'document_path', 'document_sha256')},
        'checked_passage_sha256': hashlib.sha256(quote.encode()).hexdigest(),
        'normalized_passage_offset': document.index(quote),
        'source_human_reviewed': False, 'whole_route_supported': False,
        'physical_mechanism_verified': False, 'training_source_overlap': 'unknown',
        'conditions_match': spec.get('conditions_match', 'not_assessed_against_actor_auxiliary_mixture')}
    if not _passage_checked(record):
        raise ValueError('bound preparation failed the actual source-evidence verifier')
    return record


def build_registry(spec_path, cache_root, output_path):
    spec_path, output_path = Path(spec_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    specification = json.loads(spec_path.read_bytes())
    records = [bind_preparation(s, cache_root) for s in specification['records']]
    if not records or len({r['source_id'] for r in records}) != len(records):
        raise ValueError('preparation source identities must be nonempty and unique')
    result = {'schema': 'lgrv_primary_source_registry_v1',
        'scope': specification.get('scope', 'Explicitly checked performed primary preparations.'),
        'review_spec_sha256': file_sha256(spec_path), 'producer_sha256': file_sha256(__file__),
        'records': records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-spec', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_registry(args.review_spec, args.cache_root, args.output)
    print(json.dumps({'bound_checked_preparations': len(result['records']),
                      'registry_sha256': file_sha256(args.output)}), flush=True)


if __name__ == '__main__':
    main()
