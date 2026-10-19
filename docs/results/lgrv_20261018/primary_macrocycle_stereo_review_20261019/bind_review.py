"""Bind an explicitly read stereo-limit note, without assigning a new tier."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_sources import file_sha256, normalize_primary_text


def bind(spec_path, query_path, cache_root):
    spec = json.loads(spec_path.read_bytes())
    key = spec['query_reaction_key']
    chemistry = make_reaction(spec['source_constitution_only_product'],
                              spec['source_constitution_only_precursors'])
    assert chemistry['reaction_key'] == key
    with gzip.open(query_path, 'rt') as handle:
        queries = [json.loads(line) for line in handle]
    selected = [r for r in queries if r['reaction_key'] == key]
    assert len(selected) == 1
    # Complete query identity is preserved. A matching constitution-only key
    # does not supply the absent relative stereochemical constraint.
    query = selected[0]
    assert make_reaction(query['product_raw'], query['structural_components'])['reaction_key'] == key
    assert '@' not in query['product_raw']
    quote = spec['checked_passage']
    assert len(quote.split()) <= 25
    bindings = []
    for source in spec['source_documents']:
        receipt_path = cache_root/(hashlib.sha256(source['source_url'].encode()).hexdigest()+'.json')
        receipt = json.loads(receipt_path.read_bytes())
        assert receipt['source_url'] == source['source_url']
        assert receipt['document_sha256'] == source['document_sha256']
        raw_path, text_path = Path(receipt['raw_document_path']), Path(receipt['document_path'])
        assert file_sha256(raw_path) == receipt['raw_document_sha256']
        assert file_sha256(text_path) == receipt['document_sha256']
        document = text_path.read_text(encoding='utf-8')
        assert receipt['text_normalization'] == 'html_data_join_unicode_whitespace_v1'
        assert normalize_primary_text(raw_path.read_bytes(), receipt.get('source_encoding', 'utf-8')) == document
        assert document.count(quote) == 1
        bindings.append({**source, 'source_receipt_path': str(receipt_path.resolve()),
            'source_receipt_sha256': file_sha256(receipt_path),
            **{k: receipt[k] for k in ['raw_document_path', 'raw_document_sha256', 'document_path']},
            'normalized_passage_offset': document.index(quote),
            'checked_passage_sha256': hashlib.sha256(quote.encode()).hexdigest(),
            'source_paragraph_checked': True, 'full_stereospecific_identity_verified': False})
    image_bindings = []
    for figure in spec["source_images"]:
        selected_sources = [s for s in bindings if s["source_url"] == figure["linked_from_source_url"]]
        assert len(selected_sources) == 1
        raw_document = Path(selected_sources[0]["raw_document_path"]).read_text()
        assert figure["source_url"] in raw_document and figure["html_figure_id"] in raw_document
        path = Path(figure["path"])
        assert path.read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a"))
        assert file_sha256(path) == figure["sha256"]
        image_bindings.append({**figure, "linked_raw_document_sha256": selected_sources[0]["raw_document_sha256"]})
    return {**spec, "source_image_bindings": image_bindings, 'schema': 'lgrv_checked_stereo_identity_limit_bound_v1',
            'source_bindings': bindings, 'query_reaction_smiles': query['reaction_smiles'],
            'source_query_path': str(query_path), 'source_query_sha256': file_sha256(query_path),
            'review_spec_sha256': file_sha256(spec_path), 'producer_sha256': file_sha256(__file__),
            'evidence_grades_changed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['review-spec', 'query-view', 'cache-root', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    review = bind(args.review_spec, args.query_view, args.cache_root)
    with args.output.open('x') as handle:
        handle.write(json.dumps(review, indent=2)+'\n')
    print(json.dumps({'bound_primary_paragraphs': len(review['source_bindings']),
                      'tier_retained': review['query_evidence_tier_retained'],
                      'output_sha256': file_sha256(args.output)}), flush=True)
