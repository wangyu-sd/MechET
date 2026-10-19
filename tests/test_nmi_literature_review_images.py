"""A source image must bind actual bytes and the inspected primary page."""
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_sources import snapshot_primary_document


@pytest.mark.parametrize('change', ['checksum', 'image_url', 'figure_id', 'parent_source', 'png_signature'])
def test_source_image_corruption_cannot_be_certified_as_an_inspected_patent_figure(tmp_path, change):
    # Removing the binder's image-byte/link validation must make this test fail.
    quote = 'Example 55 Preparation of Compound 18 Prepared by way of method I'
    image_url = 'https://example.test/source.png'
    raw = tmp_path / 'primary.html'
    raw.write_text('<p>' + quote + '</p><chemistry id="CHEM-fixture"><img src="' + image_url + '"></chemistry>')
    cache = tmp_path / 'cache'
    receipt = snapshot_primary_document(raw.as_uri(), cache)
    image = tmp_path / 'source.png'
    image.write_bytes(bytes.fromhex('89504e470d0a1a0a') + b'fixture bytes; not a scientific image')
    figure = dict(source_url=image_url, path=str(image),
                  sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                  linked_from_source_url=raw.as_uri(), html_figure_id='CHEM-fixture')
    if change == 'checksum':
        figure['sha256'] = '0' * 64
    elif change == 'image_url':
        figure['source_url'] = 'https://example.test/unrelated.png'
    elif change == 'figure_id':
        figure['html_figure_id'] = 'CHEM-not-inspected'
    elif change == 'parent_source':
        figure['linked_from_source_url'] = 'https://example.test/unrelated-primary'
    else:
        image.write_bytes(b'not a PNG')
        figure['sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
    chemistry = make_reaction('CO', ['CBr', '[OH-]'])
    query = tmp_path / 'query.jsonl.gz'
    import gzip
    with gzip.open(query, 'wt') as stream:
        stream.write(json.dumps(chemistry) + '\n')
    spec = dict(query_reaction_key=chemistry['reaction_key'],
                source_constitution_only_product='CO',
                source_constitution_only_precursors=['CBr', '[OH-]'],
                checked_passage=quote, source_images=[figure],
                source_documents=[dict(source_url=raw.as_uri(), document_sha256=receipt['document_sha256'])])
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec))
    binder_path = Path(__file__).resolve().parents[1] / 'docs/results/lgrv_20261018/primary_macrocycle_stereo_review_20261019/bind_review.py'
    bind = runpy.run_path(str(binder_path))['bind']
    with pytest.raises(AssertionError):
        bind(spec_path, query, cache)
