"""Draw performed sources separately from all six new generated analogues."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import textwrap
import yaml

from mechet.analysis.figure_qa import inspect_figure
from mechet.analysis.literature_evidence import assess
from mechet.analysis.literature_reaction_keys import canonical_components, make_reaction
from mechet.analysis.literature_sources import file_sha256
from scripts.render_nmi_tree_figures import molecule_svg, embed, label, svg_document, export


BASE = Path('docs/results/lgrv_20261018')
REVIEW = BASE / 'grouped_primary_review_20261019/source_family_review.json'
REGISTRY = Path('configs/analysis/lgrv_primary_sources_20261019_v8.json')
QUERY_ROOT = Path('/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/paper/nmi_literature_evidence/source_checked_20261019_v13')
STYLE = Path('configs/figures/mechet_nmi_tree_style_v1.yaml')
OUT = BASE / 'analogue_source_query_gallery_20261019_v14'


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return list(map(json.loads, stream))


def main():
    manifest_path = OUT / 'figure_manifest.json'
    if manifest_path.exists() or (OUT / 'figures').exists():
        raise FileExistsError('do not replace a previously rendered plate')
    review = json.loads(REVIEW.read_bytes())
    registry = json.loads(REGISTRY.read_bytes())
    if registry['review_spec_sha256'] != review['review_spec_sha256']:
        raise ValueError('source registry and reviewed pairs differ')
    qpath = QUERY_ROOT / 'unique_reaction_queries_enriched.jsonl.gz'
    cpath = QUERY_ROOT / 'retrieval_query_contexts.jsonl.gz'
    queries = {r['reaction_key']: r for r in read_rows(qpath)}
    contexts = {r['reaction_key']: r for r in read_rows(cpath)}
    sources = {r['source_id']: r for r in review['sources']}
    grouped = defaultdict(list)
    for r in review['reviews']:
        if r['assessed_tier'] == 'B':
            grouped[r['source_id']].append(r)
    if sum(map(len, grouped.values())) != len(registry['records']) or len(registry['records']) != 6:
        raise ValueError('draw every new supported pair, not a preferred method subset')
    style = yaml.safe_load(STYLE.read_text())
    width = style['format']['full_width_mm'] * 72 / 25.4
    height = 53 + 107 * 3
    if height * 25.4 / 72 > style['format']['max_height_mm']:
        raise ValueError('gallery exceeds fixed publication height')
    captions = {
        'US7855213B2': 'Pd(dppf)Cl2/Cs2CO3; DME/H2O/EtOH6:3:1; Ar,microwave130 C15min. Example85:16% reported yield.',
        'US8247576B2': 'Pd[P(tBu)3]2/K2CO3; toluene/EtOH/H2O; N2,microwave135 C15min. Example55:individual yield unstated.'
    }
    figures = {}
    included = []
    for source_id, pairs in sorted(grouped.items()):
        source = sources[source_id]
        for start in range(0, len(pairs), 2):
            selected = sorted(pairs, key=lambda r: r['query_reaction_key'])[start:start + 2]
            if len(selected) != 2:
                raise ValueError('this fixed three-row recipe requires two reviewed queries per plate')
            body = label(7, 13, 'Performed source and generated analogue disconnections', size=7, weight='bold')
            body += label(7, 25, 'Only the source row is an experiment; generated query rows are Tier B compatibility inferences.', size=6)
            display = [(source['source_precursors'], source['source_product'],
                        source['patent_id'] + ' | performed source | ' + ('Example85' if '7855213' in source_id else 'Example55'),
                        captions[source['patent_id']])]
            pair_data = []
            for pair in selected:
                key = pair['query_reaction_key']
                query = {**queries[key], 'stereo_family_context': contexts[key]}
                evidence = assess(query, [], registry['records'])
                if evidence['tier'] != 'B' or evidence['physical_mechanism_verified'] is not False:
                    raise ValueError('generated query is not a checked analogue')
                if make_reaction(query['product_raw'], query['structural_components'])['reaction_key'] != key:
                    raise ValueError('generated query identity differs')
                display.append((query['structural_components'], query['product_raw'],
                    'Generated query ' + key[:12] + ' | Tier B analogue, not a performed experiment',
                    'Specific substrate conversion,yield,selectivity and auxiliary-mixture match are unmeasured.'))
                pair_data.append({**pair, 'evidence': evidence, 'raw_product': query['product_raw'],
                                  'structural_precursors': query['structural_components']})
                included.append(key)
            for i, (precursors, product, title, condition) in enumerate(display):
                y = 40 + i * 107
                body += label(7, y, title, size=6.5, weight='bold')
                parts = [c for p in precursors for c in canonical_components(p)]
                cell = width * .49 / len(parts)
                for j, smiles in enumerate(parts):
                    drawing, _ = molecule_svg(smiles, [], style, uid=f'pair-{i}-r{j}', width=cell - 9, height=61)
                    body += embed(drawing, 7 + j * cell, y + 6, cell - 9, 61)
                    if j < len(parts) - 1:
                        body += label((j + 1) * cell - 1, y + 38, '+', size=7)
                x = width * .51
                body += f'<path d="M{x} {y+36} h24 l-4 -3 m4 3 l-4 3" stroke="{style["palette"]["ink"]}" fill="none"/>'
                drawing, _ = molecule_svg('.'.join(canonical_components(product)), [], style,
                    uid=f'pair-{i}-p', width=width * .41, height=61)
                body += embed(drawing, width * .58, y + 6, width * .41, 61)
                for j, line in enumerate(textwrap.wrap(condition, 120)):
                    if j >= 3:
                        raise ValueError('caption needs a larger dedicated plate')
                    body += label(7, y + 78 + j * 7, line, size=5.5)
            body += label(7, height - 6, 'Forward-oriented net schemes; no source validation of individual symbolic electron arrows or a complete route.', size=5.7)
            slug = 'source_query_pair_' + source['patent_id'].lower() + '_' + str(start // 2 + 1)
            files = export(svg_document(body, style, height_pt=height), OUT / 'figures' / slug, style)
            qa = inspect_figure(files, style)
            if not qa['machine_checks_passed']:
                raise ValueError('figure QA failed: ' + str(qa['problems']))
            figures[slug] = dict(files=files, qa=qa, source=source, query_pairs=pair_data)
    if len(included) != len(set(included)) or len(included) != 6:
        raise ValueError('supporting pair omitted or duplicated')
    producers = [Path(__file__), Path('src/mechet/analysis/literature_evidence.py'),
                 Path('scripts/render_nmi_tree_figures.py'), Path('src/mechet/analysis/figure_qa.py')]
    manifest = dict(schema='lgrv_source_query_analogue_gallery_v1', figures=figures,
        all_new_supported_pairs_included=True, supported_query_pairs=included,
        source_files={str(p): file_sha256(p) for p in [REVIEW, REGISTRY, qpath, cpath, STYLE]},
        producer_files={str(p): file_sha256(p) for p in producers},
        visual_inspection_status='pending', new_model_inference=False,
        physical_mechanism_verified=False,
        population_result='not computed by this renderer; see separately sealed full v14 analysis',
        interpretation='Performed source and generated analogue queries are distinct; TierB is not exact experimental query validation.')
    with manifest_path.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(plates=len(figures), supported_pairs=len(included), output_root=str(OUT))), flush=True)


if __name__ == '__main__':
    main()
