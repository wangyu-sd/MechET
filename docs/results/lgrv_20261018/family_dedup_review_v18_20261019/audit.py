"""Compare the whole pre/post bibliographic-fix population, never model samples."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from mechet.analysis.literature_patent_families import load_family_registry, _patent_id
from mechet.analysis.literature_sources import file_sha256


def rows(root, name):
    with gzip.open(root / name, 'rt') as handle:
        return list(map(json.loads, handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-root', type=Path, required=True)
    parser.add_argument('--current-root', type=Path, required=True)
    parser.add_argument('--family-registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifests = []
    for root in [args.previous_root, args.current_root]:
        manifest = json.loads((root / 'analysis_manifest.json').read_bytes())
        assert manifest['complete']
        for name, row in manifest['output_files'].items():
            assert file_sha256(root / name) == row['sha256'], name
        manifests.append(manifest)
    old, new = args.previous_root, args.current_root
    unchanged = ['unique_reaction_queries_enriched.jsonl.gz', 'retrieval_query_contexts.jsonl.gz',
                 'generated_reaction_steps.jsonl.gz', 'program_reaction_links.jsonl.gz',
                 'route_evidence.jsonl.gz']
    for name in unchanged:
        assert rows(old, name) == rows(new, name), name
    old_e = {r['reaction_key']: r for r in rows(old, 'reaction_evidence.jsonl.gz')}
    new_e = {r['reaction_key']: r for r in rows(new, 'reaction_evidence.jsonl.gz')}
    assert set(old_e) == set(new_e) and len(new_e) == 18210
    for key, record in new_e.items():
        assert {k:v for k,v in record.items() if k != 'source_candidates'} == {
            k:v for k,v in old_e[key].items() if k != 'source_candidates'}, key
    assert Counter(r['tier'] for r in new_e.values()) == {'A':47, 'B':26, 'C':18137}
    old_metrics = json.loads((old / 'metrics.json').read_bytes())
    new_metrics = json.loads((new / 'metrics.json').read_bytes())
    for name in ['runs', 'composition', 'paired', 'paired_disconnections', 'efficiency']:
        assert new_metrics[name] == old_metrics[name], name
    resolver = load_family_registry(args.family_registry)
    grouped = defaultdict(list)
    for row in rows(new, 'literature_hits.jsonl.gz'):
        grouped[row['query_reaction_key']].append(row)
    old_grouped = defaultdict(list)
    for row in rows(old, 'literature_hits.jsonl.gz'):
        old_grouped[row['query_reaction_key']].append(row)
    assert set(grouped) == set(old_grouped) and len(grouped) == 165
    changed = []; unresolved = set()
    for key, hits in grouped.items():
        assert len(hits) <= 5
        family_keys = [resolver.resolve(h) for h in hits]
        assert len(family_keys) == len(set(family_keys)), key
        assert [h['source_family_key'] for h in hits] == family_keys
        for hit in hits:
            if hit.get('patent_id') and _patent_id(hit['patent_id']) not in resolver.families:
                unresolved.add(hit['patent_id'])
        before = [h['source_id'] for h in old_grouped[key]]
        after = [h['source_id'] for h in hits]
        if before != after:
            changed.append({'query_reaction_key':key, 'before_source_ids':before,
                            'after_source_ids':after,
                            'retained_representation_ids':[h['source_representation_ids'] for h in hits]})
    # Missing identity must be visible rather than claimed as independent.
    result = {'schema':'lgrv_whole_patent_family_correction_audit_v1', 'complete':True,
              'queries':len(new_e), 'conditions':len(new_metrics['runs']),
              'indexed_queries':len(grouped), 'old_candidate_hits':sum(map(len, old_grouped.values())),
              'current_candidate_hits':sum(map(len, grouped.values())),
              'changed_candidate_lists':changed, 'unchanged_scientific_outputs':unchanged,
              'all_reaction_grades_and_selected_sources_unchanged':True,
              'all_six_run_route_composition_paired_efficiency_results_unchanged':True,
              'retained_patents_without_resolved_family':sorted(unresolved),
              'all_current_query_family_keys_distinct':True,
              'previous_analysis_manifest_sha256':file_sha256(old / 'analysis_manifest.json'),
              'current_analysis_manifest_sha256':file_sha256(new / 'analysis_manifest.json'),
              'family_registry_sha256':file_sha256(args.family_registry),
              'family_overlay':{k:v for k,v in resolver.manifest.items() if k != 'documents'},
              'producer_sha256':file_sha256(__file__), 'new_model_inference':False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['changed_candidate_lists', 'family_overlay', 'unchanged_scientific_outputs']}), flush=True)


if __name__ == '__main__':
    main()
