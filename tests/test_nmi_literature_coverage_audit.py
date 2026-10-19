"""Coverage must separate retrieval, document acquisition and checked chemistry."""
import copy

import pytest


def inputs():
    from mechet.analysis.literature_reaction_keys import make_reaction
    queries = [make_reaction(p, [r]) for p, r in
               [('CO', 'C.O'), ('CCO', 'CC.O'), ('CCCO', 'CCC.O'),
                ('CCCCO', 'CCCC.O'), ('CCCCCO', 'CCCCC.O')]]
    contexts = [{'reaction_key': q['reaction_key'], 'status': 'available'} for q in queries]
    contexts[4]['status'] = 'conflicting_native_families'
    evidence = [{'reaction_key': q['reaction_key'], 'tier': 'A' if i in (0, 3) else 'C',
                 'supported': i in (0, 3)} for i, q in enumerate(queries)]
    hits = []
    for i, match in [(0, 'exact'), (1, 'exact'), (2, 'local_edit_analogue_candidate')]:
        hits.append({'query_reaction_key': queries[i]['reaction_key'],
                     'source_id': f'source{i}', 'match_type': match,
                     'doi': '10.example/shared-dataset', 'patent_id': None,
                     'source_url': 'https://doi.org/10.example/shared-dataset',
                     'reaction_key': queries[i]['reaction_key']})
    return queries, contexts, evidence, hits


def test_audit_keeps_every_query_and_does_not_turn_no_hit_into_wrong_chemistry():
    from mechet.analysis.literature_coverage_audit import summarize
    report, rows, queue = summarize(*inputs())
    assert report['unique_reaction_queries'] == 5
    assert report['stage_counts'] == {
        'indexed_queries': 3, 'no_index_hit_queries': 2,
        'exact_candidate_queries': 2, 'analogue_candidate_queries': 1,
        'source_supported_queries': 2, 'supported_with_index_hit': 1,
        'supported_without_index_hit': 1, 'unresolved_with_candidates': 2}
    assert len(rows) == 5 and len(queue) == 2
    assert sorted(x['next_action'] for x in queue) == ['check_exact_primary_example', 'review_analogue_compatibility']
    assert report['unique_nonpatent_urls'] == 1  # Three citations are not three papers.
    assert report['population_chemical_accuracy_established'] is False
    missing = [r for r in rows if r['index_hit_count'] == 0 and r['tier'] == 'C']
    assert len(missing) == 1 and missing[0]['next_action'] == 'resolve_reaction_center_then_retrieve'
    assert missing[0]['chemical_validity'] == 'not_determined_by_retrieval'


@pytest.mark.parametrize('mutation', ['duplicate_query', 'missing_context', 'foreign_hit', 'bad_tier'])
def test_audit_refuses_filtered_or_inconsistent_populations(mutation):
    from mechet.analysis.literature_coverage_audit import summarize
    q, c, e, h = copy.deepcopy(inputs())
    if mutation == 'duplicate_query': q.append(q[0])
    if mutation == 'missing_context': c.pop()
    if mutation == 'foreign_hit': h[0]['query_reaction_key'] = 'unknown'
    if mutation == 'bad_tier': e[0]['supported'] = False
    with pytest.raises(ValueError): summarize(q, c, e, h)


def test_primary_traceback_uses_candidate_not_query_identity_and_preserves_analogue_status(tmp_path):
    from mechet.analysis.literature_coverage_audit import recover_patent_references
    from mechet.analysis.literature_index import ReactionIndex
    from mechet.analysis.literature_reaction_keys import make_reaction
    index = ReactionIndex(tmp_path/'sources.sqlite')
    records = [dict(source_id='patent1', license='public', patent_id='US01234',
                    source_url='https://patents.google.com/patent/US01234/en',
                    product='CCO', precursors=['CC', 'O']),
               dict(source_id='wrong_substrate', license='public', patent_id='US05678',
                    source_url='https://patents.google.com/patent/US05678/en',
                    product='CO', precursors=['C', 'O'])]
    for r in records: index.add(r)
    index.commit()
    source = make_reaction('CCO', ['CC', 'O'])
    hit = dict(source_id='dataset_copy', query_reaction_key='query-that-differs',
               reaction_key=source['reaction_key'], product='CCO', precursors=['CC', 'O'],
               match_type='local_edit_analogue_candidate', patent_id=None,
               source_url='https://doi.org/10.example/dataset')
    rows = recover_patent_references(index, [hit])
    assert len(rows) == 1
    row = rows[0]
    assert row['query_reaction_key'] == 'query-that-differs'
    assert row['match_type'] == 'local_edit_analogue_candidate'
    assert row['candidate_reaction_key'] == source['reaction_key']
    assert row['recovered_patents'][0]['lookup_patent_id'] == 'US1234'
    assert len(row['recovered_patents']) == 1
    assert row['performed_example_checked'] is False and row['new_evidence_grade'] is False
    bad = {**hit, 'product': 'CO'}
    with pytest.raises(ValueError, match='candidate structure'):
        recover_patent_references(index, [bad])
    index.close()


def test_audit_export_binds_all_source_rows_and_refuses_changed_payload(tmp_path):
    import gzip
    import hashlib
    import json
    from mechet.analysis.literature_coverage_audit import export_audit
    root = tmp_path/'analysis'; root.mkdir()
    names = ['unique_reaction_queries_enriched.jsonl.gz', 'retrieval_query_contexts.jsonl.gz',
             'reaction_evidence.jsonl.gz', 'literature_hits.jsonl.gz']
    outputs = {}
    for name, rows in zip(names, inputs()):
        with gzip.open(root/name, 'wt') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
        outputs[name] = {'sha256': hashlib.sha256((root/name).read_bytes()).hexdigest()}
    (root/'analysis_manifest.json').write_text(json.dumps({'complete': True, 'output_files': outputs}))
    result = export_audit(root, tmp_path/'out')
    assert result['complete'] is True and result['unique_reaction_queries'] == 5
    with gzip.open(tmp_path/'out/query_coverage_audit.jsonl.gz', 'rt') as f:
        assert len(list(f)) == 5
    with gzip.open(root/names[0], 'at') as f: f.write('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        export_audit(root, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
