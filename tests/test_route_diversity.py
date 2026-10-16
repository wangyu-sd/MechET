"""Finite AND/OR graph combinatorics, not sampled chemical/model evidence."""


def edge(name,product,children):
    return {'id':name,'product':product,'precursors':children,'cost':1,'tokens':1}


def test_complete_routes_include_inner_alternatives_not_only_root_alternatives():
    from mechet.route_diversity import iter_complete_routes
    edges=[edge('root','target',['shared']),edge('a','shared',['leaf1']),edge('b','shared',['leaf2'])]
    routes=list(iter_complete_routes('target',{'leaf1','leaf2'},edges))
    assert {tuple(sorted(r['reaction_edge_ids'])) for r in routes}=={('a','root'),('b','root')}


def test_shared_intermediate_has_one_production_choice_and_multiplicity_is_preserved():
    from mechet.route_diversity import iter_complete_routes
    edges=[edge('root','target',['shared','branch','shared']),edge('branch','branch',['shared']),
           edge('a','shared',['leaf1']),edge('b','shared',['leaf2'])]
    routes=list(iter_complete_routes('target',{'leaf1','leaf2'},edges))
    assert len(routes)==2  # NOT independently 2^3 choices for the shared molecule
    assert all(len(r['reaction_edge_ids'])==3 for r in routes)
    assert all(r['complete_reaction_edges'][0]['precursors'].count('shared')==2 for r in routes)


def test_cycles_and_unclosed_and_branches_do_not_count_as_complete_routes():
    from mechet.route_diversity import iter_complete_routes
    edges=[edge('root','target',['shared']),edge('cycle','shared',['target']),
           edge('valid','shared',['leaf']),edge('incomplete','shared',['leaf','missing'])]
    routes=list(iter_complete_routes('target',{'leaf'},edges))
    assert [set(r['reaction_edge_ids']) for r in routes]==[{'root','valid'}]
    assert list(iter_complete_routes('leaf',{'leaf'},edges))==[]  # zero-step separately


def test_count_has_no_python_recursion_or_chemical_depth_cutoff():
    from mechet.route_diversity import iter_complete_routes
    edges=[edge(str(i),str(i),[str(i+1)]) for i in range(1200)]
    routes=list(iter_complete_routes('0',{'1200'},edges))
    assert len(routes)==1
    assert len(routes[0]['reaction_edge_ids'])==1200


def test_distinct_root_and_inner_combinations_have_distinct_stable_route_keys():
    from mechet.route_diversity import iter_complete_routes
    edges=[edge('r1','target',['a']),edge('r2','target',['b']),
           edge('a1','a',['leaf1']),edge('a2','a',['leaf2']),edge('b','b',['leaf1'])]
    first=list(iter_complete_routes('target',{'leaf1','leaf2'},edges))
    second=list(iter_complete_routes('target',{'leaf1','leaf2'},list(reversed(edges))))
    assert len(first)==3
    assert {r['route_key'] for r in first}=={r['route_key'] for r in second}


def test_partially_consumed_iterator_does_not_claim_exhaustive_count():
    from mechet.route_diversity import iter_complete_routes
    stream=iter_complete_routes('target',{'leaf'},[edge('r','target',['leaf'])])
    first=next(stream)
    assert first.get('route_enumeration_exhaustive') is not True


def test_incomplete_full_run_cannot_publish_an_exhaustive_diversity_manifest(tmp_path,monkeypatch):
    import hashlib,json,sys
    import pytest
    from scripts.enumerate_frozen_route_diversity import main
    stock=tmp_path/'stock.txt';stock.write_text('leaf\n')
    assets={'stock_preparation_complete':True,'full_targets_sha256':'unit-source',
        'full_target_rows':460,'preparation_code_sha':'unit-code','normalization':'unit',
        'stocks':{'s':{'canonical_path':str(stock),'canonical_sha256':hashlib.sha256(stock.read_bytes()).hexdigest(),
                       'normalized_rows':1,'sha256':'unit-original'}},
        'cohorts':{'c':{'stock':'s','target_ids':[str(i) for i in range(460)],'rows':460}}}
    path=tmp_path/'assets.json';path.write_text(json.dumps(assets))
    output=tmp_path/'counts'
    monkeypatch.setattr(sys,'argv',['enumerate','--assets',str(path),'--run',str(tmp_path/'missing-run'),
                                  '--mode','tree','--output',str(output)])
    with pytest.raises(ValueError,match='incomplete'):
        main()
    assert not (output/'manifest.json').exists()
