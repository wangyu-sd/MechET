"""Software fixtures verify full-source preservation, never benchmark success."""
import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


def molecule(smiles, children=None):
    value = dict(smiles=smiles, annotations={})
    if children is not None:
        value['product_of'] = dict(reactants=children, annotations={'source_id': 'patent;step'})
    return value


def fixtures(tmp_path, training=None, validation=None):
    tree = molecule('CO', [molecule('C'), molecule('O')])
    training = training if training is not None else [tree, tree]
    validation = validation if validation is not None else [tree]
    files = {}
    for split, trees in [('training', training), ('validation', validation)]:
        path = tmp_path/(split+'.jsonl.gz')
        with gzip.open(path, 'wt') as sink:
            for index, route in enumerate(trees):
                sink.write(json.dumps(dict(id=f'{split}-{index}', split=split,
                    route=dict(target=route, schema_version='2', annotations={}),
                    sources=[dict(dataset='all', patent_id='patent', raw_index=index,
                                  raw_route_hash=f'{split}-raw-{index}')]))+'\n')
        files[split] = path
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path/'source_manifest.json'
    manifest.write_text(json.dumps(dict(schema_version='2', release_name='reaction-holdout-n1-n5',
        output_files={s:dict(file_hash=digest(p)) for s,p in files.items()},
        statistics=dict(all_records=dict(training=len(training), validation=len(validation), total=len(training)+len(validation))))))
    checksums = tmp_path/'SHA256SUMS'
    checksums.write_text(''.join(digest(p)+'  reaction-holdout-n1-n5/'+p.name+'\n' for p in [*files.values(), manifest]))
    stock = tmp_path/'stock.txt'; stock.write_text('CO\nN\n')
    benchmark = tmp_path/'benchmark.jsonl'; benchmark.write_text('{"id":"test-CO","product":"CO"}\n')
    return dict(training_source=files['training'], validation_source=files['validation'],
                source_manifest=manifest, release_checksums=checksums,
                stock_file=stock, expected_stock_sha256=digest(stock),
                release='v2026-06-05', stock_condition='project_d1_ursa_guarded_training_stock_v1',
                benchmark_targets=[benchmark], coverage_stocks={'n1-n5-stock': stock})


def builder():
    assert importlib.util.find_spec('mechet.route_standard_data') is not None, 'full-source translator is missing'
    return importlib.import_module('mechet.route_standard_data').prepare_standard_route_data


def test_full_source_keeps_split_rows_depth_one_stock_roots_and_exposure(tmp_path):
    args = fixtures(tmp_path)
    out = tmp_path/'converted'
    report = builder()(**args, output=out)
    train = [json.loads(s) for s in (out/'training.jsonl').read_text().splitlines()]
    valid = [json.loads(s) for s in (out/'validation.jsonl').read_text().splitlines()]
    assert len(train) == 2 and len(valid) == 1
    assert len({r['task_id'] for r in train+valid}) == 3
    assert all(set(r) == {'task_id', 'target'} for r in train+valid)
    assert all(r['target'] == 'CO' for r in train+valid)
    assert report['source_records'] == 3 and report['excluded'] == {}
    assert report['cross_split_audit']['shared_targets'] == 1
    assert report['benchmark_overlap']['training']['root_identities'] == 1
    assert report['stock_protocol']['is_source_declared_training_stock'] is False
    assert report['stock_coverage']['training']['target_in_stock'] == 2
    assert report['stock_coverage']['training']['raw_reference_stock_closed'] == 0
    frequencies = [json.loads(line) for line in (out/'reaction_frequencies.jsonl').read_text().splitlines()]
    assert len(frequencies) == 1 and frequencies[0]['reference_occurrences'] == 2
    for name in ('protected_identities.json','training_protected_identities.json','validation_protected_identities.json'):
        assert json.loads((out/name).read_text()) == []
    with sqlite3.connect(out/'references.sqlite') as db:
        assert db.execute('SELECT count(*) FROM tasks').fetchone()[0] == 3
        assert db.execute('SELECT count(*) FROM route_sources').fetchone()[0] == 3
        assert db.execute('SELECT count(*) FROM route_occurrences').fetchone()[0] == 3
        assert db.execute('SELECT DISTINCT reference_steps FROM tasks').fetchall() == [(1,)]


def test_occurrences_and_duplicate_precursor_obligations_survive_edge_deduplication(tmp_path):
    branch = molecule('CO', [molecule('C'), molecule('O')])
    tree = molecule('COCOC', [branch, branch])
    args = fixtures(tmp_path, training=[tree], validation=[])
    out = tmp_path/'converted'; builder()(**args, output=out)
    with sqlite3.connect(out/'references.sqlite') as db:
        assert db.execute('SELECT reference_steps FROM tasks').fetchone() == (3,)
        assert db.execute('SELECT count(*) FROM edges').fetchone()[0] == 2
        rows = db.execute('SELECT occurrence,product,precursors FROM route_occurrences ORDER BY occurrence').fetchall()
        assert [(r[0],r[1],json.loads(r[2])) for r in rows] == [
            ('0','COCOC',['CO','CO']),('0.0','CO',['C','O']),('0.1','CO',['C','O'])]
        assert json.loads(db.execute('SELECT provenance FROM route_sources').fetchone()[0])['sources'][0]['patent_id'] == 'patent'


def test_corrupt_source_or_stock_pin_fails_before_publishing(tmp_path):
    args = fixtures(tmp_path)
    args['training_source'].write_bytes(args['training_source'].read_bytes()+b'corrupt')
    out = tmp_path/'converted'
    with pytest.raises(ValueError, match='CHECKSUM'):
        builder()(**args, output=out)
    assert not (out/'manifest.json').exists()


def test_invalid_reference_fails_whole_build_without_dropping_row(tmp_path):
    args = fixtures(tmp_path, training=[molecule('CO', [molecule('CO')])])
    out = tmp_path/'converted'
    with pytest.raises(ValueError, match='SOURCE_ROUTE_INVALID'):
        builder()(**args, output=out)
    assert not (out/'manifest.json').exists()
    errors = json.loads((out/'translation_failures.json').read_text())
    assert errors['split'] == 'training' and errors['source_record'] == 1


def test_hidden_reward_consumes_row_tasks_without_hidden_data_in_public_jsonl(tmp_path):
    from mechet.external_route_rl import HiddenRouteReward
    args = fixtures(tmp_path)
    args['stock_file'].write_text('C\nO\n')
    args['expected_stock_sha256'] = hashlib.sha256(args['stock_file'].read_bytes()).hexdigest()
    out = tmp_path/'converted'; builder()(**args, output=out)
    task = json.loads((out/'validation.jsonl').read_text().strip())
    report = dict(root='CO',status='stock_solved',reactions=[dict(occurrence='0',product='CO',precursors=['C','O'])],stock_leaves=['C','O'])
    with HiddenRouteReward(out/'references.sqlite', {'C','O'}, split='validation') as reward:
        assert reward.score(task['task_id'],report)['verified_stock_closed'] is True
