"""Lossless source-row adapter for explicitly pinned Standard route datasets.

This is separate from the historical project curator: published source splits,
single-step fragments, repeated targets and unavailable stock leaves all survive.
References and complete source provenance are private reward artifacts only.
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import sqlite3

from .external_route_data import chemical_identity, edge_signature, file_sha, iter_records, parse_route, signature
from .route_training_contract import atomic_json


def _read_stock(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as reader:
        result = {chemical_identity(line.strip()) for line in reader if line.strip()}
    if not result:
        raise ValueError('EMPTY_STOCK')
    return result


def _benchmark_targets(paths):
    targets, sources = set(), {}
    for value in paths:
        path = Path(value)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as reader:
            text = reader.read()
        try:
            payload = json.loads(text)
            rows = payload.get('targets', {}).values() if isinstance(payload, dict) and 'targets' in payload else payload if isinstance(payload, list) else [payload]
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        local = set()
        for row in rows:
            molecule = next((row[k] for k in ('product','product_unmapped','target','target_smiles','smiles') if isinstance(row.get(k), str)), None)
            if molecule is None:
                raise ValueError('BENCHMARK_TARGET_SCHEMA:'+str(path))
            local.add(chemical_identity(molecule))
        targets.update(local)
        sources[str(path.resolve())] = dict(sha256=file_sha(path), unique_targets=len(local))
    return targets, sources


def _occurrences(record):
    root = record['route']['target']
    def walk(node, occurrence, parent):
        reaction = node.get('product_of')
        if reaction is None:
            return
        product = chemical_identity(node['smiles'])
        precursors = [chemical_identity(child['smiles']) for child in reaction['reactants']]
        yield occurrence, parent, product, precursors, edge_signature(product, precursors)
        for index, child in enumerate(reaction['reactants']):
            yield from walk(child, occurrence+'.'+str(index), occurrence)
    yield from walk(root, '0', None)


def prepare_standard_route_data(training_source, validation_source, output, stock_file, *,
        source_manifest, release_checksums, release, stock_condition, expected_stock_sha256,
        benchmark_targets=(), coverage_stocks=None, source_audit=None, progress=None):
    """Validate pins and translate every source row, or fail the entire build.

    Task IDs bind source bytes, split and row ordinal; equal targets remain
    distinct tasks. ``edges`` retains the existing reward lookup interface;
    ``route_occurrences`` preserves repeated reactions and precursor multisets.
    A named stock condition is an experiment choice, never imputed to upstream.
    """
    output, stock_file = Path(output), Path(stock_file)
    if output.exists():
        raise FileExistsError(output)
    if not release or release == 'latest' or not stock_condition:
        raise ValueError('EXPLICIT_RELEASE_AND_STOCK_CONDITION_REQUIRED')
    paths = dict(training=Path(training_source), validation=Path(validation_source))
    source_manifest, release_checksums = Path(source_manifest), Path(release_checksums)
    upstream = json.loads(source_manifest.read_text())
    artifact = upstream['release_name']
    checksums = {line.split(None,1)[1].strip():line.split(None,1)[0]
                 for line in release_checksums.read_text().splitlines() if line.strip()}
    pins = {}
    for split, path in paths.items():
        actual = file_sha(path)
        if actual != upstream['output_files'][split]['file_hash'] or actual != checksums.get(artifact+'/'+path.name):
            raise ValueError('SOURCE_CHECKSUM_MISMATCH:'+split)
        pins[split] = dict(path=str(path.resolve()), sha256=actual,
                          expected_records=upstream['statistics']['all_records'][split])
    if file_sha(source_manifest) != checksums.get(artifact+'/'+source_manifest.name):
        raise ValueError('SOURCE_MANIFEST_CHECKSUM_MISMATCH')
    if file_sha(stock_file) != expected_stock_sha256:
        raise ValueError('STOCK_CHECKSUM_MISMATCH')
    stock = _read_stock(stock_file)
    candidates = {name:_read_stock(path) for name,path in (coverage_stocks or {}).items()}
    benchmarks, benchmark_sources = _benchmark_targets(benchmark_targets)
    identities = {s:{name:set() for name in ('targets','molecules','reactions','routes','patents','raw_route_hashes')} for s in paths}
    distributions = {s:{name:Counter() for name in ('depth','reaction_occurrences')} for s in paths}
    coverage = {s:Counter() for s in paths}
    candidate_coverage = {name:{s:Counter() for s in paths} for name in candidates}
    benchmark_hits = {s:Counter() for s in paths}
    counts, frequencies = Counter(), Counter()
    report = dict(schema='external_target_only_route_standard_v1', release=release,
        artifact=artifact, source_files=pins, source_manifest=str(source_manifest.resolve()),
        source_manifest_sha256=file_sha(source_manifest), release_checksums=str(release_checksums.resolve()),
        release_checksums_sha256=file_sha(release_checksums), source_records=0, accepted_routes=0,
        excluded={}, duplicate_routes=0, source_rows_filtered=0,
        source_split_policy='preserve_source_training_validation_without_reassignment',
        task_identity='one_source_row_per_task_bound_to_source_sha_split_and_ordinal_v1',
        reference_action_labels=False, policy_input='target_main_product_only_v1',
        stock_source=str(stock_file.resolve()), stock_source_sha256=expected_stock_sha256,
        stock_protocol=dict(condition=stock_condition,is_source_declared_training_stock=False,
            inventory_rule='explicit_project_experiment_inventory_no_reference_leaf_union',
            source_release_training_stock='not_declared',stock_pruning=False,stock_closure_filter=False),
        protected_identity_policy='empty_no_censoring_preserve_all_known_exposure',
        reaction_frequency_scope='source_training_only_excludes_validation_statistics',
        benchmark_target_sources=benchmark_sources,
        identity_contract='existing charge/isotope/stereo-preserving canonical SMILES; differs from source InChIKey tautomer identity',
        caveats=['Source TRAIN/VALID may share targets, reactions, routes and patents.',
                 'Route-RL reaction holdout does not undo frozen FlowER SFT pretraining exposure.',
                 'Known test target exposure is retained and audited; this is not target-clean training.',
                 'Non-stock-closed source references remain in the complete denominator.',
                 'Source route references provide no native electron-arrow action labels.'])
    if source_audit is not None:
        p = Path(source_audit)
        report['upstream_source_audit'] = dict(path=str(p.resolve()),sha256=file_sha(p))
    report['candidate_stock_sources'] = {name:dict(path=str(Path(path).resolve()),sha256=file_sha(path)) for name,path in (coverage_stocks or {}).items()}
    output.mkdir(parents=True, exist_ok=False)
    temporary = output/'references.sqlite.building'
    try:
        with sqlite3.connect(temporary) as db:
            db.executescript('''
                PRAGMA journal_mode=MEMORY; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY;
                CREATE TABLE tasks (id TEXT PRIMARY KEY,target TEXT,public_target TEXT,split TEXT,reference_routes INTEGER,reference_steps INTEGER);
                CREATE TABLE edges (task_id TEXT,signature TEXT,product TEXT,precursors TEXT,PRIMARY KEY(task_id,signature));
                CREATE TABLE routes (signature TEXT,task_id TEXT PRIMARY KEY,tree TEXT,original_tree TEXT,reaction_count INTEGER,depth INTEGER,source_record INTEGER);
                CREATE TABLE route_sources (task_id TEXT PRIMARY KEY,route_signature TEXT,source_record INTEGER,split TEXT,provenance TEXT);
                CREATE TABLE route_occurrences (task_id TEXT,occurrence TEXT,parent_occurrence TEXT,product TEXT,precursors TEXT,signature TEXT,PRIMARY KEY(task_id,occurrence));
            ''')
            with (output/'reference_witnesses.jsonl').open('w') as witnesses:
                for split,path in paths.items():
                    records = 0
                    with (output/(split+'.jsonl')).open('w') as public:
                        for ordinal,row in enumerate(iter_records(path),1):
                            records += 1
                            if row.get('split') != split:
                                raise ValueError('SOURCE_SPLIT_MISMATCH:'+split+':'+str(ordinal))
                            try:
                                route = parse_route(row)
                                occurrences = list(_occurrences(row))
                            except (ValueError,KeyError,TypeError,RecursionError) as exc:
                                atomic_json(output/'translation_failures.json',dict(split=split,source=str(path.resolve()),source_record=ordinal,source_id=row.get('id'),error=str(exc),source_rows_dropped=0,build_complete=False))
                                raise ValueError('SOURCE_ROUTE_INVALID:'+split+':'+str(ordinal)+':'+str(exc)) from exc
                            if len(occurrences) != route['reaction_count']:
                                raise ValueError('SOURCE_OCCURRENCE_ACCOUNTING_MISMATCH')
                            task = signature([release,artifact,pins[split]['sha256'],split,ordinal,row.get('id')])
                            tree = json.dumps(route['tree'],separators=(',',':'))
                            db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?)',(task,route['target'],route['public_target'],split,1,route['reaction_count']))
                            db.execute('INSERT INTO routes VALUES (?,?,?,?,?,?,?)',(route['signature'],task,tree,tree,route['reaction_count'],route['depth'],ordinal))
                            db.execute('INSERT INTO route_sources VALUES (?,?,?,?,?)',(task,route['signature'],ordinal,split,json.dumps(row,separators=(',',':'),ensure_ascii=False)))
                            db.executemany('INSERT OR IGNORE INTO edges VALUES (?,?,?,?)',[(task,e['signature'],e['product'],json.dumps(e['precursors'])) for e in route['edges']])
                            db.executemany('INSERT INTO route_occurrences VALUES (?,?,?,?,?,?)',[(task,o,parent,p,json.dumps(cs),key) for o,parent,p,cs,key in occurrences])
                            public.write(json.dumps(dict(task_id=task,target=route['public_target']))+'\n')
                            witnesses.write(json.dumps(dict(task_id=task,route_signature=route['signature'],reaction_occurrences=route['reaction_count']))+'\n')
                            counts['source_records']+=1;counts[split+'_targets']+=1;counts[split+'_reaction_occurrences']+=route['reaction_count']
                            distributions[split]['depth'][route['depth']]+=1
                            distributions[split]['reaction_occurrences'][route['reaction_count']]+=1
                            values=identities[split]
                            values['targets'].add(route['target']);values['molecules'].update(route['molecules'])
                            values['routes'].add(route['signature']);values['reactions'].update(e['signature'] for e in route['edges'])
                            values['patents'].update(row['route'].get('annotations',{}).get('source_patent_ids',[]))
                            values['patents'].update(s['patent_id'] for s in row.get('sources',[]) if s.get('patent_id'))
                            values['raw_route_hashes'].update(s['raw_route_hash'] for s in row.get('sources',[]) if s.get('raw_route_hash'))
                            if split == 'training':
                                frequencies.update(e['signature'] for e in route['edges'])
                            coverage[split]['target_in_stock']+=route['target'] in stock
                            coverage[split]['raw_reference_stock_closed']+=all(v in stock for v in route['leaves'])
                            coverage[split]['raw_leaf_occurrences']+=len(route['leaves'])
                            coverage[split]['raw_leaf_missing_occurrences']+=sum(v not in stock for v in route['leaves'])
                            for name,inventory in candidates.items():
                                candidate_coverage[name][split]['target_in_stock']+=route['target'] in inventory
                                candidate_coverage[name][split]['raw_reference_stock_closed']+=all(v in inventory for v in route['leaves'])
                            benchmark_hits[split]['source_rows_with_root_match']+=route['target'] in benchmarks
                            benchmark_hits[split]['source_rows_with_any_molecule_match']+=bool(set(route['molecules'])&benchmarks)
                            if ordinal%2048==0:
                                db.commit()
                                if progress:progress(dict(event='standard-route-translation',split=split,source_records=ordinal))
                    if records != pins[split]['expected_records']:
                        raise ValueError('SOURCE_RECORD_COUNT_MISMATCH:'+split)
            db.execute('CREATE INDEX routes_signature ON routes(signature)')
            db.execute('CREATE INDEX tasks_split ON tasks(split)')
            db.execute('CREATE INDEX occurrences_signature ON route_occurrences(signature)')
            report['private_reference_edges']=db.execute('SELECT count(*) FROM edges').fetchone()[0]
            report['private_source_witnesses']=db.execute('SELECT count(*) FROM route_sources').fetchone()[0]
            report['private_reaction_occurrences']=db.execute('SELECT count(*) FROM route_occurrences').fetchone()[0]
        temporary.rename(output/'references.sqlite')
        report.update(counts);report['accepted_routes']=counts['source_records']
        report['stock_coverage']=coverage;report['candidate_stock_coverage']=candidate_coverage
        report['unique_identities']={s:{k:len(v) for k,v in group.items()} for s,group in identities.items()}
        report['route_distributions']=distributions
        report['cross_split_audit']={'shared_'+k:len(identities['training'][k]&identities['validation'][k]) for k in identities['training']}
        report['benchmark_overlap']={s:dict(benchmark_hits[s],root_identities=len(group['targets']&benchmarks),any_route_molecule_identities=len(group['molecules']&benchmarks)) for s,group in identities.items()}
        (output/'stock.txt').write_text('\n'.join(sorted(stock))+'\n')
        for name in ('protected_identities.json','training_protected_identities.json','validation_protected_identities.json'):
            atomic_json(output/name,[])
        with (output/'reaction_frequencies.jsonl').open('w') as sink:
            for key,count in sorted(frequencies.items()):sink.write(json.dumps(dict(signature=key,reference_occurrences=count))+'\n')
        if report['private_reaction_occurrences'] != sum(counts[s+'_reaction_occurrences'] for s in paths):
            raise ValueError('COMPLETE_OCCURRENCE_ACCOUNTING_MISMATCH')
        report['artifacts']={p.name:file_sha(p) for p in output.iterdir() if p.is_file()}
        atomic_json(output/'manifest.json',report)
        return report
    except Exception:
        # Keep owned partial artifacts for diagnosis; a missing manifest means
        # this directory is not a published complete dataset and cannot resume.
        raise
