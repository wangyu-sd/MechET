"""External route trees -> public targets and a separate hidden reward database."""
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

from .route_training_contract import chemical_identity as _chemical_identity
from .route_training_contract import public_molecule, atomic_json

chemical_identity=lru_cache(maxsize=262144)(_chemical_identity)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda: reader.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def edge_signature(product, precursors):
    return signature([chemical_identity(product), sorted(chemical_identity(v) for v in precursors)])


def iter_records(path):
    """Stream JSONL or a top-level JSON array, optionally gzip-compressed."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as reader:
        first = reader.read(1)
        while first and first.isspace():
            first = reader.read(1)
        if first != '[':
            if not first:
                return
            line = first+reader.readline()
            if line.strip():
                yield json.loads(line)
            for line in reader:
                if line.strip():
                    yield json.loads(line)
            return
        decoder = json.JSONDecoder()
        buffer, eof, expect_value, after_comma = '', False, True, False
        while True:
            buffer = buffer.lstrip()
            if not buffer and not eof:
                chunk = reader.read(1024*1024)
                buffer += chunk; eof = not chunk
                continue
            if buffer.startswith(']'):
                if after_comma:
                    raise ValueError('JSON_ARRAY_TRAILING_COMMA')
                if not expect_value or not buffer[1:].strip():
                    if buffer[1:].strip() or reader.read().strip():
                        raise ValueError('JSON_ARRAY_TRAILING_DATA')
                    return
                raise ValueError('INVALID_JSON_ARRAY')
            if not expect_value:
                if not buffer.startswith(','):
                    raise ValueError('JSON_ARRAY_EXPECTED_COMMA')
                buffer = buffer[1:].lstrip(); expect_value = True; after_comma = True
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError('INCOMPLETE_JSON_ARRAY')
                chunk = reader.read(1024*1024)
                buffer += chunk; eof = not chunk
                continue
            yield value
            buffer, expect_value, after_comma = buffer[end:], False, False


def parse_route(record):
    """PaRoutes mol/reaction, DMS mol/children, or RetroCast schema-2 trees.

    There is no mechanism compilation or teacher-action reconstruction here.
    A reference is one AND tree; ambiguous OR alternatives must be separate rows.
    """
    value = record.get('route', record)
    root = value.get('target', value)
    if isinstance(root, str):
        raise ValueError('EXPECTED_REFERENCE_TREE')
    edges, leaves, molecules = [], [], set()

    def visit(node, ancestors):
        product = chemical_identity(node['smiles'])
        if product in ancestors:
            raise ValueError('REFERENCE_CYCLE')
        molecules.add(product)
        reaction = node.get('product_of')
        if reaction is not None:
            if not isinstance(reaction, dict):
                raise ValueError('AMBIGUOUS_REFERENCE_OR_NODE')
            children = reaction['reactants']
        else:
            children = node.get('children', [])
            reactions = [v for v in children if v.get('type') == 'reaction']
            if reactions:
                if len(reactions) != 1 or len(children) != 1:
                    raise ValueError('AMBIGUOUS_REFERENCE_OR_NODE')
                children = reactions[0]['children']
        if not children:
            leaves.append(product)
            return [product, []], 0
        if '.' in product:
            raise ValueError('NON_MAIN_ONLY_REFERENCE_PRODUCT')
        descendants = [visit(c, ancestors | {product}) for c in children]
        precursors = sorted(chemical_identity(c['smiles']) for c in children)
        if precursors == [product]:
            raise ValueError('NONPRODUCTIVE_REFERENCE')
        edges.append(dict(product=product, precursors=precursors,
                          signature=edge_signature(product, precursors)))
        branches = sorted((v[0] for v in descendants), key=lambda v: json.dumps(v, sort_keys=True))
        return [product, branches], 1+max(d for _, d in descendants)

    tree, depth = visit(root, set())
    if '.' in tree[0]:
        raise ValueError('NON_MAIN_ONLY_REFERENCE_ROOT')
    return dict(target=tree[0], public_target=public_molecule(root['smiles']), tree=tree,
                edges=edges, leaves=leaves, molecules=sorted(molecules), depth=depth,
                reaction_count=len(edges), signature=signature(tree))


def prune_stock(route, stock):
    """Only count reactions an executable route actually needs before stock."""
    edges,leaves=[],[]
    def visit(tree):
        product,children=tree
        if product in stock or not children:
            leaves.append(product)
            return [product,[]],0
        descendants=[visit(child) for child in children]
        precursors=sorted(child[0] for child in children)
        edges.append(dict(product=product,precursors=precursors,
                          signature=edge_signature(product,precursors)))
        return [product,[v for v,_ in descendants]],1+max(d for _,d in descendants)
    tree,depth=visit(route['tree'])
    return dict(route,tree=tree,depth=depth,edges=edges,leaves=leaves,
                reaction_count=len(edges),signature=signature(tree))


def protected_identities(paths):
    result, counts = set(), {}
    for path in paths:
        count = 0
        for row in iter_records(path):
            value = next((row[k] for k in ('product_unmapped', 'product', 'target', 'target_smiles', 'smiles')
                          if isinstance(row.get(k), str)), None)
            if value is None:
                raise ValueError('PROTECTED_TARGET_SCHEMA:'+str(path))
            result.add(chemical_identity(value)); count += 1
        if not count:
            raise ValueError('EMPTY_PROTECTED_TARGET_FILE:'+str(path))
        counts[str(Path(path).resolve())] = count
    if not counts:
        raise ValueError('PROTECTED_EVALUATION_TARGETS_REQUIRED')
    return result, counts


def build_dataset(source, output, stock_file, protected_files, *, validation_fraction=.1,
                  seed=17, protected_reactions=None, progress=None, validation_dataset=None):
    if validation_dataset is not None:
        return build_dataset_v2(source,output,stock_file,protected_files,
                                validation_dataset=validation_dataset,
                                validation_fraction=validation_fraction,seed=seed,
                                protected_reactions=protected_reactions,progress=progress)
    if not 0 <= validation_fraction < 1:
        raise ValueError('INVALID_VALIDATION_FRACTION')
    source, output, stock_file = Path(source), Path(output), Path(stock_file)
    output.mkdir(parents=True, exist_ok=False)
    protected, protected_counts = protected_identities(protected_files)
    stock = {chemical_identity(v.strip()) for v in stock_file.read_text().splitlines() if v.strip()}
    if not stock:
        raise ValueError('EMPTY_STOCK')
    forbidden = set()
    if protected_reactions:
        forbidden = {edge_signature(r['product'], r['precursors']) for r in iter_records(protected_reactions)}
    report = dict(schema='external_target_only_route_rl_v1', source_records=0,
                  duplicate_routes=0, accepted_routes=0, excluded=Counter(),
                  source=str(source.resolve()), source_sha256=file_sha(source),
                  stock_source=str(stock_file.resolve()), stock_source_sha256=file_sha(stock_file),
                  protected_sources={str(Path(p).resolve()): file_sha(p) for p in protected_files},
                  protected_source_records=protected_counts, seed=seed,
                  validation_fraction=validation_fraction,
                  policy_input='target_main_product_only_v1', reference_action_labels=False,
                  holdout_scope='all route molecules against protected targets; optional reaction identities')
    if protected_reactions:
        report['protected_reactions_source'] = str(Path(protected_reactions).resolve())
        report['protected_reactions_sha256'] = file_sha(protected_reactions)
    temporary = output/'references.sqlite.building'
    with sqlite3.connect(temporary) as db:
        db.executescript('''
            CREATE TABLE tasks (id TEXT PRIMARY KEY, target TEXT UNIQUE, public_target TEXT,
                                split TEXT, reference_routes INTEGER, reference_steps INTEGER);
            CREATE TABLE edges (task_id TEXT, signature TEXT, product TEXT, precursors TEXT,
                                PRIMARY KEY(task_id, signature));
            CREATE TABLE routes (signature TEXT PRIMARY KEY, task_id TEXT);
        ''')
        for row in iter_records(source):
            report['source_records'] += 1
            if progress and report['source_records'] % 2048 == 0:
                progress(dict(event='external-route-curation',source_records=report['source_records'],
                              accepted_routes=report['accepted_routes'],excluded=dict(report['excluded'])))
            if row.get('split') in ('test','validation','valid','evaluation','eval'):
                report['excluded']['upstream_holdout'] += 1
                continue
            try:
                route = parse_route(row)
            except (ValueError, KeyError, TypeError, RecursionError) as exc:
                report['excluded']['invalid_route:'+type(exc).__name__] += 1
                continue
            if route['depth'] < 2:
                reason = 'not_multistep'
            elif protected.intersection(route['molecules']):
                reason = 'protected_molecule'
            elif any(e['signature'] in forbidden for e in route['edges']):
                reason = 'protected_reaction'
            elif route['target'] in stock:
                reason = 'already_in_stock'
            else:
                reason = ''
            if reason:
                report['excluded'][reason] += 1
                continue
            route=prune_stock(route,stock)
            if route['depth'] < 2:
                report['excluded']['stock_pruned_not_multistep'] += 1
                continue
            if not all(v in stock for v in route['leaves']):
                report['excluded']['reference_not_stock_closed'] += 1
                continue
            task_id = signature(route['target'])
            exists = db.execute('SELECT 1 FROM routes WHERE signature=?', (route['signature'],)).fetchone()
            if exists:
                report['duplicate_routes'] += 1
                continue
            split = 'validation' if int(signature([seed, route['target']]),16)/2**256 < validation_fraction else 'training'
            db.execute('INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,0,?)',
                       (task_id, route['target'], route['public_target'], split, route['reaction_count']))
            db.execute('UPDATE tasks SET reference_routes=reference_routes+1, reference_steps=min(reference_steps,?) WHERE id=?',
                       (route['reaction_count'], task_id))
            db.execute('INSERT INTO routes VALUES (?,?)', (route['signature'], task_id))
            db.executemany('INSERT OR IGNORE INTO edges VALUES (?,?,?,?)',
                           [(task_id,e['signature'],e['product'],json.dumps(e['precursors'])) for e in route['edges']])
            report['accepted_routes'] += 1
            if report['source_records'] % 4096 == 0:
                db.commit()
        for split in ('training', 'validation'):
            count = 0
            with (output/f'{split}.jsonl').open('w') as sink:
                for task_id, target in db.execute('SELECT id,public_target FROM tasks WHERE split=? ORDER BY id', (split,)):
                    sink.write(json.dumps(dict(task_id=task_id,target=target),ensure_ascii=False)+'\n'); count += 1
            report[split+'_targets'] = count
        report['private_reference_edges'] = db.execute('SELECT count(*) FROM edges').fetchone()[0]
    temporary.rename(output/'references.sqlite')
    (output/'stock.txt').write_text('\n'.join(sorted(stock))+'\n')
    atomic_json(output/'protected_identities.json', sorted(protected))
    report['excluded'] = dict(report['excluded'])
    assert report['source_records'] == report['accepted_routes']+report['duplicate_routes']+sum(report['excluded'].values())
    report['artifacts'] = {p.name:file_sha(p) for p in output.iterdir() if p.is_file()}
    atomic_json(output/'manifest.json',report)
    return report


def _frozen_validation(dataset, source_sha, stock_sha, protected_sources, seed):
    """Authenticate the old snapshot before reusing its validation namespace."""
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset/'manifest.json').read_text())
    for name, expected in manifest['artifacts'].items():
        if file_sha(dataset/name) != expected:
            raise ValueError('FROZEN_VALIDATION_ARTIFACT:'+name)
    for key, actual in [('source_sha256',source_sha),('stock_source_sha256',stock_sha),
                        ('protected_sources',protected_sources),('seed',seed)]:
        if manifest[key] != actual:
            raise ValueError('FROZEN_VALIDATION_INPUT:'+key)
    with sqlite3.connect((dataset/'references.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        valid = dict(db.execute('SELECT target,id FROM tasks WHERE split="validation"'))
        training_ids = {v[0] for v in db.execute('SELECT id FROM tasks WHERE split="training"')}
        edges = {v[0] for v in db.execute('SELECT DISTINCT e.signature FROM edges e JOIN tasks t ON e.task_id=t.id WHERE t.split="validation"')}
        routes = {v[0] for v in db.execute('SELECT r.signature FROM routes r JOIN tasks t ON r.task_id=t.id WHERE t.split="validation"')}
    return dict(dataset=str(dataset), manifest_sha256=file_sha(dataset/'manifest.json'),
                canonical_stock_sha256=manifest['artifacts']['stock.txt'],
                public_sha256=file_sha(dataset/'validation.jsonl'), validation_fraction=manifest['validation_fraction'], roots=valid,
                reactions=edges, routes=routes, training_ids=training_ids)


def _tree_stats(tree):
    """Count occurrence branches, including the separate convergence criterion."""
    _, children = tree
    if not children:
        return 0,0
    counts = [_tree_stats(v) for v in children]
    return (int(len(children)>1)+sum(v[0] for v in counts),
            int(sum(bool(c[1]) for c in children)>1)+sum(v[1] for v in counts))


def _provenance_patents(record):
    patents = {v['patent_id'] for v in record.get('sources',[]) if v.get('patent_id')}
    route = record.get('route',record)
    patents.update(route.get('annotations',{}).get('source_patent_ids',[]))
    return patents


def _scaffold(target):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    molecule = Chem.MolFromSmiles(target)
    if molecule is None:
        raise ValueError('SCAFFOLD_IDENTITY_PARSE')
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule,includeChirality=True)


def build_dataset_v2(source, output, stock_file, protected_files, *, validation_dataset,
                     validation_fraction=.1, seed=17, protected_reactions=None, progress=None):
    """Remove whole original TRAIN witnesses crossing frozen VALID labels.

    No graph is cut into invented new targets. A target survives only through an
    original witness that still independently closes against the fixed stock.
    Source JSON is private and preserves patent and per-reaction provenance.
    """
    source,output,stock_file=Path(source),Path(output),Path(stock_file)
    source_sha,stock_sha=file_sha(source),file_sha(stock_file)
    protected,protected_counts=protected_identities(protected_files)
    protected_sources={str(Path(p).resolve()):file_sha(p) for p in protected_files}
    frozen=_frozen_validation(validation_dataset,source_sha,stock_sha,protected_sources,seed)
    stock_rows=stock_file.read_text().splitlines()
    # The authenticated predecessor already canonicalized its published stock.
    # Byte-identical input needs no second RDKit pass over 255k molecules.
    stock=({v for v in stock_rows if v} if stock_sha==frozen['canonical_stock_sha256']
           else {chemical_identity(v.strip()) for v in stock_rows if v.strip()})
    if not stock:
        raise ValueError('EMPTY_STOCK')
    forbidden=set()
    if protected_reactions:
        forbidden={edge_signature(r['product'],r['precursors']) for r in iter_records(protected_reactions)}
    valid_roots=set(frozen['roots']);valid_reactions=frozen['reactions']
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='external_target_only_route_rl_v2',source_records=0,accepted_routes=0,
                duplicate_routes=0,excluded=Counter(),source=str(source.resolve()),
                source_sha256=source_sha,stock_source=str(stock_file.resolve()),
                stock_source_sha256=stock_sha,protected_sources=protected_sources,
                protected_source_records=protected_counts,seed=seed,
                validation_fraction=frozen['validation_fraction'],policy_input='target_main_product_only_v1',
                reference_action_labels=False,
                holdout_scope='all original TRAIN witness molecules and reactions against frozen VALID roots/reactions; all original route molecules against benchmark targets',
                reference_witness_rule='minimum_reaction_occurrences_then_canonical_signature',
                fixed_validation=dict(dataset=frozen['dataset'],manifest_sha256=frozen['manifest_sha256'],
                                      targets=len(valid_roots),reactions=len(valid_reactions),
                                      public_sha256=frozen['public_sha256']))
    if protected_reactions:
        report.update(protected_reactions_source=str(Path(protected_reactions).resolve()),
                      protected_reactions_sha256=file_sha(protected_reactions))
    rejected_training_ids=set();raw_train_molecules=set();raw_train_reactions=set()
    patents={'training':set(),'validation':set()};scaffolds={'training':set(),'validation':set()}
    route_stats={s:dict(depth=Counter(),reaction_occurrences=Counter(),and_branch_occurrences=Counter(),
                       convergent_branch_occurrences=Counter(),shape=Counter()) for s in patents}
    reaction_frequency=Counter()
    temporary=output/'references.sqlite.building'
    with sqlite3.connect(temporary) as db:
        db.executescript('''
            CREATE TABLE tasks (id TEXT PRIMARY KEY,target TEXT UNIQUE,public_target TEXT,
                                split TEXT,reference_routes INTEGER,reference_steps INTEGER);
            CREATE TABLE edges (task_id TEXT,signature TEXT,product TEXT,precursors TEXT,
                                PRIMARY KEY(task_id,signature));
            CREATE TABLE routes (signature TEXT PRIMARY KEY,task_id TEXT,tree TEXT,
                                 original_tree TEXT,reaction_count INTEGER,depth INTEGER,
                                 source_record INTEGER,provenance TEXT);
            CREATE TABLE route_sources (route_signature TEXT,source_record INTEGER,provenance TEXT,
                                        PRIMARY KEY(route_signature,source_record));
        ''')
        for row in iter_records(source):
            report['source_records']+=1
            ordinal=report['source_records']
            if progress and ordinal%2048==0:
                progress(dict(event='external-route-v2-curation',source_records=ordinal,
                              accepted_routes=report['accepted_routes'],excluded=dict(report['excluded'])))
            if row.get('split') in ('test','validation','valid','evaluation','eval'):
                report['excluded']['upstream_holdout']+=1;continue
            try:
                original=parse_route(row)
            except (ValueError,KeyError,TypeError,RecursionError) as exc:
                report['excluded']['invalid_route:'+type(exc).__name__]+=1;continue
            if original['depth']<2:reason='not_multistep'
            elif protected.intersection(original['molecules']):reason='protected_molecule'
            elif any(e['signature'] in forbidden for e in original['edges']):reason='protected_reaction'
            elif original['target'] in stock:reason='already_in_stock'
            else:reason=''
            if reason:
                report['excluded'][reason]+=1;continue
            route=prune_stock(original,stock)
            if route['depth']<2:
                report['excluded']['stock_pruned_not_multistep']+=1;continue
            if not all(v in stock for v in route['leaves']):
                report['excluded']['reference_not_stock_closed']+=1;continue
            task_id=signature(route['target'])
            split='validation' if route['target'] in valid_roots else 'training'
            if split=='training':
                if task_id not in frozen['training_ids']:
                    raise ValueError('NEW_TARGET_OUTSIDE_FROZEN_V1_TRAIN')
                if valid_roots.intersection(original['molecules']):reason='validation_molecule'
                elif any(e['signature'] in valid_reactions for e in original['edges']):reason='validation_reaction'
                if reason:
                    report['excluded'][reason]+=1;rejected_training_ids.add(task_id);continue
                raw_train_molecules.update(original['molecules'])
                raw_train_reactions.update(e['signature'] for e in original['edges'])
            provenance=json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False)
            db.execute('INSERT INTO route_sources VALUES (?,?,?)',(route['signature'],ordinal,provenance))
            patents[split].update(_provenance_patents(row))
            if db.execute('SELECT 1 FROM routes WHERE signature=?',(route['signature'],)).fetchone():
                report['duplicate_routes']+=1;continue
            db.execute('INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,0,?)',
                       (task_id,route['target'],route['public_target'],split,route['reaction_count']))
            db.execute('UPDATE tasks SET reference_routes=reference_routes+1,reference_steps=min(reference_steps,?) WHERE id=?',
                       (route['reaction_count'],task_id))
            db.execute('INSERT INTO routes VALUES (?,?,?,?,?,?,?,?)',
                       (route['signature'],task_id,json.dumps(route['tree']),json.dumps(original['tree']),
                        route['reaction_count'],route['depth'],ordinal,provenance))
            db.executemany('INSERT OR IGNORE INTO edges VALUES (?,?,?,?)',
                           [(task_id,e['signature'],e['product'],json.dumps(e['precursors'])) for e in route['edges']])
            reaction_frequency.update(e['signature'] for e in route['edges'])
            and_count,convergence=_tree_stats(route['tree'])
            stats=route_stats[split]
            for key,value in [('depth',route['depth']),('reaction_occurrences',route['reaction_count']),
                              ('and_branch_occurrences',and_count),('convergent_branch_occurrences',convergence),
                              ('shape','convergent' if convergence else 'linear')]:
                stats[key][value]+=1
            report['accepted_routes']+=1
            if ordinal%4096==0:db.commit()
        actual_valid_routes={r[0] for r in db.execute('SELECT r.signature FROM routes r JOIN tasks t ON r.task_id=t.id WHERE t.split="validation"')}
        if actual_valid_routes!=frozen['routes']:
            raise ValueError('FROZEN_VALIDATION_WITNESSES_CHANGED')
        train_ids={r[0] for r in db.execute('SELECT id FROM tasks WHERE split="training"')}
        removed=frozen['training_ids']-train_ids
        report['validation_cleaning']=dict(training_tasks_before=len(frozen['training_ids']),
                                           training_tasks_removed=len(removed),
                                           training_tasks_retained_with_alternative=len(rejected_training_ids&train_ids),
                                           training_tasks_with_rejected_witness=len(rejected_training_ids),
                                           removed_task_ids=sorted(removed))
        report['cross_split_audit']=dict(train_original_molecules_intersect_valid_roots=len(raw_train_molecules&valid_roots),
                                        train_original_reactions_intersect_valid_reactions=len(raw_train_reactions&valid_reactions),
                                        train_original_molecules_intersect_benchmark_targets=len(raw_train_molecules&protected),
                                        train_root_ids_intersect_valid_root_ids=len(train_ids & set(frozen['roots'].values())))
        for split in ('training','validation'):
            count=0
            with (output/f'{split}.jsonl').open('w') as sink:
                for task_id,target,identity in db.execute('SELECT id,public_target,target FROM tasks WHERE split=? ORDER BY id',(split,)):
                    sink.write(json.dumps(dict(task_id=task_id,target=target),ensure_ascii=False)+'\n');count+=1
                    scaffold=_scaffold(identity)
                    if scaffold:scaffolds[split].add(scaffold)
            report[split+'_targets']=count
        if file_sha(output/'validation.jsonl')!=frozen['public_sha256']:
            raise ValueError('FROZEN_VALIDATION_PUBLIC_CHANGED')
        db.execute('CREATE INDEX routes_task_count_signature ON routes(task_id,reaction_count,signature)')
        with (output/'reference_witnesses.jsonl').open('w') as sink:
            for task_id, in db.execute('SELECT id FROM tasks ORDER BY id'):
                chosen=db.execute('SELECT signature,reaction_count FROM routes WHERE task_id=? ORDER BY reaction_count,signature LIMIT 1',(task_id,)).fetchone()
                sink.write(json.dumps(dict(task_id=task_id,route_signature=chosen[0],reaction_occurrences=chosen[1]))+'\n')
        report['private_reference_edges']=db.execute('SELECT count(*) FROM edges').fetchone()[0]
        report['private_source_witnesses']=db.execute('SELECT count(*) FROM route_sources').fetchone()[0]
        report['provenance_overlap']=dict(training_patents=len(patents['training']),validation_patents=len(patents['validation']),
                                          shared_patents=len(patents['training']&patents['validation']),
                                          training_nonempty_root_scaffolds=len(scaffolds['training']),
                                          validation_nonempty_root_scaffolds=len(scaffolds['validation']),
                                          shared_nonempty_root_scaffolds=len(scaffolds['training']&scaffolds['validation']),
                                          scaffold_definition='RDKit Murcko, chirality retained, empty acyclic scaffold excluded',
                                          patent_definition='union of source patent_ids and route annotation source_patent_ids; source-step identifiers retained privately')
    temporary.rename(output/'references.sqlite')
    (output/'stock.txt').write_text('\n'.join(sorted(stock))+'\n')
    atomic_json(output/'protected_identities.json',sorted(protected))
    atomic_json(output/'validation_protected_identities.json',sorted(protected))
    atomic_json(output/'training_protected_identities.json',sorted(protected|valid_roots))
    with (output/'reaction_frequencies.jsonl').open('w') as sink:
        for edge,count in sorted(reaction_frequency.items()):
            sink.write(json.dumps(dict(signature=edge,reference_occurrences=count))+'\n')
    report['route_distributions']={split:{key:dict(value) for key,value in stats.items()} for split,stats in route_stats.items()}
    report['excluded']=dict(report['excluded'])
    if report['source_records']!=report['accepted_routes']+report['duplicate_routes']+sum(report['excluded'].values()):
        raise ValueError('SOURCE_RECORD_ACCOUNTING_MISMATCH')
    if any(report['cross_split_audit'].values()):
        raise ValueError('V2_CROSS_SPLIT_ISOLATION_FAILURE')
    report['artifacts']={p.name:file_sha(p) for p in output.iterdir() if p.is_file()}
    atomic_json(output/'manifest.json',report)
    return report
