"""Public official-source downloads, verified before publication to the cache.

No credentials, license-gated scraping or undocumented mirrors. Binary source
datasets remain outside Git. CC-BY-SA ORD attribution travels in the manifest.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from html.parser import HTMLParser
from datetime import datetime,timezone


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch_verified(url, destination, expected_sha256):
    destination = Path(destination)
    if destination.exists():
        if file_sha256(destination) != expected_sha256:
            raise ValueError('existing source checksum mismatch: '+str(destination))
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_name(destination.name+'.part')
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as handle:
        for block in iter(lambda: response.read(1024 * 1024), b''):
            handle.write(block); digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('downloaded source checksum mismatch: '+url)
    os.replace(temporary,destination)
    return destination


def official_ord_catalog(mirror, github_tree):
    official = {item['path']:item['sha'] for item in github_tree
                if item.get('type')=='blob' and item['path'].startswith('data/')
                and '/ord_dataset-' in item['path'] and item['path'].endswith('.parquet')}
    files, excluded = [], []
    for item in mirror:
        if item['type'] != 'file':
            continue
        if item['path'] not in official:
            excluded.append(item['path'])
            continue
        lfs_sha = item.get('lfs',{}).get('oid')
        if not lfs_sha:
            raise ValueError('official Git LFS object missing in mirror')
        pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {item["size"]}\n'.encode()
        blob_sha = hashlib.sha1(b'blob '+str(len(pointer)).encode()+b'\0'+pointer).hexdigest()
        if blob_sha != official[item['path']]:
            raise ValueError('mirror differs from official Git LFS pointer: '+item['path'])
        files.append(item)
    if {item['path'] for item in files} != set(official):
        raise ValueError('official ORD files missing from mirror catalog')
    return files, sorted(excluded)


def download_ord(output, *, revision, github_revision, workers=8, github_catalog=None):
    if len(revision) != 40 or len(github_revision) != 40:
        raise ValueError('immutable 40-character source revisions required')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    endpoint = f'https://huggingface.co/api/datasets/open-reaction-database/ord-data/tree/{revision}/data?recursive=true&limit=1000'
    with urllib.request.urlopen(endpoint,timeout=45) as handle:
        catalog = json.load(handle)
        # Never silently accept a paginated prefix as the whole corpus.
        if handle.headers.get('Link'):
            raise ValueError('ORD catalog requires pagination; refusing partial download')
    github_url = f'https://api.github.com/repos/open-reaction-database/ord-data/git/trees/{github_revision}?recursive=1'
    if github_catalog is None:
        with urllib.request.urlopen(github_url,timeout=45) as handle:
            github_catalog = json.load(handle)
    if github_catalog.get('truncated'):
        raise ValueError('official Git tree was truncated')
    files, excluded = official_ord_catalog(catalog,github_catalog['tree'])
    if not files:
        raise ValueError('unexpected official ORD dataset format; recheck adapter')
    manifest = {'schema':'lgrv_ord_source_manifest_v1', 'official_repo':'https://github.com/open-reaction-database/ord-data',
                'official_read_mirror':'https://huggingface.co/datasets/open-reaction-database/ord-data',
                'github_revision':github_revision,'mirror_revision':revision,'license':'CC-BY-SA-4.0',
                'attribution':'Open Reaction Database contributors; upstream provenance retained per record',
                'download_complete':False,'files':[],'declared_file_count':len(files),
                'declared_bytes':sum(x['size'] for x in files),'catalog_url':endpoint,
                'official_git_catalog_url':github_url,'mirror_only_fixture_files_excluded':excluded,
                'file_eligibility':'every official Git ord_dataset Parquet; LFS pointer blob SHA agrees with mirror'}
    manifest_path = output/'source_corpus_manifest.json'
    def save():
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest,indent=2)+'\n')
        os.replace(temporary,manifest_path)
    save()
    def download(entry):
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe source path')
        url = f'https://huggingface.co/datasets/open-reaction-database/ord-data/resolve/{revision}/{relative.as_posix()}'
        path = fetch_verified(url,output/relative,entry['lfs']['oid'])
        if path.stat().st_size != entry['size']:
            raise ValueError('source byte count differs')
        return {'path':relative.as_posix(),'bytes':entry['size'],'sha256':entry['lfs']['oid'],'source_url':url}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download,entry) for entry in files]
        for future in as_completed(futures):
            manifest['files'].append(future.result())
            manifest['files'].sort(key=lambda row:row['path'])
            save()
            print(json.dumps({'phase':'download_ord','completed':len(manifest['files']),
                              'total':len(files),'bytes':sum(x['bytes'] for x in manifest['files'])}),flush=True)
    manifest['download_complete'] = True
    save()
    return manifest


def _smiles(identifiers):
    return next((item['value'] for item in identifiers if item.get('type')=='SMILES' and item.get('value')),None)


def ord_records(raw, *, dataset, row_index):
    """Adapt actual ORD protobuf JSON; do not guess missing substrate structures.

    Explicit ORD REACTANT roles are preserved. Mined mapped reaction identifiers
    are a second retrieval representation when available; their role/projection
    status is explicit, never automatically promoted to primary-source proof.
    """
    precursors, agents, missing = [], [], []
    input_roles = []
    for input_name, value in sorted(raw.get('inputs',{}).items()):
        for index, component in enumerate(value.get('components',[])):
            smiles = _smiles(component.get('identifiers',[]))
            role = component.get('reaction_role','UNSPECIFIED')
            input_roles.append({'input_name':input_name,'component_index':index,'role':role,'smiles':smiles})
            if role=='REACTANT':
                if smiles:
                    precursors.append(smiles)
                else:
                    missing.append(input_name)
            elif smiles:
                agents.append(smiles)
    identifiers = [item for item in raw.get('identifiers',[]) if item.get('type') in {'REACTION_SMILES','REACTION_CXSMILES'}]
    mapped = next((item for item in identifiers if item.get('is_mapped')),None)
    reaction_identifier = None
    if mapped:
        pieces = mapped['value'].split(' |',1)[0].split('>')
        if len(pieces)==3 and pieces[0] and pieces[2]:
            reaction_identifier = pieces
    if (missing or not precursors) and reaction_identifier is None:
        raise ValueError('missing structural reactant SMILES or explicit mapped reaction identifier')
    provenance = raw.get('provenance',{})
    patent = provenance.get('patent')
    doi = provenance.get('doi')
    url = ('https://patents.google.com/patent/'+patent+'/en') if patent else provenance.get('publication_url') or ('https://doi.org/'+doi if doi else None)
    procedure = raw.get('notes',{}).get('procedure_details','')
    common = {'license':'CC-BY-SA-4.0','source_corpus':'Open Reaction Database',
              'dataset_id':dataset['dataset_id'],'ord_reaction_id':raw.get('reaction_id'),
              'source_locator':{'parquet_row':row_index,'reaction_id':raw.get('reaction_id'),
                                'dataset_file_sha256':dataset['file_sha256'],'dataset_path':dataset.get('path')},
              'source_url':url,'patent_id':patent,'patent_family':None,'patent_family_status':'unresolved',
              'doi':doi,'is_mined':provenance.get('is_mined',False), 'input_roles':input_roles,
              'conditions':raw.get('conditions',{}),'procedure_sha256':hashlib.sha256(procedure.encode()).hexdigest(),
              'procedure_available':bool(procedure),'verification_status':'machine_structural_match',
              'performed_example_verified':False,'training_source_overlap':'unknown',
              'missing_declared_reactant_smiles':missing}
    records = []
    for outcome_index,outcome in enumerate(raw.get('outcomes',[])):
        products = outcome.get('products',[])
        desired = [p for p in products if p.get('is_desired_product') is True]
        selected = desired or products
        product_smiles = [_smiles(p.get('identifiers',[])) for p in selected]
        if not product_smiles or any(p is None for p in product_smiles):
            continue
        product = '.'.join(product_smiles)
        if not missing and precursors:
            records.append({**common,'source_id':f'ORD:{dataset["dataset_id"]}:{raw["reaction_id"]}:outcome{outcome_index}:roles',
                            'product':product,'precursors':precursors,'agents':agents,
                            'role_basis':'ORD_explicit_REACTANT_roles', 'outcome_index':outcome_index,
                            'measurements':[p.get('measurements',[]) for p in selected]})
    if reaction_identifier:
        left, middle, right = reaction_identifier
        records.append({**common,'source_id':f'ORD:{dataset["dataset_id"]}:{raw["reaction_id"]}:mapped_identifier',
                        'product':right,'precursors':[left],'agents':[middle] if middle else [],
                        'role_basis':'ORD_extracted_reaction_identifier_not_primary_verified',
                        'outcome_index':None,'mapped_identifier_raw':mapped['value']})
    if not records:
        raise ValueError('ORD outcome lacks a complete structural product')
    return records


def iter_ord_messages(path, decoder, *, batch_size=1024):
    """Stream actual Parquet messages; row offsets never reset at row groups."""
    import pyarrow.parquet as pq
    path = Path(path)
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    dataset = {'dataset_id':metadata[b'ord.dataset_id'].decode(),
               'name':metadata.get(b'ord.name',b'').decode(),'path':str(path),
               'file_sha256':file_sha256(path)}
    row_index = 0
    for batch in parquet.iter_batches(batch_size=batch_size,columns=['reaction_id','reaction']):
        for row in batch.to_pylist():
            yield row_index,dataset,decoder(row['reaction'])
            row_index += 1


def normalize_primary_text(payload, encoding='utf-8'):
    class ExtractText(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True);self.parts=[]
        def handle_data(self,value):
            if value.strip():
                self.parts.append(value)
    extractor=ExtractText();extractor.feed(payload.decode(encoding,errors='replace'))
    return ' '.join(' '.join(extractor.parts).split())


def snapshot_primary_document(url, output_root):
    """Cache primary bytes once and hash normalized text; never publish full text.

    Normalization only collapses HTML whitespace, not chemical identities. A
    checker must still inspect a specific performed example and its structures.
    """
    root = Path(output_root);root.mkdir(parents=True,exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    receipt_path = root/(key+'.json')
    if receipt_path.exists():
        record = json.loads(receipt_path.read_bytes())
        if file_sha256(record['raw_document_path'])!=record['raw_document_sha256'] or file_sha256(record['document_path'])!=record['document_sha256']:
            raise ValueError('cached primary document checksum mismatch')
        return record
    with urllib.request.urlopen(url,timeout=90) as response:
        payload = response.read(32*1024*1024+1)
        encoding = response.headers.get_content_charset() or 'utf-8'
    if len(payload)>32*1024*1024:
        raise ValueError('primary document exceeds safe memory buffer; use a streaming source adapter')
    text = normalize_primary_text(payload,encoding)
    raw_path = root/(key+'.html');text_path = root/(key+'.txt')
    raw_path.write_bytes(payload);text_path.write_text(text,encoding='utf-8')
    record = {'source_url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),
              'raw_document_path':str(raw_path.resolve()),'raw_document_sha256':file_sha256(raw_path),
              'document_path':str(text_path.resolve()),'document_sha256':file_sha256(text_path),
              'source_encoding':encoding,
              'text_normalization':'html_data_join_unicode_whitespace_v1',
              'source_passage_checked':False,'redistribute_full_document':False}
    temporary = receipt_path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n')
    os.replace(temporary,receipt_path)
    return record
