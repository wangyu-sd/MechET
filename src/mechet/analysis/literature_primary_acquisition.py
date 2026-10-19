"""Acquire public primary documents without changing frozen source identities.

ORD contains zero-padded US grant identifiers that Google Patents does not
resolve. Normalization repairs a lookup address only: it does not infer a
publication kind, patent family, performed example, or chemical evidence tier.
Full documents stay in the external cache, never in the Git distribution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import urllib.error
from urllib.parse import urlsplit

from .literature_sources import file_sha256, snapshot_primary_document


def patent_reference(raw_id, original_url):
    """Keep the original reference and derive a public, non-guessed lookup."""
    if not isinstance(raw_id, str) or not re.fullmatch(r'[A-Z]{2}[0-9]+(?:[A-Z][0-9]*)?', raw_id):
        raise ValueError('invalid patent identifier')
    parsed = urlsplit(original_url)
    if (parsed.scheme != 'https' or parsed.netloc != 'patents.google.com' or
            parsed.path != '/patent/' + raw_id + '/en' or parsed.query or parsed.fragment):
        raise ValueError('patent identifier and public source URL differ')
    lookup_id = raw_id
    if raw_id.startswith('US'):
        match = re.fullmatch(r'US([0-9]+)([A-Z][0-9]*)?', raw_id)
        number = match[1].lstrip('0')
        if not number:
            raise ValueError('invalid zero patent number')
        lookup_id = 'US' + number + (match[2] or '')
    return {'original_patent_id': raw_id, 'original_source_url': original_url,
            'lookup_patent_id': lookup_id,
            'lookup_url': 'https://patents.google.com/patent/' + lookup_id + '/en',
            'publication_kind_inferred': False, 'patent_family_inferred': False}


def acquire(hits_path, cache_root, output_root, *, workers=4):
    """Attempt every distinct public patent URL, saving progress per completion.

    External HTTP failures are recorded once, not retried or bypassed. A local
    checksum/provenance error aborts and leaves complete=false. Non-patent rows
    remain in an explicit deferred queue rather than silently disappearing.
    """
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('invalid document acquisition worker count')
    hits_path, cache_root, output_root = map(Path, (hits_path, cache_root, output_root))
    grouped = {}; deferred = []; count = 0
    with gzip.open(hits_path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line); count += 1
            if not row.get('source_id') or not row.get('query_reaction_key'):
                raise ValueError('source candidate identity missing')
            if not row.get('patent_id'):
                deferred.append({'reason': 'not_a_patent_reference', 'input_reference': row})
                continue
            reference = patent_reference(row['patent_id'], row['source_url'])
            grouped.setdefault(reference['lookup_url'], []).append({**row, **reference})
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'lgrv_primary_document_acquisition_v1', 'complete': False,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'input_hits_path': str(hits_path.resolve()), 'input_hits_sha256': file_sha256(hits_path),
                'producer_sha256': file_sha256(__file__),
                'cache_producer_sha256': file_sha256(Path(__file__).with_name('literature_sources.py')),
                'cache_root': str(cache_root.resolve()), 'input_source_records': count,
                'unique_lookup_urls': len(grouped), 'cached': 0, 'unavailable': 0,
                'completed_lookups': 0, 'deferred_records': deferred, 'records': [],
                'source_passages_checked': False, 'new_evidence_grades': False,
                'publication_kind_inferred': False, 'patent_family_inferred': False,
                'redistribute_full_documents': False}
    path = output_root / 'acquisition_manifest.json'

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)

    def fetch(url):
        record = {'lookup_url': url, 'input_references': grouped[url]}
        try:
            receipt = snapshot_primary_document(url, cache_root)
        except (urllib.error.URLError, TimeoutError) as error:
            record.update(status='unavailable', error_type=type(error).__name__, error=str(error))
            if isinstance(error, urllib.error.HTTPError):
                record['http_status'] = error.code
        else:
            record.update(status='document_cached', receipt=receipt)
        return record

    save()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, url): url for url in sorted(grouped)}
        for future in as_completed(futures):
            record = future.result()
            manifest['records'].append(record)
            manifest['records'].sort(key=lambda item: item['lookup_url'])
            manifest['completed_lookups'] += 1
            manifest['cached' if record['status'] == 'document_cached' else 'unavailable'] += 1
            save()
            print(json.dumps({'phase': 'primary_document_acquisition',
                              **{k: manifest[k] for k in ('completed_lookups', 'unique_lookup_urls',
                                                          'cached', 'unavailable')}}), flush=True)
    manifest['complete'] = True
    manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('literature-hits', 'cache-root', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    acquire(args.literature_hits, args.cache_root, args.output_root, workers=args.workers)


if __name__ == '__main__':
    main()
