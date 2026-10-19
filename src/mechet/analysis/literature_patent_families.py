"""Bibliographic family identities authenticated against original cached pages.

This overlay leaves the frozen ORD index untouched. Only a page's own Family
section supplies its family identity; cited patents, guessed publication kinds
and chemical similarity cannot establish family membership or evidence grades.
"""
import hashlib
import json
from pathlib import Path
import re

from .literature_index import source_family
from .literature_sources import file_sha256, normalize_primary_text


def _patent_id(value):
    match = re.fullmatch(r'US([0-9]+)([A-Z][0-9]*)?', value or '')
    if match and int(match[1]):
        return 'US' + str(int(match[1])) + (match[2] or '')
    return value


class PatentFamilyResolver:
    def __init__(self, families, manifest):
        self.families = dict(families)
        self.manifest = manifest

    def resolve(self, record):
        family = self.families.get(_patent_id(record.get('patent_id')))
        if family:
            known = record.get('patent_family')
            if known and str(known) not in {family, family.split(':', 1)[1]}:
                raise ValueError('record and original-source patent family conflict')
            return family
        return source_family(record)


def load_family_registry(path):
    """Load a sealed receipt list, rechecking source bytes and normalization."""
    path = Path(path).resolve()
    spec = json.loads(path.read_bytes())
    if spec.get('schema') != 'lgrv_patent_family_sources_v1' or not isinstance(spec.get('receipts'), list):
        raise ValueError('unsupported patent-family source registry')
    families = {}; documents = []; unresolved = 0
    for entry in spec['receipts']:
        receipt_path = Path(entry['path'])
        if not receipt_path.is_absolute():
            receipt_path = path.parent / receipt_path
        if file_sha256(receipt_path) != entry['sha256']:
            raise ValueError('patent-family receipt checksum differs')
        receipt = json.loads(receipt_path.read_bytes())
        url = receipt.get('source_url', '')
        match = re.fullmatch(r'https://patents\.google\.com/patent/([A-Z]{2}[0-9]+(?:[A-Z][0-9]*)?)/en', url)
        if not match or receipt_path.name != hashlib.sha256(url.encode()).hexdigest() + '.json':
            raise ValueError('patent-family source URL and receipt identity differ')
        raw = Path(receipt['raw_document_path']).read_bytes()
        text = Path(receipt['document_path']).read_bytes()
        if (hashlib.sha256(raw).hexdigest() != receipt['raw_document_sha256'] or
                hashlib.sha256(text).hexdigest() != receipt['document_sha256'] or
                receipt.get('text_normalization') != 'html_data_join_unicode_whitespace_v1' or
                normalize_primary_text(raw, receipt.get('source_encoding', 'utf-8')).encode() != text):
            raise ValueError('patent-family original document identity differs')
        html = raw.decode(receipt.get('source_encoding', 'utf-8'), errors='replace')
        sections = re.findall(r'<section\b[^>]*\bitemprop=["\']family["\'][^>]*>(.*?)</section\s*>', html, re.S)
        ids = set(re.findall(r'<h2>\s*ID=([0-9]+)\s*</h2>', sections[0])) if len(sections) == 1 else set()
        family = 'GOOGLE_PATENTS_FAMILY:' + next(iter(ids)) if len(ids) == 1 else None
        patent = _patent_id(match[1])
        if family:
            if patent in families and families[patent] != family:
                raise ValueError('original patent-family metadata conflict')
            families[patent] = family
        else:
            unresolved += 1
        documents.append({'patent_id': patent, 'source_url': url, 'family_key': family,
                          'receipt_path': str(receipt_path.resolve()), 'receipt_sha256': entry['sha256'],
                          'raw_document_sha256': receipt['raw_document_sha256'],
                          'document_sha256': receipt['document_sha256'],
                          'locator': 'own section[itemprop=family] h2 ID',
                          'status': 'source_metadata_checked' if family else 'unresolved'})
    manifest = {'schema': 'lgrv_patent_family_overlay_v1', 'registry_path': str(path),
                'registry_sha256': file_sha256(path), 'producer_sha256': file_sha256(__file__),
                'document_count': len(documents), 'family_count': len(set(families.values())),
                'resolved_publications': len(families), 'unresolved_documents': unresolved,
                'unknown_family_policy': 'publication_fallback_explicitly_unresolved',
                'publication_kind_inferred': False, 'chemical_evidence_grades_changed': False,
                'documents': documents}
    return PatentFamilyResolver(families, manifest)
