"""Disk-backed, deterministic structure retrieval with explicit source identity."""
from __future__ import annotations

import json
import sqlite3

from .literature_reaction_keys import make_reaction


def source_family(record):
    identity = record.get('patent_family') or record.get('patent_id') or record.get('doi')
    if identity:
        return identity
    if record.get('dataset_id') and record.get('ord_reaction_id'):
        return f'ORD:{record["dataset_id"]}:{record["ord_reaction_id"]}'
    return record['source_id']


class ReactionIndex:
    def __init__(self, path, *, readonly=False, family_resolver=None):
        self.family_resolver = family_resolver
        self.connection = sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True) if readonly else sqlite3.connect(path)
        if not readonly:
            self.connection.executescript('''
                CREATE TABLE IF NOT EXISTS reactions (
                    source_id TEXT PRIMARY KEY, reaction_key TEXT NOT NULL,
                    family_key TEXT, source_family TEXT NOT NULL, record TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS exact_key ON reactions(reaction_key);
                CREATE INDEX IF NOT EXISTS family_key ON reactions(family_key);
            ''')

    def add(self, record):
        if not record.get('source_id') or not record.get('license'):
            raise ValueError('source ID and license required')
        reaction = make_reaction(record['product'], record['precursors'], auxiliary=record.get('agents', []))
        family = source_family(record)
        payload = {**record, 'reaction_key':reaction['reaction_key'],
                   'reaction_family_key':reaction['reaction_family_key']}
        old = self.connection.execute('SELECT record FROM reactions WHERE source_id=?', (record['source_id'],)).fetchone()
        serialized = json.dumps(payload,sort_keys=True)
        if old and old[0] != serialized:
            raise ValueError('source ID reused with changed payload: '+record['source_id'])
        self.connection.execute('INSERT OR IGNORE INTO reactions VALUES (?,?,?,?,?)',
            (record['source_id'],reaction['reaction_key'],reaction['reaction_family_key'],family,serialized))

    def query(self, reaction, *, cap=5, include_analogues=True):
        if not 1 <= cap <= 5:
            raise ValueError('LGRV primary retrieval cap must be between 1 and 5')
        exact = self.connection.execute('SELECT source_family,record FROM reactions WHERE reaction_key=? ORDER BY source_id',
                                        (reaction['reaction_key'],)).fetchall()
        analogous = []
        if include_analogues and reaction.get('reaction_family_key'):
            analogous = self.connection.execute('SELECT source_family,record FROM reactions WHERE family_key=? AND reaction_key!=? ORDER BY source_id',
                (reaction['reaction_family_key'],reaction['reaction_key'])).fetchall()
        result, families = [], {}
        for match, candidates in [('exact',exact),('local_edit_analogue_candidate',analogous)]:
            for _, payload in candidates:
                record = json.loads(payload)
                # Derive from preserved provenance even for v1 disk indices;
                # representation aliases must never consume independent slots.
                family = self.source_family(record)
                if family in families:
                    families[family]['source_representation_ids'].append(record['source_id'])
                    continue
                if len(result)>=cap:
                    continue
                selected={**record,'match_type':match,'source_family_key':family,
                          'source_representation_ids':[record['source_id']],
                          'retrieval_status':'indexed_candidate','verification_status':'machine_structural_match'}
                families[family]=selected;result.append(selected)
        return result

    def source_family(self, record):
        resolver = getattr(self, 'family_resolver', None)
        return resolver.resolve(record) if resolver is not None else source_family(record)

    def count(self):
        return self.connection.execute('SELECT count(*) FROM reactions').fetchone()[0]

    def commit(self):
        self.connection.commit()

    def close(self):
        self.connection.commit()
        self.connection.close()
