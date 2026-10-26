"""Shared byte offsets, not another expanded decision dataset.

Only the index builder parses all JSON once. Readers seek exact original UTF-8
bytes for the current round. An ID must have a contiguous history block; otherwise
fail instead of silently dropping an earlier block. Publication is atomic.
"""
import json
from pathlib import Path
import sqlite3
import tempfile
import time


def build_index(source, destination, *, grouped, expected_reactions):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name+'.building-', dir=destination.parent)
    import os
    os.close(handle)
    db = sqlite3.connect(temporary)
    try:
        db.execute('CREATE TABLE offsets (rid TEXT PRIMARY KEY, begin INTEGER, end INTEGER)')
        start_stat = source.stat()
        previous, begin, end, count, records = None, 0, 0, 0, 0
        def store(rid, first, last):
            try:
                db.execute('INSERT INTO offsets VALUES (?,?,?)', (rid,first,last))
            except sqlite3.IntegrityError as exc:
                raise ValueError('EARHO_NONCONTIGUOUS_OR_DUPLICATE_ID:'+rid) from exc
        tick = time.monotonic()
        with source.open('rb') as reader:
            while True:
                position = reader.tell()
                line = reader.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = str(row['source_id'])
                if previous == rid and not grouped:
                    raise ValueError('EARHO_DUPLICATE_SOURCE_ID:'+rid)
                if previous != rid:
                    if previous is not None:
                        store(previous,begin,end)
                        count += 1
                    previous, begin = rid, position
                end = reader.tell()
                records += 1
                if time.monotonic()-tick >= 30:
                    print(json.dumps(dict(event='earho-index-progress', source=str(source),
                          records=records, bytes_read=end, total_bytes=start_stat.st_size)),flush=True)
                    tick = time.monotonic()
        if previous is not None:
            store(previous,begin,end)
            count += 1
        if count != expected_reactions:
            raise ValueError('EARHO_INDEX_DENOMINATOR_CHANGED')
        end_stat = source.stat()
        if (end_stat.st_size,end_stat.st_mtime_ns) != (start_stat.st_size,start_stat.st_mtime_ns):
            raise ValueError('EARHO_INDEX_SOURCE_CHANGED_DURING_BUILD')
        db.execute('CREATE TABLE metadata (value TEXT)')
        db.execute('INSERT INTO metadata VALUES (?)', (json.dumps(dict(
            size=start_stat.st_size, mtime_ns=start_stat.st_mtime_ns,
            reactions=count, records=records)),))
        db.commit()
        db.close()
        Path(temporary).replace(destination)
    except BaseException:
        db.close()
        raise


class IndexedJsonl:
    def __init__(self, source, index):
        self.source = Path(source)
        self.db = sqlite3.connect(f'file:{Path(index).resolve()}?mode=ro', uri=True)
        metadata = json.loads(self.db.execute('SELECT value FROM metadata').fetchone()[0])
        stat = self.source.stat()
        if (metadata['size'],metadata['mtime_ns']) != (stat.st_size,stat.st_mtime_ns):
            self.db.close()
            raise ValueError('EARHO_INDEX_SOURCE_CHANGED')
        self.reader = self.source.open('rb')

    def ids(self):
        return [r[0] for r in self.db.execute('SELECT rid FROM offsets ORDER BY begin')]

    def get(self, rid):
        record = self.db.execute('SELECT begin,end FROM offsets WHERE rid=?',(str(rid),)).fetchone()
        if record is None:
            raise KeyError(rid)
        first,last = record
        self.reader.seek(first)
        rows = [json.loads(line) for line in self.reader.read(last-first).splitlines() if line.strip()]
        if any(str(row['source_id']) != str(rid) for row in rows):
            raise ValueError('EARHO_INDEX_WRONG_REACTION')
        return rows

    def close(self):
        self.reader.close()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
