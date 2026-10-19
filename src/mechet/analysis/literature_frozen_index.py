"""Fast reads of a completed, SHA-pinned, never-written source index.

SQLite immutable=1 skips remote locking/change-detection calls. It is NOT for
an active builder or shared writable database. Source identity is checked
before open and again at close; a changed file cannot seal analysis outputs.
See https://www.sqlite.org/uri.html#uriimmutable .
"""
from pathlib import Path
import sqlite3

from .literature_index import ReactionIndex
from .literature_sources import file_sha256


class FrozenReactionIndex(ReactionIndex):
    def __init__(self,path,*,index_sha256,family_resolver=None):
        self.family_resolver=family_resolver
        self.path=Path(path).resolve()
        if not self.path.is_file():raise ValueError('completed frozen source index missing')
        self.identity=self._identity()
        if file_sha256(self.path)!=index_sha256 or self._identity()!=self.identity:
            raise ValueError('frozen source index checksum or file identity differs')
        self.connection=sqlite3.connect(self.path.as_uri()+'?mode=ro&immutable=1',uri=True)

    def _identity(self):
        stat=self.path.stat()
        return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns

    def close(self):
        try:
            if self._identity()!=self.identity:
                raise ValueError('source index changed during immutable read analysis')
        except OSError as error:
            raise ValueError('source index disappeared during immutable read analysis') from error
        finally:
            self.connection.close()
