"""Bounded, rollback-safe appends to newly owned preprocessing journals.

No existing file can be adopted. On transient I/O errors only this writer's
uncommitted tail is rolled back; source artifacts are never opened for writing.
"""
import errno
import os
from pathlib import Path
import time


class DurableAppendSet:
    def __init__(self, paths, *, max_attempts=3, on_retry=None):
        self.paths = {k: Path(p) for k, p in paths.items()}
        if len({str(p.resolve()) for p in self.paths.values()}) != len(self.paths):
            raise ValueError("DUPLICATE_JOURNAL_PATH")
        self.max_attempts = max_attempts
        self.on_retry = on_retry or (lambda event: print(event, flush=True))
        self.fds, self.identities, self.offsets = {}, {}, {}
        try:
            for key, path in self.paths.items():
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
                self.fds[key] = fd
                stat = os.fstat(fd)
                self.identities[key] = (stat.st_dev, stat.st_ino)
                self.offsets[key] = 0
        except BaseException:
            self.close()
            raise

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _rollback(self):
        # Reopen after EIO: discard a failed descriptor, but never follow a
        # replacement inode or truncate below the previously committed prefix.
        for key, old_fd in list(self.fds.items()):
            os.close(old_fd)
            del self.fds[key]
            fd = os.open(self.paths[key], os.O_RDWR | os.O_NOFOLLOW)
            self.fds[key] = fd
            stat = os.fstat(fd)
            if (stat.st_dev, stat.st_ino) != self.identities[key] or stat.st_size < self.offsets[key]:
                raise ValueError("COMMITTED_JOURNAL_IDENTITY_OR_PREFIX_LOST")
            os.ftruncate(fd, self.offsets[key])
            os.lseek(fd, self.offsets[key], os.SEEK_SET)

    def commit(self, payloads):
        if set(payloads) - set(self.fds) or any(not isinstance(v, bytes) for v in payloads.values()):
            raise ValueError("INVALID_JOURNAL_PAYLOAD")
        for attempt in range(1, self.max_attempts + 1):
            try:
                for key, payload in payloads.items():
                    view = memoryview(payload)
                    while view:
                        written = os.write(self.fds[key], view)
                        if written <= 0:
                            raise OSError(errno.EIO, "zero-length journal write")
                        view = view[written:]
                for fd in self.fds.values():
                    os.fsync(fd)
                self.offsets = {k: self.offsets[k] + len(payloads.get(k, b"")) for k in self.fds}
                return dict(self.offsets)
            except OSError as exc:
                self._rollback()
                if exc.errno not in (errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EINTR) or attempt == self.max_attempts:
                    raise
                self.on_retry({"stage": "journal-io-retry", "attempt": attempt,
                               "errno": exc.errno, "error": str(exc), "committed_offsets": dict(self.offsets)})
                time.sleep(min(attempt, 3))
