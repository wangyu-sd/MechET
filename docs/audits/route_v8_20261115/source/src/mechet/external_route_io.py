"""Coalesce Ceph receipts without weakening GPU-before-consume durability."""
import json
import os
from pathlib import Path

from .route_policy_sampling import NativeRouteSampler
from .route_phase_profile import phase


class ReceiptBatch:
    def __init__(self,path,cfg=None):
        self.path=Path(path)
        self.cfg=cfg
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.pending=[]

    def _write(self,rows):
        with self.path.open('a') as sink:
            with phase(self.cfg,'io.journal_write',rows=len(rows)):
                for row in rows:
                    sink.write(json.dumps(row,ensure_ascii=False)+'\n')
            with phase(self.cfg,'io.journal_flush'):
                sink.flush();os.fsync(sink.fileno())

    def append(self,row):
        if row.get('event') in ('gpu-requested','gpu-returned'):
            self.flush()
            self._write([row])
        else:
            self.pending.append(row)

    def flush(self):
        if self.pending:
            rows,self.pending=self.pending,[]
            self._write(rows)


class BatchedNativeRouteSampler(NativeRouteSampler):
    def __init__(self,model,tokenizer,cfg,raw_path,execution_writer):
        self.writer=ReceiptBatch(raw_path,cfg=cfg)
        self.execution_writer=execution_writer
        super().__init__(model,tokenizer,cfg,raw_path,receipt_writer=self.writer.append)

    def generate(self,requests):
        self.execution_writer.flush()
        try:
            return super().generate(requests)
        finally:
            self.writer.flush()
