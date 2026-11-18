"""Use FLA's supported core-only import without loading optional model zoos.

FLA0.5.2 treats absent fla.layers/models as optional extensions. Qwen imports
fla.modules and fla.ops directly. Temporarily hiding only those two optional
registries exercises the ordinary upstream fallback; no tensor code is patched.
"""
import importlib
from importlib.abc import MetaPathFinder
import sys
from threading import RLock

_lock=RLock()


class _OptionalRegistries(MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ('fla.layers','fla.models'):
            raise ModuleNotFoundError('Optional FLA model registry excluded during core startup',name=fullname)
        return None


def import_core_fla():
    with _lock:
        if 'fla' in sys.modules:return sys.modules['fla']
        blocker=_OptionalRegistries();sys.meta_path.insert(0,blocker)
        try:return importlib.import_module('fla')
        finally:sys.meta_path.remove(blocker)
