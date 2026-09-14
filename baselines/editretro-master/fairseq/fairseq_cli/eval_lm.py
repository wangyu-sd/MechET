"""Compatibility shim for source archives that flattened this symlink."""

from pathlib import Path


target = Path(__file__).resolve().parents[1] / "eval_lm.py"
exec(compile(target.read_bytes(), str(target), "exec"), globals(), globals())
