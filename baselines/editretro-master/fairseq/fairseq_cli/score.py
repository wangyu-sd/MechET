"""Compatibility shim for source archives that flattened this symlink."""

from pathlib import Path


target = Path(__file__).resolve().parents[1] / "score.py"
exec(compile(target.read_bytes(), str(target), "exec"), globals(), globals())
if "cli_main" not in globals():
    cli_main = main
