#!/usr/bin/env python3
"""Validate internal documentation links, commands, configs, and CLI entrypoints."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_HTML_LINK = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']", re.IGNORECASE)
_FENCE = re.compile(r"```(?:bash|sh|shell|console)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CONFIG_PATH = re.compile(r"(?<![\w./-])(configs/[A-Za-z0-9_./-]+\.ya?ml)")
_SCRIPT_PATH = re.compile(r"(?<![\w./-])(scripts/[A-Za-z0-9_./-]+\.py)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def github_anchor(text: str) -> str:
    value = re.sub(r"<[^>]+>", "", text.strip().lower())
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def heading_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for _marks, heading in _HEADING.findall(text):
        base = github_anchor(heading)
        if not base:
            continue
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def markdown_files(root: Path) -> list[Path]:
    values = [root / "README.md"]
    values.extend(sorted((root / "docs").rglob("*.md")))
    values.extend(sorted((root / "knowledge").glob("*.md")))
    return [path for path in values if path.exists()]


def _target_parts(raw: str) -> tuple[str, str]:
    value = raw.strip().split()[0].strip("<>")
    parsed = urlsplit(value)
    return unquote(parsed.path), unquote(parsed.fragment)


def _resolve_link(source: Path, raw: str, root: Path) -> tuple[Path | None, str]:
    value = raw.strip()
    if not value or value.startswith(("http://", "https://", "mailto:", "data:")):
        return None, ""
    path_value, fragment = _target_parts(value)
    if path_value.startswith("/"):
        target = root / path_value.lstrip("/")
    elif path_value:
        target = (source.parent / path_value).resolve()
    else:
        target = source.resolve()
    return target, fragment


def _logical_shell_lines(block: str) -> Iterable[str]:
    current = ""
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "$")):
            continue
        current = f"{current} {line}".strip() if current else line
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        yield current
        current = ""
    if current:
        yield current


def _python_script_from_command(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None
    while tokens and "=" in tokens[0] and not tokens[0].startswith(("./", "/")):
        tokens.pop(0)
    if tokens and tokens[0] in {"python", "python3", sys.executable}:
        tokens.pop(0)
    if tokens and tokens[0] == "-m":
        return None
    if tokens and tokens[0].startswith("scripts/") and tokens[0].endswith(".py"):
        return tokens[0]
    return None


def _script_contract(path: Path) -> list[str]:
    failures: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    has_main = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
        for node in tree.body
    )
    has_entrypoint = "if __name__" in path.read_text(encoding="utf-8")
    if not has_main:
        failures.append("missing main()")
    if not has_entrypoint:
        failures.append("missing __main__ entrypoint")
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if compile_result.returncode:
        failures.append(compile_result.stderr.strip() or "py_compile failed")
    return failures


def check(root: Path, *, check_cli_help: bool = False) -> dict[str, Any]:
    files = markdown_files(root)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    referenced_scripts: set[str] = set()
    referenced_configs: set[str] = set()

    anchor_cache = {
        path.resolve(): heading_anchors(path.read_text(encoding="utf-8"))
        for path in files
    }
    for source in files:
        text = source.read_text(encoding="utf-8")
        for raw in [*_MARKDOWN_LINK.findall(text), *_HTML_LINK.findall(text)]:
            target, fragment = _resolve_link(source, raw, root)
            if target is None:
                continue
            try:
                target.relative_to(root.resolve())
            except ValueError:
                errors.append(
                    {
                        "type": "link_escapes_repository",
                        "source": str(source.relative_to(root)),
                        "target": raw,
                    }
                )
                continue
            if not target.exists():
                errors.append(
                    {
                        "type": "missing_internal_link",
                        "source": str(source.relative_to(root)),
                        "target": raw,
                    }
                )
                continue
            if fragment and target.suffix.lower() == ".md":
                anchors = anchor_cache.get(target.resolve())
                if anchors is None:
                    anchors = heading_anchors(target.read_text(encoding="utf-8"))
                    anchor_cache[target.resolve()] = anchors
                if github_anchor(fragment) not in anchors:
                    errors.append(
                        {
                            "type": "missing_markdown_anchor",
                            "source": str(source.relative_to(root)),
                            "target": raw,
                        }
                    )

        referenced_scripts.update(_SCRIPT_PATH.findall(text))
        referenced_configs.update(_CONFIG_PATH.findall(text))
        for block in _FENCE.findall(text):
            for command in _logical_shell_lines(block):
                script = _python_script_from_command(command)
                if script:
                    referenced_scripts.add(script)

    for script in sorted(referenced_scripts):
        path = root / script
        if not path.exists():
            errors.append({"type": "missing_documented_script", "path": script})
            continue
        for failure in _script_contract(path):
            errors.append(
                {
                    "type": "invalid_documented_script",
                    "path": script,
                    "error": failure,
                }
            )
        if check_cli_help:
            result = subprocess.run(
                [sys.executable, str(path), "--help"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode:
                errors.append(
                    {
                        "type": "documented_cli_help_failed",
                        "path": script,
                        "error": result.stderr.strip() or result.stdout.strip(),
                    }
                )

    yaml_module = None
    try:
        import yaml as yaml_module  # type: ignore
    except ImportError:
        warnings.append(
            {
                "type": "yaml_parse_skipped",
                "reason": "PyYAML is not installed",
            }
        )
    for config in sorted(referenced_configs):
        path = root / config
        if not path.exists():
            errors.append({"type": "missing_documented_config", "path": config})
            continue
        if yaml_module is not None:
            try:
                yaml_module.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(
                    {
                        "type": "invalid_documented_config",
                        "path": config,
                        "error": str(exc),
                    }
                )

    return {
        "artifact_type": "documentation_integrity_report",
        "n_markdown_files": len(files),
        "n_referenced_scripts": len(referenced_scripts),
        "n_referenced_configs": len(referenced_configs),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-cli-help", action="store_true")
    args = parser.parse_args()
    result = check(args.root.resolve(), check_cli_help=args.check_cli_help)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
