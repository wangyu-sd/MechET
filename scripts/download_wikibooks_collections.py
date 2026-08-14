#!/usr/bin/env python3
"""Download every registered Wikibooks dump-prefix collection into one snapshot."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("knowledge/source_registry.yaml"))
    parser.add_argument("--output", type=Path, default=Path("knowledge/raw_corpus_v2"))
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=Path("knowledge/downloads/enwikibooks-latest-pages-articles.xml.bz2"),
    )
    parser.add_argument("--source-id", action="append", default=[])
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    available = {
        source_id: source
        for source_id, source in registry["sources"].items()
        if source.get("downloader") == "wikimedia_xml_dump" and source.get("dump_prefix")
    }
    selected = args.source_id or sorted(available)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise KeyError(f"sources lack a registered dump_prefix: {missing}")
    script = Path(__file__).with_name("download_wikibooks_dump.py")
    for source_id in selected:
        source = available[source_id]
        command = [
            sys.executable,
            str(script),
            "--registry",
            str(args.registry),
            "--source-id",
            source_id,
            "--prefix",
            str(source["dump_prefix"]),
            "--dump-path",
            str(args.dump_path),
            "--output",
            str(args.output),
        ]
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
