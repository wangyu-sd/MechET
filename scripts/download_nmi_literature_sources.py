#!/usr/bin/env python3
"""Download checksum-pinned official ORD data (including Lowe patent shards)."""
import argparse
from pathlib import Path
import sys
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mechet.analysis.literature_sources import download_ord


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--github-revision',required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--github-client',choices=['public','gh'],default='public',
                        help='Use configured gh authentication for the official public Git catalog when needed')
    args = parser.parse_args()
    catalog = None
    if args.github_client == 'gh':
        catalog = json.loads(subprocess.check_output(['gh','api',
            f'repos/open-reaction-database/ord-data/git/trees/{args.github_revision}?recursive=1']))
    download_ord(args.output_root,revision=args.revision,github_revision=args.github_revision,
                 workers=args.workers,github_catalog=catalog)


if __name__ == '__main__':
    main()
