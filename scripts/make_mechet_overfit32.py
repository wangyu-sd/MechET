#!/usr/bin/env python3
"""Carve a tiny overfit32 slice for MechET SFT smoke tests.

This is NOT a formal split. It samples a topology-balanced subset from an
existing MECH_ET JSONL (default: valid.jsonl) so ``configs/overfit32.yaml``
can verify the train loop before running ``sft_pilot.yaml``.

Example:
  python scripts/make_mechet_overfit32.py \\
    --src data/mechet_sft/valid.jsonl \\
    --out-dir data/mechet_sft/overfit32
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TOPO_ORDER = ("linear", "tree", "dag_branch_join")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _topology(row: dict) -> str:
    topo = (row.get("metadata") or {}).get("topology") or "unknown"
    return str(topo)


def _quota(n: int, keys: list[str]) -> dict[str, int]:
    """Near-equal integer quotas that sum to n."""
    if not keys:
        return {}
    base, rem = divmod(n, len(keys))
    out = {k: base for k in keys}
    for k in keys[:rem]:
        out[k] += 1
    return out


def _take(pool: list[dict], need: int, rng: random.Random) -> list[dict]:
    if need <= 0 or not pool:
        return []
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled[:need]


def select_overfit(
    rows: list[dict],
    *,
    n_train: int,
    n_valid: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    by_topo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_topo[_topology(row)].append(row)

    # Prefer the three canonical FlowER topologies; fall back to whatever exists.
    keys = [k for k in TOPO_ORDER if by_topo.get(k)]
    if not keys:
        keys = sorted(by_topo.keys())
    if not keys:
        raise SystemExit("no rows to sample")

    train_q = _quota(n_train, keys)
    valid_q = _quota(n_valid, keys)

    train: list[dict] = []
    valid: list[dict] = []
    used_ids: set[str] = set()

    def _id(row: dict) -> str:
        return str(row.get("id") or id(row))

    # Valid first (held-out), then train from remaining — keeps smoke eval disjoint.
    for topo in keys:
        pool = [r for r in by_topo[topo] if _id(r) not in used_ids]
        taken_v = _take(pool, valid_q[topo], rng)
        for row in taken_v:
            used_ids.add(_id(row))
        valid.extend(taken_v)

        pool = [r for r in by_topo[topo] if _id(r) not in used_ids]
        taken_t = _take(pool, train_q[topo], rng)
        for row in taken_t:
            used_ids.add(_id(row))
        train.extend(taken_t)

    # Top up if some topologies were too thin.
    remaining = [r for r in rows if _id(r) not in used_ids]
    rng.shuffle(remaining)
    while len(valid) < n_valid and remaining:
        row = remaining.pop()
        used_ids.add(_id(row))
        valid.append(row)
    while len(train) < n_train and remaining:
        row = remaining.pop()
        used_ids.add(_id(row))
        train.append(row)

    if len(train) < n_train or len(valid) < n_valid:
        raise SystemExit(
            f"not enough rows: got train={len(train)}/{n_train} valid={len(valid)}/{n_valid} "
            f"from {len(rows)} source rows"
        )

    rng.shuffle(train)
    rng.shuffle(valid)
    stats = {
        "seed": seed,
        "n_train": len(train),
        "n_valid": len(valid),
        "train_topology": dict(Counter(_topology(r) for r in train)),
        "valid_topology": dict(Counter(_topology(r) for r in valid)),
        "disjoint": not ({_id(r) for r in train} & {_id(r) for r in valid}),
    }
    return train[:n_train], valid[:n_valid], stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=REPO / "data/mechet_sft/valid.jsonl",
        help="Source MECH_ET JSONL (prefer valid.jsonl for a locked smoke set)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data/mechet_sft/overfit32",
    )
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--n-valid", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11, help="Match configs/overfit32.yaml seed")
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(
            f"missing source {args.src}\n"
            "Build SFT first: python scripts/build_mechet_sft.py --out-dir data/mechet_sft\n"
            "Or symlink: ln -s /path/to/orbit_mech_et_sft data/mechet_sft"
        )

    rows = _load_jsonl(args.src)
    train, valid, stats = select_overfit(
        rows, n_train=args.n_train, n_valid=args.n_valid, seed=args.seed
    )
    _write_jsonl(args.out_dir / "train.jsonl", train)
    _write_jsonl(args.out_dir / "valid.jsonl", valid)
    manifest = {
        "version": "mechet_overfit32_v1",
        "purpose": "topology-balanced smoke / overfit slice for configs/overfit32.yaml",
        "source": str(args.src),
        "out_dir": str(args.out_dir),
        **stats,
        "train_ids": [r.get("id") for r in train],
        "valid_ids": [r.get("id") for r in valid],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("n_train", "n_valid", "train_topology", "valid_topology", "disjoint", "out_dir")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
