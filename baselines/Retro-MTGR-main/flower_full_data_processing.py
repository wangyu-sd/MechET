#!/usr/bin/env python3
"""Prepare the frozen Flower Full splits for Retro-MTGR."""
from pathlib import Path

from retro_mtgr_data_processing import cli


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(cli(
        dataset="flower_full",
        default_input_dir=ROOT / "data/MechET/flower_full",
        default_output_dir=ROOT / "data/MechET/flower_full_retro_mtgr_processed",
    ))
