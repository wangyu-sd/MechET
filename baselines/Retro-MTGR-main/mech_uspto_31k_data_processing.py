#!/usr/bin/env python3
"""Prepare the frozen mech-USPTO-31K splits for Retro-MTGR."""
from pathlib import Path

from retro_mtgr_data_processing import cli


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(cli(
        dataset="mech_uspto_31k_full",
        default_input_dir=ROOT / "data/MechET/mech_uspto_31k_full",
        default_output_dir=ROOT / "data/MechET/mech_uspto_31k_full_retro_mtgr_processed",
    ))
