"""Real stalled FlowER metal state; keep exact stereo/H/isotope identity."""
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.parametrize("stereo_mismatch", [False, True])
def test_identical_high_symmetry_metal_states_do_not_enter_exponential_vf2(stereo_mismatch):
    path = Path(__file__).parent / "fixtures/pr96_high_symmetry_metal_state.json"
    command = [sys.executable, "-c", """
import json,sys
from scripts.build_mech31k_main_product_only_sft import same_chemical_state
row=json.load(open(sys.argv[1]))
if sys.argv[2]=='True':
    assert not same_chemical_state(row['left']+'.C[C@H](N)O',row['right']+'.C[C@@H](N)O')
else:
    assert same_chemical_state(row['left'],row['right'])
""", str(path), str(stereo_mismatch)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        raise AssertionError("exact same high-symmetry state still requires exponential matching")
    assert completed.returncode == 0, completed.stderr
