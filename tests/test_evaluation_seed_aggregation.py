import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "aggregate_evaluation_seeds.py"
spec = importlib.util.spec_from_file_location("mechet_seed_aggregation", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def runtime(seed: int) -> dict:
    return {
        "normal": {
            "runtime_contracts": [
                {
                    "seed": seed,
                }
            ]
        }
    }


def test_extracts_h1_seed_and_effects():
    value = {
        "scientific_hypothesis": "H1_causal_faithfulness",
        "runtime_contracts": runtime(17),
        "paired_effects": {
            "remove_tool_observations": {
                "structural_exact_delta_normal_minus_intervention": 0.2
            }
        },
    }
    assert module._seed_from_runtime(value) == "17"
    assert module._effects(value) == {"remove_tool_observations": 0.2}


def test_extracts_h3_effects_and_skips_missing_contrast():
    value = {
        "scientific_hypothesis": "H3_empirical_evidence_separation",
        "runtime_contracts": runtime(23),
        "paired_contrasts": {
            "textbook_minus_trace_only": {"delta_left_minus_right": 0.1},
            "missing": None,
        },
    }
    assert module._seed_from_runtime(value) == "23"
    assert module._effects(value) == {"textbook_minus_trace_only": 0.1}
