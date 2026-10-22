from copy import deepcopy
import pytest

from mechet.qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION
from test_pr96_validation_protocol import passing_metrics


def test_preflight_requires_eight_real_rank_records_of_the_requested_type():
    from mechet.pr96_artifacts import validate_preflight
    checks = {k: True for k in ("finite_nonzero_gradients", "parameters_updated", "assistant_mask_verified",
                               "tool_prefix_parity", "all_language_targets_present", "language_only", "generation_completed")}
    value = {"track": "mech31k", "base_revision": OFFICIAL_REVISION,
             "ranks": [{"rank": i, "gpu_names": ["A100"]*8, **checks} for i in range(8)]}
    validate_preflight(value, "mech31k")
    for mutation in (lambda x: x["ranks"].pop(), lambda x: x["ranks"][0].update(parameters_updated=False),
                     lambda x: x["ranks"][0].update(gpu_names=["H20"]*8)):
        bad = deepcopy(value); mutation(bad)
        with pytest.raises(ValueError): validate_preflight(bad, "mech31k")


def test_token_authorization_rejects_old_masks_and_different_source_hash():
    from mechet.pr96_artifacts import validate_tokens
    split = {"n_rows": 4, "truncation_count": 0, "n_windowed_source_rows": 0,
             "min_supervised_tokens": 3, "assistant_mask_methods": ["qwen35_owned_xml_tool_span_v1"]}
    cache = {"model_name_or_path": OFFICIAL_MODEL, "model_revision": OFFICIAL_REVISION,
             "encoding_protocol": "qwen35_tool_span_v1", "sources": {"train": {"sha256": "a"}, "validation": {"sha256": "b"}},
             "splits": {"train": deepcopy(split), "validation": deepcopy(split)}}
    validate_tokens(cache, {"train": ("a", 4), "validation": ("b", 4)})
    cache["splits"]["train"]["assistant_mask_methods"] = ["legacy"]
    with pytest.raises(ValueError): validate_tokens(cache, {"train": ("a", 4), "validation": ("b", 4)})


def test_parent_requires_actual_passing_gate_and_matching_own_track_checkpoint():
    from mechet.pr96_artifacts import validate_parent
    parent = {"track": "mech31k", "stage": "state", "lineage": {"base_revision": OFFICIAL_REVISION},
              "files": {"adapter_model.safetensors": "weight-sha"}}
    report = {"track": "mech31k", "stage": "state", "adapter_model_sha256": "weight-sha",
              "cohorts_sha256": "cohort-sha", "base_revision": OFFICIAL_REVISION, "metrics": passing_metrics()}
    validate_parent(parent, report, track="mech31k", weight_sha="weight-sha", cohort_sha="cohort-sha")
    report["metrics"]["structural_hits"] = 0
    report["gate"] = {"go": True}
    with pytest.raises(ValueError, match="GATE"):
        validate_parent(parent, report, track="mech31k", weight_sha="weight-sha", cohort_sha="cohort-sha")
