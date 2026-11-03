from copy import deepcopy
import pytest

from mechet.qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION
from test_pr96_validation_protocol import passing_metrics


def test_preflight_requires_eight_real_rank_records_of_the_requested_type():
    from mechet.pr96_artifacts import validate_preflight
    checks = {k: True for k in ("finite_nonzero_gradients", "parameters_updated", "assistant_mask_verified",
                               "tool_prefix_parity", "all_language_targets_present", "language_only", "generation_completed",
                               "ddp_gradient_synchronization")}
    value = {"track": "mech31k", "base_revision": OFFICIAL_REVISION,
             "ranks": [{"rank": i, "gpu_names": ["A100"]*8, **checks} for i in range(8)]}
    validate_preflight(value, "mech31k")
    for mutation in (lambda x: x["ranks"].pop(), lambda x: x["ranks"][0].update(parameters_updated=False),
                     lambda x: x["ranks"][0].update(ddp_gradient_synchronization=False),
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


def checkpoint_fixture(tmp_path):
    import json
    from mechet.pr96_eval_runtime import sha
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    files = {}
    for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json"):
        (checkpoint / name).write_text(json.dumps({"global_step": 12, "epoch": 1}) if name == "trainer_state.json" else name)
        files[name] = sha(checkpoint / name)
    rng, counters = {}, {}
    for rank in range(8):
        name = f"rng_state_{rank}.pth"
        (checkpoint / name).write_text(name)
        rng[name] = sha(checkpoint / name)
        name = f"pr96_counters.rank{rank:02d}.json"
        (checkpoint / name).write_text(json.dumps({"counts": [12, 123, 456]}))
        counters[name] = sha(checkpoint / name)
    cfg = {"model_name_or_path": OFFICIAL_MODEL, "training": {"model_revision": OFFICIAL_REVISION},
           "lora": {"r": 16}, "contract": {}, "pr96": {"track": "mech31k", "stage": "state",
           "reference_approval_sha256": "source", "cohorts_sha256": "cohort",
           "token_cache_manifest_sha256": "tokens", "base_manifest_sha256": "base"}}
    from mechet.pr96_artifacts import training_identity
    manifest = {"track": "mech31k", "stage": "state", "step": 12, "epoch": 1,
                "lineage": {"training_identity": training_identity(cfg)}, "files": files,
                "rng_files": rng, "counter_files": counters}
    (checkpoint / "pr96_checkpoint_manifest.json").write_text(json.dumps(manifest))
    return cfg, checkpoint


def test_resume_checks_scientific_identity_and_all_actual_optimizer_rng_counter_hashes(tmp_path):
    from mechet.pr96_artifacts import verify_resume_checkpoint
    cfg, checkpoint = checkpoint_fixture(tmp_path)
    verify_resume_checkpoint(cfg, checkpoint)
    for section, key, value in (("pr96", "track", "flower"), ("training", "learning_rate", .5), ("lora", "r", 32)):
        changed = deepcopy(cfg)
        changed[section][key] = value
        with pytest.raises(ValueError, match="RESUME"):
            verify_resume_checkpoint(changed, checkpoint)
    for name in ("optimizer.pt", "rng_state_7.pth", "pr96_counters.rank03.json"):
        path = checkpoint / name
        original = path.read_bytes()
        path.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="RESUME"):
            verify_resume_checkpoint(cfg, checkpoint)
        path.write_bytes(original)


def test_evaluation_binds_actual_valid_sources_and_rejects_same_id_mutation(tmp_path):
    import json
    from mechet.pr96_artifacts import verify_evaluation_bindings
    from mechet.pr96_eval_runtime import sha
    cfg, checkpoint = checkpoint_fixture(tmp_path)
    reference = tmp_path / "reference"
    records = {}
    for kind in ("inverse", "state_sft"):
        (reference / kind).mkdir(parents=True)
        path = reference / kind / "valid.jsonl"
        path.write_text('{"source_id":"one","target_smiles":"CO"}\n')
        records[kind] = {"path": str(path), "sha256": sha(path)}
    manifest = reference / "manifest.json"
    manifest.write_text(json.dumps({"reports": {"valid": {"outputs": records}}}))
    cohort = tmp_path / "cohorts.json"
    cohort.write_text('{"paired_ids":["one"],"independent_ids":["two"]}')
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"track": "mech31k", "reference_training_authorized": True,
                                   "reference_manifest": str(manifest), "reference_manifest_sha256": sha(manifest),
                                   "frozen_cohorts_sha256": sha(cohort)}))
    cfg["pr96"].update(reference_dir=str(reference), reference_approval=str(approval),
                        reference_approval_sha256=sha(approval), cohorts_file=str(cohort), cohorts_sha256=sha(cohort))
    # Match the checkpoint's recorded scientific identity to this complete fixture.
    from mechet.pr96_artifacts import training_identity
    metadata = json.loads((checkpoint / "pr96_checkpoint_manifest.json").read_text())
    metadata["lineage"]["training_identity"] = training_identity(cfg)
    (checkpoint / "pr96_checkpoint_manifest.json").write_text(json.dumps(metadata))
    verify_evaluation_bindings(cfg, checkpoint)
    (reference / "inverse/valid.jsonl").write_text('{"source_id":"one","target_smiles":"CC"}\n')
    with pytest.raises(ValueError, match="EVALUATION"):
        verify_evaluation_bindings(cfg, checkpoint)
