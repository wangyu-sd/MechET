"""Evidence types: a symbolic transition is not an experimental observation."""


def validate_evidence(record, *, complete_reaction):
    tier = record.get("tier", "C")
    if tier not in {"A", "B", "C", "D"}:
        raise ValueError("unknown chemistry evidence tier")
    if tier in {"A", "B", "D"} and not complete_reaction:
        raise ValueError("literature evidence applies to a complete reaction, not a partial FLOW")
    if tier == "D" and not record.get("contradiction_evidence"):
        raise ValueError("Tier D needs documented contradiction, not missing retrieval")
    verified = record.get("evidence_status") == "verified"
    if verified and tier in {"A", "B"} and not record.get("source_url"):
        raise ValueError("verified literature requires a source")
    if verified and tier == "A" and not all(record.get(k) is True for k in ("product_match", "precursor_match", "stereo_match")):
        raise ValueError("Tier A needs exact product, precursor and stereo checks")
    return {**record, "tier": tier, "verified_support": verified and tier in {"A", "B"},
        "physical_mechanism_verified": False}


def exposure_compatible(claim, *, actor_exposure, verified_program_absent=False, minimum_primitive_frequency=None, reference_fully_annotated=False):
    if claim == "H2":
        return actor_exposure == "H2_train_only"
    if claim == "C1":
        # Full-data actors may have genuine route C1 cases, but only relative
        # to their *entire* verified exposure, not historical H2 labels.
        return (actor_exposure in {"H2_train_only", "FlowER_full"}
            and verified_program_absent is True and reference_fully_annotated is True
            and minimum_primitive_frequency is not None and minimum_primitive_frequency >= 5)
    return False
