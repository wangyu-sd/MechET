"""Verify and copy an interrupted reference build without trusting partial tails.

This is only storage recovery. The later full independent history/review gates
remain mandatory; a recovered prefix is not a new authorization or chemistry.
"""
from collections import Counter
from contextlib import ExitStack
import ast
import hashlib
import json
from pathlib import Path


def encoded(value):
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def result_payload(ok, value, split):
    if not ok:
        value = dict(value, split=split)
        reason = value["error"].split(":", 1)[-1].split(":", 1)[0]
        return {"unresolved": encoded(value)}, Counter({"error:" + reason: 1})
    counts = Counter(d["metadata"]["decision_type"] + "_decisions" for d in value["decisions"])
    counts["first_" + value["audit"]["first_tool"]] += 1
    return {"inverse": encoded(value["inverse"]), "audits": encoded(value["audit"]),
            "state_sft": b"".join(encoded(d) for d in value["decisions"])}, counts


def recover_prefix(reader, old_paths, progress, writer, emit):
    from scripts.finalize_pr96_reference import verify_source_identity
    accepted, unresolved, counts = set(), set(), Counter()
    bad_remaining = progress["unresolved"]
    with ExitStack() as stack:
        old = {k: stack.enter_context(Path(p).open("rb")) for k,p in old_paths.items()}
        def row(kind):
            line = old[kind].readline()
            if not line.endswith(b"\n"):
                raise ValueError("COMMITTED_PREFIX_JOURNAL_INCOMPLETE")
            return json.loads(line)
        bad = row("unresolved") if bad_remaining else None
        pending = {k: bytearray() for k in old_paths}
        for index in range(1, progress["processed"] + 1):
            source = json.loads(next(reader))
            rid = str(source["source_id"])
            if rid in accepted or rid in unresolved:
                raise ValueError("DUPLICATE_SOURCE_ID_IN_RECOVERY")
            digest = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if bad is not None and rid == str(bad["source_id"]):
                if bad["source_row"] != source or bad["source_row_sha256"] != digest or bad.get("training_allowed") is not False:
                    raise ValueError("UNRESOLVED_SOURCE_ID_OR_HASH_CHANGED")
                payload, increment = result_payload(False, bad, bad["split"])
                unresolved.add(rid)
                bad_remaining -= 1
                bad = row("unresolved") if bad_remaining else None
            else:
                inverse, audit = row("inverse"), row("audits")
                verify_source_identity(source, inverse)
                if (str(audit["source_id"]) != rid or audit["source_row_sha256"] != digest
                        or not all(audit.get(k) is True for k in
                            ("reference_endpoint_exact", "original_moves_preserved", "model_prefix_parity"))):
                    raise ValueError("COMMITTED_PREFIX_AUDIT_CHANGED")
                decisions = [row("state_sft") for _ in range(audit["decision_count"])]
                for dindex, decision in enumerate(decisions):
                    meta = decision["metadata"]
                    if (str(decision["source_id"]) != rid or meta["decision_index"] != dindex
                            or meta.get("source_electron_moves_preserved") is not True
                            or meta.get("executor_replayed") is not True):
                        raise ValueError("COMMITTED_PREFIX_DECISION_CHANGED")
                if (not decisions or decisions[-1]["metadata"]["decision_type"] != "finish"
                        or decisions[0]["messages"][2]["tool_calls"][0]["function"]["name"] != audit["first_tool"]
                        or sum(d["metadata"]["decision_type"] == "event" for d in decisions) != len(source["metadata"]["trace_plan"]["steps"])):
                    raise ValueError("COMMITTED_PREFIX_REACTION_INCOMPLETE")
                payload, increment = result_payload(True, {"inverse": inverse, "audit": audit, "decisions": decisions}, "train")
                accepted.add(rid)
            counts.update(increment)
            for k, value in payload.items():
                pending[k].extend(value)
            if index % 50 == 0 or index == progress["processed"]:
                offsets = writer.commit({k: bytes(v) for k,v in pending.items()})
                pending = {k: bytearray() for k in old_paths}
                emit({"stage": "verified-prefix-copy", "processed": index, "total": progress["processed"],
                      "percent": 100 * index / progress["processed"], "committed_offsets": offsets})
    if (bad_remaining or len(accepted) != progress["reference_pass"] or len(unresolved) != progress["unresolved"]
            or counts != progress["counts"]):
        raise ValueError("COMMITTED_PREFIX_COUNTERS_CHANGED")
    return accepted, unresolved, counts


def verify_recovery_code(old_dir, code_paths, root):
    """Only serialization/main-loop changes may differ from the old builder."""
    for current in code_paths:
        relative = current.relative_to(root)
        saved = old_dir / "code_snapshot" / relative
        if relative.as_posix() != "scripts/build_main_product_training_data.py":
            if saved.read_bytes() != current.read_bytes():
                raise ValueError(f"RECOVERY_CHEMICAL_DEPENDENCY_CHANGED:{relative}")
        else:
            def chemistry(path):
                tree = ast.parse(path.read_text())
                tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
                             and not (isinstance(n, ast.FunctionDef) and n.name == "main")
                             and not isinstance(n, ast.If)]
                return ast.dump(tree, include_attributes=False)
            if chemistry(saved) != chemistry(current):
                raise ValueError("RECOVERY_REFERENCE_ALGORITHM_CHANGED")


def copy_completed_split(old_dir, split, source_sha, expected, paths, bad_path, emit):
    from mechet.pr96_durable_io import DurableAppendSet
    from scripts.build_mech31k_main_product_only_sft import sha
    report_path = old_dir / f"{split}.report.json"
    report = json.loads(report_path.read_text())
    if report["source_sha256"] != source_sha or report["original_count"] != expected:
        raise ValueError("RECOVERY_COMPLETED_SOURCE_HASH_OR_DENOMINATOR_CHANGED")
    sources = {k: Path(v["path"]) for k,v in report["outputs"].items()}
    sources["unresolved"] = Path(report["unresolved_path"])
    hashes = {k: v["sha256"] for k,v in report["outputs"].items()}
    hashes["unresolved"] = report["unresolved_sha256"]
    if any(not path.resolve().is_relative_to(old_dir.resolve()) for path in sources.values()):
        raise ValueError("RECOVERY_COMPLETED_PATH_OUTSIDE_SOURCE")
    for kind, path in sources.items():
        if sha(path) != hashes[kind]:
            raise ValueError("RECOVERY_COMPLETED_JOURNAL_HASH_CHANGED")
    with ExitStack() as stack:
        readers = {k: stack.enter_context(p.open("rb")) for k,p in sources.items()}
        writer = stack.enter_context(DurableAppendSet(dict(paths, unresolved=bad_path), on_retry=emit))
        processed = 0
        total = sum(p.stat().st_size for p in sources.values())
        while True:
            payload = {k: r.read(1024 * 1024) for k,r in readers.items()}
            if not any(payload.values()):
                break
            writer.commit(payload)
            processed += sum(len(b) for b in payload.values())
            emit({"stage": "verified-completed-split-copy", "split": split,
                  "processed_bytes": processed, "total_bytes": total,
                  "percent": 100 * processed / total})
    for kind, path in dict(paths, unresolved=bad_path).items():
        if sha(path) != hashes[kind]:
            raise ValueError("RECOVERY_COMPLETED_COPY_HASH_CHANGED")
    report["outputs"] = {k: {"path": str(p), "sha256": hashes[k]} for k,p in paths.items()}
    report.update(unresolved_path=str(bad_path), unresolved_sha256=hashes["unresolved"])
    report["reused_report"] = {"path": str(report_path), "sha256": sha(report_path)}
    return report
