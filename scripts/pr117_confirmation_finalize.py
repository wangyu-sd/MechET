"""Seal an independently replayed Reference run or a read-only partial snapshot."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def replay_status(audit):
    rows = audit["rows"]
    full = audit["fully_audited"] and audit["final_summary_verified"]
    audited = sum(row["gpu_requests"] for row in rows)
    issued = audit["journal"].get("gpu_requests", 0)
    returned = audit["journal"].get("gpu_returns", 0)
    if not audited <= returned <= issued:
        raise ValueError("Completed replays exceed the observed physical journal")
    if full and (len(rows) != audit["expected"] or not audited == returned == issued):
        raise ValueError("Completed replay does not cover the declared physical work")
    closures = Counter(row["arm"] for row in rows if row["solved"])
    return dict(status="COMPLETE_REPLAY_VERIFIED" if full else "PARTIAL_RECEIPTS_VERIFIED",
                expected_arms=audit["expected"], completed_arms=len(rows),
                completed_arms_by_arm=dict(Counter(row["arm"] for row in rows)),
                fully_paired_targets=audit["fully_paired_targets"], audited_native_calls=audited,
                issued_native_calls=issued, returned_native_calls=returned,
                audited_closures_by_arm=dict(closures))


def finalize(args):
    launch = args.launch.resolve()
    if not launch.is_relative_to("/aaa/fionafyang/buddy1"):
        raise ValueError("Reference evidence belongs on shared storage")
    results = launch / "results"
    staging = Path(tempfile.mkdtemp(prefix=".reference-analysis-", dir=launch))
    audit_path = staging / "receipt_audit.json"
    registration = []
    if args.confirmation_registration is not None:
        registration = ["--confirmation-registration", args.confirmation_registration,
                        "--confirmation-registration-sha256", args.confirmation_registration_sha256]
    subprocess.run([sys.executable, "-S", args.analysis_code / "audit_reference_receipts.py",
                    "--runtime-contract", args.runtime_contract, "--panel", args.panel,
                    "--results", results, "--write", audit_path, *registration], check=True)
    audit = json.loads(audit_path.read_bytes())
    status = replay_status(audit)

    if status["status"] == "COMPLETE_REPLAY_VERIFIED":
        summary = results / "screen_result.json"
        subprocess.run([sys.executable, "-S", args.analysis_code / "analyze_paired_reference.py",
                        args.panel, summary, audit_path, staging / "paired_results.json",
                        staging / "target_arm.csv"], check=True)
        subprocess.run([sys.executable, "-S", "-m", "scripts.pr117_matched_work", "--panel", args.panel,
                        "--audit", audit_path, "--results", results, "--output",
                        staging / "matched_physical_work.json"], check=True)

    status["binding"] = audit["binding"]
    status["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir()}
    now = datetime.now().astimezone()
    status["recorded_at"] = now.isoformat(timespec="seconds")
    with (staging / "evidence.json").open("x") as f:
        f.write(json.dumps(status, indent=2) + "\n")
    output = launch / f"evidence-{now:%Y%m%dT%H%M%S}-{staging.name.rsplit('-', 1)[-1]}"
    staging.rename(output)
    print(json.dumps(dict(evidence_path=str(output), **status)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", type=Path)
    parser.add_argument("--analysis-code", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--confirmation-registration", type=Path)
    parser.add_argument("--confirmation-registration-sha256")
    args = parser.parse_args()
    if bool(args.confirmation_registration) != bool(args.confirmation_registration_sha256):
        parser.error("confirmation registration and hash must be supplied together")
    finalize(args)


if __name__ == "__main__":
    main()
