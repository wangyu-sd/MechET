"""Pure-stdlib regression tests for PR83 resumable product-start evaluation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "pr83_resume.py"
spec = importlib.util.spec_from_file_location("pr83_resume_test_module", MODULE)
resume = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = resume
spec.loader.exec_module(resume)


def manifest(ids=("a", "b")):
    return resume.evaluation_manifest(
        selected_ids=ids,
        source_sha256="source", source_path="/original/valid.jsonl",
        qwen_adapter_sha256="adapter", graph_sha256="graph",
        graph_rl_head_sha256=None, model="Qwen/Qwen3-0.6B",
        model_revision="revision",
        config={"graph_flow_topk": 1, "max_decisions": 40},
        code_sha256={"scripts/run_crn_qwen_hybrid.py": "code"},
    )


def entry(case):
    return {
        "id": case, "policy": "crn_jepa_qwen_routed_flow_v1",
        "graph_checkpoint_sha256": "graph", "qwen_adapter_sha256": "adapter",
        "top1_structural_exact": False, "wall_seconds": 1.0,
        "hybrid_timing": {}, "hybrid_counts": {},
    }


class ResumeTests(unittest.TestCase):
    def test_fresh_resume_and_completed_id_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run"
            frozen = manifest()
            self.assertEqual(
                resume.prepare_resume(target, manifest=frozen, selected_ids=("a", "b")),
                [],
            )
            with (target / resume.RESULTS_NAME).open("w") as fh:
                fh.write(json.dumps(entry("a")) + "\n")
            cases = resume.prepare_resume(
                target, manifest=frozen, selected_ids=("a", "b"), resume=True,
            )
            self.assertEqual([r["id"] for r in cases], ["a"])
            self.assertRaises(
                FileExistsError, resume.prepare_resume, target,
                manifest=frozen, selected_ids=("a", "b"),
            )

    def test_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run"
            frozen = manifest()
            resume.prepare_resume(target, manifest=frozen, selected_ids=("a", "b"))
            changed = manifest()
            changed["graph_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "changed"):
                resume.prepare_resume(
                    target, manifest=changed,
                    selected_ids=("a", "b"), resume=True,
                )

    def test_truncated_tail_is_quarantined_not_miscounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run"
            frozen = manifest()
            resume.prepare_resume(target, manifest=frozen, selected_ids=("a", "b"))
            raw = (json.dumps(entry("a")) + '\n{"id": "b"').encode("utf-8")
            (target / resume.RESULTS_NAME).write_bytes(raw)
            rows = resume.prepare_resume(
                target, manifest=frozen, selected_ids=("a", "b"), resume=True,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                (target / resume.RESULTS_NAME).read_bytes(),
                (json.dumps(entry("a")) + "\n").encode("utf-8"),
            )
            tails = list(target.glob("results.partial-tail-*.bin"))
            self.assertEqual(len(tails), 1)
            self.assertEqual(tails[0].read_bytes(), b'{"id": "b"')

    def test_complete_line_corruption_cannot_be_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run"
            frozen = manifest()
            resume.prepare_resume(target, manifest=frozen, selected_ids=("a", "b"))
            (target / resume.RESULTS_NAME).write_text(
                json.dumps(entry("b")) + "\n",
            )
            with self.assertRaisesRegex(ValueError, "ID mismatch"):
                resume.prepare_resume(
                    target, manifest=frozen,
                    selected_ids=("a", "b"), resume=True,
                )


class TrainingResumeTests(unittest.TestCase):
    def test_training_state_requires_same_manifest_and_checkpoint(self):
        frozen = {"source_sha256": "source", "settings": {"slate": 8}}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "rl"
            self.assertIsNone(resume.prepare_training_resume(
                target, manifest=frozen,
            ))
            self.assertRaises(
                FileExistsError, resume.prepare_training_resume,
                target, manifest=frozen,
            )
            with self.assertRaisesRegex(ValueError, "checkpoint missing"):
                resume.prepare_training_resume(
                    target, manifest=frozen, resume=True,
                )
            saved = target / "latest_train_state.pt"
            saved.write_bytes(b"fixture")
            self.assertEqual(
                resume.prepare_training_resume(
                    target, manifest=frozen, resume=True,
                ),
                saved,
            )
            other = {"source_sha256": "source", "settings": {"slate": 16}}
            with self.assertRaisesRegex(ValueError, "changed"):
                resume.prepare_training_resume(
                    target, manifest=other, resume=True,
                )


if __name__ == "__main__":
    unittest.main()
