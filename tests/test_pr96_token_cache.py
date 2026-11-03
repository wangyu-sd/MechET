import json
import os

import pytest


def test_qwen35_cache_refuses_to_drop_current_state_even_with_lossless_label_windows(tmp_path):
    snapshot = os.environ.get("MECHET_QWEN35_SNAPSHOT")
    if not snapshot:
        pytest.skip("pinned official tokenizer required")
    from transformers import AutoTokenizer
    from scripts.prepare_tool_sft_arrow import tokenize_shard
    from test_pr96_history import state_rows
    rows = state_rows()
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    with pytest.raises(ValueError, match="QWEN35_STATE_TRUNCATION_FORBIDDEN"):
        tokenize_shard(source=source, target=tmp_path / "train.arrow", tokenizer=tokenizer,
                       rank=0, world_size=1, max_length=32, context_tokens=16,
                       require_trace_owned=False, require_tool_decision=True,
                       allow_fallback=False, encoding_protocol="qwen35_tool_span_v1")
