def test_byte_shards_do_not_drop_a_line_when_boundary_is_exactly_a_newline(tmp_path):
    from scripts.prepare_tool_sft_arrow import iter_jsonl_byte_shard
    source = tmp_path / "rows.jsonl"
    source.write_text('{"a":1}\n{"a":2}\n')
    assert [l for rank in range(2) for l in iter_jsonl_byte_shard(source, rank, 2)] == ['{"a":1}\n', '{"a":2}\n']
