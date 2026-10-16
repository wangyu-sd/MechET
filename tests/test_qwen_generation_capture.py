import torch
from mechet.qwen_generation_capture import attach_generation_capture


def test_capture_preserves_generation_and_records_exact_input_output():
    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return ",".join(map(str, tokens))
    class Model:
        def generate(self, **kwargs):
            self.result = type("Generation", (), {
                "sequences": torch.tensor([[10, 20, 30, 40]])})()
            return self.result
    model = Model()
    records = attach_generation_capture(model, Tokenizer())
    result = model.generate(input_ids=torch.tensor([[10, 20]]), max_new_tokens=384)
    assert result is model.result
    assert records[0]["input_token_ids"] == [10, 20]
    assert records[0]["output_token_ids"] == [30, 40]
    assert records[0]["input_text"] == "10,20"
    assert records[0]["output_text"] == "30,40"
    records.clear()
    model.generate(input_ids=torch.tensor([[10, 20]]))
    assert len(records) == 1


def test_capture_keeps_failed_generation_input():
    import pytest
    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return "input"
    class Model:
        def generate(self, **kwargs):
            raise ValueError("bad generation")
    model = Model()
    records = attach_generation_capture(model, Tokenizer())
    with pytest.raises(ValueError, match="bad generation"):
        model.generate(input_ids=torch.tensor([[1]]))
    assert records[0]["error"] == "ValueError:bad generation"
    assert records[0]["input_token_ids"] == [1]
