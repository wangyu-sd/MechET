"""Keep actual model inputs and raw outputs without changing generation."""
from __future__ import annotations


def attach_generation_capture(model, tokenizer) -> list[dict]:
    records: list[dict] = []
    original = model.generate

    def generate(*args, **kwargs):
        inputs = kwargs.get("input_ids")
        if inputs is None and args:
            inputs = args[0]
        input_rows = inputs.detach().cpu().tolist()
        width = inputs.shape[-1]
        current = [{"input_token_ids": ids,
                    "input_text": tokenizer.decode(ids, skip_special_tokens=False),
                    "max_new_tokens": kwargs.get("max_new_tokens")}
                   for ids in input_rows]
        try:
            result = original(*args, **kwargs)
            sequences = getattr(result, "sequences", result)
            for record, sequence in zip(current, sequences.detach().cpu().tolist(), strict=True):
                ids = sequence[width:]
                record.update(output_token_ids=ids,
                              output_text=tokenizer.decode(ids, skip_special_tokens=False))
            return result
        except Exception as exc:
            for record in current:
                record["error"] = f"{type(exc).__name__}:{exc}"
            raise
        finally:
            records.extend(current)

    model.generate = generate
    return records
