"""Frozen native Stage-II Qwen3.5 electron-tool actor; never decodes reactants directly."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION, load_language_model, tool_prefix
from scripts.build_natural_language_event_sft import TOOLS

STAGE2 = {
    "mech31k": {
        "path": "/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/pr96_mech31k_qwen35_08b_mainonly_20261022_01/models/history/checkpoint-660",
        "sha256": "cd5f2f9dbdc883674be2f87781452366a3fb2c01c7c930482e0b5e4b3dab49ca",
    },
    "flower": {
        "path": "/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/pr96_flower_qwen35_08b_mainonly_20261023_06/models/history/checkpoint-89442",
        "sha256": "2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453",
    },
}

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

class FrozenStageIIActor:
    """Every output is an IMPORT/FLOW/FINISH tool call handled by MechET."""
    def __init__(self, track, *, device="cuda:0", base=OFFICIAL_MODEL, log_path):
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer
        if track not in STAGE2:
            raise ValueError("UNKNOWN_STAGE_II_TRACK")
        binding = STAGE2[track]
        self.checkpoint = Path(binding["path"])
        actual = file_sha256(self.checkpoint / "adapter_model.safetensors")
        if actual != binding["sha256"]:
            raise ValueError("STAGE_II_ADAPTER_SHA256_MISMATCH")
        self.actor_sha, self.track = actual, track
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.torch = torch
        tokenizer = AutoTokenizer.from_pretrained(base, revision=OFFICIAL_REVISION,
                                                   trust_remote_code=False, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model, _ = load_language_model(base, dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base_model, str(self.checkpoint),
                                               is_trainable=False).to(device).eval()
        self.tokenizer = tokenizer

    def generate(self, messages, *, max_new_tokens, seed):
        import torch
        from transformers import GenerationConfig
        if max_new_tokens < 1:
            raise ValueError("GENERATION_BUDGET_EXHAUSTED")
        prompt = tool_prefix(self.tokenizer, messages, TOOLS)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        width = encoded["input_ids"].shape[1]
        if width + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("QWEN35_CONTEXT_OVERFLOW_NO_TRUNCATION")
        ids_before = encoded["input_ids"][0].tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cfg = GenerationConfig.from_model_config(self.model.config)
        cfg.do_sample, cfg.temperature, cfg.top_p, cfg.num_beams = True, 1.0, 1.0, 1
        # No sampling warp: native electron-tree NLL must use actual model probabilities.
        cfg.top_k, cfg.repetition_penalty = 0, 1.0
        cfg.pad_token_id = self.tokenizer.pad_token_id
        cfg.eos_token_id = self.tokenizer.eos_token_id
        with torch.inference_mode():
            samples = self.model.generate(**encoded, generation_config=cfg,
                                          max_new_tokens=max_new_tokens, use_cache=True,
                                          return_dict_in_generate=True, output_scores=True)
        logps = self.model.compute_transition_scores(
            samples.sequences, samples.scores, normalize_logits=True)[0]
        generated = samples.sequences[0, width:].tolist()
        ended = self.tokenizer.eos_token_id in generated
        if ended:
            generated = generated[:generated.index(self.tokenizer.eos_token_id) + 1]
        output = {"raw":self.tokenizer.decode(generated, skip_special_tokens=False),
                  "generated_token_ids":generated, "prompt_token_ids":ids_before,
                  "truncated":not ended, "logprob_available":True,
                  "logprob":float(logps[:len(generated)].sum().item()),
                  "track":self.track, "stage":"Stage-II-frozen", "actor_sha":self.actor_sha}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"messages":messages, "prompt":prompt,
                                     "seed":seed, "max_new_tokens":max_new_tokens,
                                     **output}, ensure_ascii=False) + "\n")
            handle.flush()
        return output
