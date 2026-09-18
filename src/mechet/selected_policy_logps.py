"""Exact action-token log probabilities without allocating full context×vocab logits."""
from types import MethodType


def install_selected_logps_forward(model, chunk_size=128):
    """Install on a Qwen causal LM before PEFT wrapping; normal forward stays intact.

    The transformer processes the full history. Only policy-token hidden states
    are projected to vocabulary logits, in checkpointed chunks. This changes
    memory use, not attention, token likelihoods, or policy-gradient masks.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    original = model.forward

    def forward(self, input_ids=None, attention_mask=None, loss_token_mask=None, **kwargs):
        if loss_token_mask is None:
            return original(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if input_ids.shape[0] != 1:
            raise ValueError("Selected-logprob forward currently requires microbatch=1")
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, return_dict=True).last_hidden_state
        indices = torch.nonzero(loss_token_mask[0, 1:] > 0, as_tuple=False).flatten()
        # Keep an autograd path even for a fully censored trajectory, so every
        # DDP rank participates in the same backward collectives.
        result = hidden.sum(dim=-1)[:, :-1].float() * 0
        if indices.numel():
            chosen = hidden[0, indices]
            targets = input_ids[0, indices + 1]

            def project(x, target):
                logits = self.lm_head(x).float()
                return logits.gather(-1, target[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)

            chunks = []
            for offset in range(0, len(indices), chunk_size):
                x, y = chosen[offset:offset + chunk_size], targets[offset:offset + chunk_size]
                chunks.append(checkpoint(project, x, y, use_reentrant=False) if torch.is_grad_enabled()
                              else project(x, y))
            result = result.scatter(1, indices[None], torch.cat(chunks)[None])
        return {"selected_logps": result}

    model.forward = MethodType(forward, model)
