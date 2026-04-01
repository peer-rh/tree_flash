from __future__ import annotations

import math

import torch


def unwrap_model(module):
    """Peel common wrapper modules such as Fabric, DDP, and torch.compile.

    Args:
        module: Wrapped or unwrapped model object.

    Returns:
        The innermost model-like object. The helper recursively follows
        ``_forward_module``, ``module``, and ``_orig_mod`` when present.
    """
    raw = module
    seen_ids: set[int] = set()
    while True:
        raw_id = id(raw)
        if raw_id in seen_ids:
            break
        seen_ids.add(raw_id)

        next_raw = None
        for attr in ("_forward_module", "module", "_orig_mod"):
            candidate = getattr(raw, attr, None)
            if candidate is not None:
                next_raw = candidate
                break
        if next_raw is None:
            break
        raw = next_raw
    return raw


def cosine_lr(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Warmup plus cosine decay learning-rate schedule."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample token ids from logits or take the argmax when temperature <= 0."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def gather_token_probability(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return the model probability assigned to ``token_ids``.

    Args:
        logits: Tensor of shape ``(..., vocab_size)``.
        token_ids: Tensor of shape ``(...)``.
        temperature: Sampling temperature used to interpret the logits.

    Returns:
        Tensor of shape ``(...)`` with probabilities in ``float32``.
    """
    if temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).to(torch.float32)
