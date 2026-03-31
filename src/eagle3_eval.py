from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .spec_decode import (
    SpecDecodeResult,
    apply_chat_template_if_requested,
    evaluate_prompt_suite,
    render_eval_suite_html,
)


OFFICIAL_EAGLE3_MODELS = {
    "vicuna-13b-v1.3": "yuhuili/EAGLE3-Vicuna1.3-13B",
    "lmsys/vicuna-13b-v1.3": "yuhuili/EAGLE3-Vicuna1.3-13B",
    "eagle3-vicuna1.3-13b": "yuhuili/EAGLE3-Vicuna1.3-13B",
    "llama-3.1-8b-instruct": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "meta-llama/llama-3.1-8b-instruct": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "eagle3-llama3.1-instruct-8b": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "llama-3.3-70b-instruct": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "meta-llama/llama-3.3-70b-instruct": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "eagle3-llama3.3-instruct-70b": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "deepseek-r1-distill-llama-8b": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    "deepseek-ai/deepseek-r1-distill-llama-8b": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    "eagle3-deepseek-r1-distill-llama-8b": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    "yuhuili/eagle3-vicuna1.3-13b": "yuhuili/EAGLE3-Vicuna1.3-13B",
    "yuhuili/eagle3-llama3.1-instruct-8b": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "yuhuili/eagle3-llama3.3-instruct-70b": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "yuhuili/eagle3-deepseek-r1-distill-llama-8b": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
}

OFFICIAL_EAGLE3_TARGETS = {
    "yuhuili/EAGLE3-Vicuna1.3-13B": "lmsys/vicuna-13b-v1.3",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B": "meta-llama/Llama-3.3-70B-Instruct",
    "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

OFFICIAL_EAGLE3_FASTCHAT_TEMPLATES = {
    "lmsys/vicuna-13b-v1.3": "vicuna",
}


def resolve_official_eagle3_model(model_name: str) -> str:
    """Resolve a short official EAGLE-3 alias to its canonical Hugging Face repo id."""
    resolved = OFFICIAL_EAGLE3_MODELS.get(model_name.lower())
    if resolved is not None:
        return resolved
    if model_name in OFFICIAL_EAGLE3_TARGETS:
        return model_name
    lowered = model_name.lower()
    if "qwen" in lowered and "eagle" in lowered:
        raise ValueError(
            "Qwen EAGLE-3 checkpoints are currently community checkpoints upstream, not official ones. "
            "Use --ea-model/--base-model directly instead of --official-eagle3-model."
        )
    raise ValueError(
        f"{model_name!r} is not a supported official EAGLE-3 checkpoint alias or repo id."
    )


def infer_target_model_for_eagle3_draft_model(draft_model: str) -> str | None:
    """Infer the base target model for a supported official EAGLE-3 checkpoint."""
    return OFFICIAL_EAGLE3_TARGETS.get(draft_model)


def _get_upstream_ea_model_class():
    from eagle.model.ea_model import EaModel

    return EaModel


def _get_fastchat_conversation_template():
    from fastchat.model import get_conversation_template

    return get_conversation_template


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def should_auto_apply_chat_template(target_model: str) -> bool:
    lowered = target_model.lower()
    return (
        target_model in OFFICIAL_EAGLE3_FASTCHAT_TEMPLATES
        or "instruct" in lowered
        or "deepseek-r1-distill-llama" in lowered
    )


def prepare_eagle3_prompt(
    *,
    tokenizer,
    prompt: str,
    target_model: str,
    apply_chat_template: bool,
) -> str:
    """Prepare a prompt for EAGLE-3 generation with optional or required chat formatting."""
    if apply_chat_template:
        return apply_chat_template_if_requested(
            tokenizer=tokenizer,
            prompt=prompt,
            enabled=True,
        )

    fastchat_template = OFFICIAL_EAGLE3_FASTCHAT_TEMPLATES.get(target_model)
    if fastchat_template is not None:
        try:
            get_conversation_template = _get_fastchat_conversation_template()
            conv = get_conversation_template(fastchat_template)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        except Exception:
            return prompt

    if should_auto_apply_chat_template(target_model) and hasattr(tokenizer, "apply_chat_template"):
        return apply_chat_template_if_requested(
            tokenizer=tokenizer,
            prompt=prompt,
            enabled=True,
        )
    return prompt


def load_upstream_eagle3_model(
    *,
    base_model: str,
    ea_model: str,
    dtype: torch.dtype,
    total_token: int,
    depth: int,
    top_k: int,
    threshold: float,
):
    """Lazily construct the upstream EAGLE-3 model wrapper."""
    try:
        EaModel = _get_upstream_ea_model_class()
    except Exception as exc:
        raise RuntimeError(
            "The upstream EAGLE package is not available. Install the optional EAGLE extra "
            "or install SafeAILab/EAGLE into the current environment."
        ) from exc

    model = EaModel.from_pretrained(
        base_model_path=base_model,
        ea_model_path=ea_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        threshold=threshold,
        use_eagle3=True,
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    return model, tokenizer


@torch.inference_mode()
def official_eagle3_generate_from_ids(
    *,
    eagle_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
) -> SpecDecodeResult:
    """Run generation through the official upstream EAGLE-3 backend."""
    model_device = next(eagle_model.base_model.parameters()).device
    prompt_ids = prompt_ids.to(model_device).view(1, -1)
    output_ids = eagle_model.eagenerate(
        prompt_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )[0]
    continuation_ids = output_ids[prompt_ids.shape[1] :].detach().cpu()
    return SpecDecodeResult(
        token_ids=output_ids.detach().cpu(),
        continuation_ids=continuation_ids,
        text=tokenizer.decode(output_ids, skip_special_tokens=True),
        acceptance_lengths=[],
        off_main_path_last_accept_flags=[],
        drafted_tokens=0,
        committed_tokens=int(continuation_ids.numel()),
    )


def _raise_flex_backend_unsupported(*, ea_model: str, base_model: str) -> None:
    raise ValueError(
        "The experimental flex backend does not currently support official EAGLE-3 checkpoints "
        f"for ea_model={ea_model!r} and base_model={base_model!r}. Use --backend upstream."
    )


def build_upstream_generate_fn(
    *,
    eagle_model,
    tokenizer,
    target_model: str,
    temperature: float,
    apply_chat_template: bool,
):
    """Build a prompt-based generation callback for evaluate_prompt_suite."""

    def generate_fn(*, prompt: str, max_new_tokens: int) -> SpecDecodeResult:
        prepared_prompt = prepare_eagle3_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            target_model=target_model,
            apply_chat_template=apply_chat_template,
        )
        prompt_ids = tokenizer(prepared_prompt, return_tensors="pt").input_ids[0]
        return official_eagle3_generate_from_ids(
            eagle_model=eagle_model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return generate_fn


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for EAGLE-3 prompt and eval-dataset modes."""
    parser = argparse.ArgumentParser(description="Evaluate official EAGLE-3 checkpoints.")
    parser.add_argument("--official-eagle3-model", default=None)
    parser.add_argument("--base-model")
    parser.add_argument("--ea-model")
    parser.add_argument("--backend", choices=["upstream", "flex"], default="upstream")
    parser.add_argument("--prompt")
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument("--html-report", default=None)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--total-token", type=int, default=-1)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="float16",
    )
    return parser


def _resolve_model_pair(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, str]:
    ea_model = args.ea_model
    if args.official_eagle3_model is not None:
        ea_model = resolve_official_eagle3_model(args.official_eagle3_model)
    if ea_model is None:
        parser.error("an EAGLE-3 checkpoint is required via --ea-model or --official-eagle3-model")
    base_model = args.base_model or infer_target_model_for_eagle3_draft_model(ea_model)
    if base_model is None:
        parser.error("a base model is required when it cannot be inferred from the EAGLE-3 checkpoint")
    return base_model, ea_model


def _report_summary(report) -> dict[str, int]:
    return {
        "total_examples": report.total_examples,
        "exact_matches": report.exact_matches,
        "accepted_count": report.accepted_count,
        "rejected_count": report.rejected_count,
        "extra_count": report.extra_count,
        "missing_count": report.missing_count,
    }


def main() -> None:
    """Run prompt decoding or eval-dataset generation for EAGLE-3 checkpoints."""
    parser = build_parser()
    args = parser.parse_args()
    if args.prompt is None and args.eval_data is None:
        parser.error("one of --prompt or --eval-data is required")

    base_model, ea_model = _resolve_model_pair(args, parser)
    dtype = _dtype_from_name(args.dtype)

    if args.backend == "flex":
        _raise_flex_backend_unsupported(ea_model=ea_model, base_model=base_model)

    eagle_model, tokenizer = load_upstream_eagle3_model(
        base_model=base_model,
        ea_model=ea_model,
        dtype=dtype,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    generate_fn = build_upstream_generate_fn(
        eagle_model=eagle_model,
        tokenizer=tokenizer,
        target_model=base_model,
        temperature=args.temperature,
        apply_chat_template=args.apply_chat_template,
    )

    if args.eval_data is not None:
        report = evaluate_prompt_suite(
            data_name=args.eval_data,
            tokenizer=tokenizer,
            generate_fn=generate_fn,
            max_examples=args.eval_max_examples,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(_report_summary(report), indent=2), flush=True)
        if args.html_report is not None:
            report_path = Path(args.html_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_eval_suite_html(report), encoding="utf-8")
        return

    result = generate_fn(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    print(result.text, flush=True)


if __name__ == "__main__":
    main()
