from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import DFlashDraftModel
from .spec_decode import (
    SpecDecodeResult,
    apply_chat_template_if_requested,
    build_dflash_sequence_tree_processor,
    evaluate_prompt_suite,
    infer_target_model_for_draft_model,
    render_eval_suite_html,
    resolve_official_dflash_model,
    speculative_generate,
)
from .trainer import unwrap_model


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def should_auto_apply_chat_template(*, draft_model: str) -> bool:
    return infer_target_model_for_draft_model(draft_model) is not None


def prepare_dflash_prompt(
    *,
    tokenizer,
    prompt: str,
    draft_model: str,
    apply_chat_template: bool | None,
) -> str:
    """Prepare a prompt for local DFlash speculative decoding."""
    enabled = apply_chat_template
    if enabled is None:
        enabled = should_auto_apply_chat_template(draft_model=draft_model)
    return apply_chat_template_if_requested(
        tokenizer=tokenizer,
        prompt=prompt,
        enabled=enabled,
    )


def build_tree_flash_generate_fn(
    *,
    target_model,
    drafter_model,
    tokenizer,
    draft_model_name: str,
    temperature: float,
    apply_chat_template: bool | None,
):
    """Build a prompt-based generation callback for local DFlash evaluation."""
    raw_drafter = unwrap_model(drafter_model)
    tree_processor = build_dflash_sequence_tree_processor(block_size=int(raw_drafter.block_size))

    def generate_fn(*, prompt: str, max_new_tokens: int) -> SpecDecodeResult:
        prepared_prompt = prepare_dflash_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            draft_model=draft_model_name,
            apply_chat_template=apply_chat_template,
        )
        return speculative_generate(
            target_model=target_model,
            drafter_model=drafter_model,
            tokenizer=tokenizer,
            prompt=prepared_prompt,
            tree_processor=tree_processor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return generate_fn


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for DFlash prompt and eval-dataset modes."""
    parser = argparse.ArgumentParser(
        description="Evaluate DFlash checkpoints with local sequence-tree speculative decoding."
    )
    parser.add_argument("--official-dflash-model", default=None)
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--prompt")
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument("--html-report", default=None)
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser


def _resolve_model_pair(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, str]:
    draft_model = args.draft_model
    if args.official_dflash_model is not None:
        draft_model = resolve_official_dflash_model(args.official_dflash_model)
    if draft_model is None:
        parser.error("a DFlash checkpoint is required via --draft-model or --official-dflash-model")
    target_model = args.target_model or infer_target_model_for_draft_model(draft_model)
    if target_model is None:
        parser.error("a target model is required when it cannot be inferred from the DFlash checkpoint")
    return target_model, draft_model


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
    """Run prompt decoding or eval-dataset generation for DFlash checkpoints."""
    parser = build_parser()
    args = parser.parse_args()
    if args.prompt is None and args.eval_data is None:
        parser.error("one of --prompt or --eval-data is required")

    target_model_name, draft_model_name = _resolve_model_pair(args, parser)
    dtype = _dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=dtype,
        attn_implementation="flex_attention",
    ).to(device)
    target_model.eval()

    drafter_model = DFlashDraftModel.from_pretrained(
        draft_model_name,
        torch_dtype=dtype,
    ).to(device)
    drafter_model.eval()

    generate_fn = build_tree_flash_generate_fn(
        target_model=target_model,
        drafter_model=drafter_model,
        tokenizer=tokenizer,
        draft_model_name=draft_model_name,
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
