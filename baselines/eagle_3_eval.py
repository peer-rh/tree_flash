from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spec_decode import (
    SpecDecodeResult,
    apply_chat_template_if_requested,
    evaluate_prompt_suite,
    render_eval_suite_html,
)


STANDARD_EVAL_DATASETS = ("humaneval", "gsm8k", "alpaca")

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


def resolve_eagle_repo_path(eagle_repo: str | None) -> Path:
    raw = eagle_repo or os.environ.get("EAGLE_REPO")
    if not raw:
        raise ValueError("An upstream EAGLE checkout is required via --eagle-repo or EAGLE_REPO.")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"EAGLE repo path does not exist or is not a directory: {path}")
    return path


def _add_eagle_repo_to_sys_path(eagle_repo: Path) -> None:
    repo_str = str(eagle_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _get_upstream_ea_model_class(eagle_repo: Path):
    _add_eagle_repo_to_sys_path(eagle_repo)
    try:
        module = importlib.import_module("eagle.model.ea_model")
    except Exception as exc:
        raise RuntimeError(
            "The supplied EAGLE repo does not expose the official upstream module "
            "`eagle.model.ea_model`."
        ) from exc
    return module.EaModel


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


def resolve_official_eagle3_model(model_name: str) -> str:
    resolved = OFFICIAL_EAGLE3_MODELS.get(model_name.lower())
    if resolved is not None:
        return resolved
    if model_name in OFFICIAL_EAGLE3_TARGETS:
        return model_name
    lowered = model_name.lower()
    if "qwen" in lowered and "eagle" in lowered:
        raise ValueError(
            "Qwen EAGLE-3 checkpoints are not handled as official aliases here. "
            "Use --ea-model/--base-model directly."
        )
    raise ValueError(f"{model_name!r} is not a supported official EAGLE-3 checkpoint alias or repo id.")


def infer_target_model_for_eagle3_draft_model(draft_model: str) -> str | None:
    return OFFICIAL_EAGLE3_TARGETS.get(draft_model)


def resolve_model_pair(
    *,
    official_eagle3_model: str | None,
    ea_model: str | None,
    base_model: str | None,
) -> tuple[str, str]:
    if official_eagle3_model and ea_model:
        raise ValueError("Pass exactly one of --official-eagle3-model or --ea-model.")
    if official_eagle3_model:
        ea_model = resolve_official_eagle3_model(official_eagle3_model)
    if ea_model is None:
        raise ValueError("An EAGLE-3 checkpoint is required via --official-eagle3-model or --ea-model.")
    resolved_base_model = base_model or infer_target_model_for_eagle3_draft_model(ea_model)
    if resolved_base_model is None:
        raise ValueError("A base model is required when it cannot be inferred from the EAGLE-3 checkpoint.")
    return resolved_base_model, ea_model


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
    eagle_repo: Path,
    base_model: str,
    ea_model: str,
    dtype: torch.dtype,
    total_token: int,
    depth: int,
    top_k: int,
    threshold: float,
):
    EaModel = _get_upstream_ea_model_class(eagle_repo)
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


def build_upstream_generate_fn(
    *,
    eagle_model,
    tokenizer,
    target_model: str,
    temperature: float,
    apply_chat_template: bool,
):
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


def _report_summary(report) -> dict[str, int]:
    return {
        "total_examples": report.total_examples,
        "exact_matches": report.exact_matches,
        "accepted_count": report.accepted_count,
        "rejected_count": report.rejected_count,
        "extra_count": report.extra_count,
        "missing_count": report.missing_count,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_standard_eval_suite(
    *,
    eagle_repo: Path,
    output_dir: Path,
    base_model: str,
    ea_model: str,
    dtype: torch.dtype,
    temperature: float,
    apply_chat_template: bool,
    max_new_tokens: int,
    eval_max_examples: int | None,
    total_token: int,
    depth: int,
    top_k: int,
    threshold: float,
) -> dict[str, Any]:
    eagle_model, tokenizer = load_upstream_eagle3_model(
        eagle_repo=eagle_repo,
        base_model=base_model,
        ea_model=ea_model,
        dtype=dtype,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        threshold=threshold,
    )
    generate_fn = build_upstream_generate_fn(
        eagle_model=eagle_model,
        tokenizer=tokenizer,
        target_model=base_model,
        temperature=temperature,
        apply_chat_template=apply_chat_template,
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_summary: dict[str, Any] = {}
    for data_name in STANDARD_EVAL_DATASETS:
        dataset_dir = output_dir / data_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        report = evaluate_prompt_suite(
            data_name=data_name,
            tokenizer=tokenizer,
            generate_fn=generate_fn,
            max_examples=eval_max_examples,
            max_new_tokens=max_new_tokens,
        )
        summary = _report_summary(report)
        summary_path = dataset_dir / "summary.json"
        html_path = dataset_dir / "report.html"
        stdout_path = dataset_dir / "stdout.log"
        _write_json(summary_path, summary)
        html_path.write_text(render_eval_suite_html(report), encoding="utf-8")
        stdout_payload = {
            "dataset": data_name,
            "ea_model": ea_model,
            "base_model": base_model,
            **summary,
        }
        stdout_text = json.dumps(stdout_payload, indent=2) + "\n"
        stdout_path.write_text(stdout_text, encoding="utf-8")
        print(stdout_text, end="", flush=True)
        datasets_summary[data_name] = {
            **summary,
            "summary_path": str(summary_path),
            "report_path": str(html_path),
            "stdout_log_path": str(stdout_path),
        }

    top_level_summary = {
        "eagle_repo": str(eagle_repo),
        "ea_model": ea_model,
        "base_model": base_model,
        "datasets": list(STANDARD_EVAL_DATASETS),
        "config": {
            "dtype": str(dtype).replace("torch.", ""),
            "temperature": temperature,
            "apply_chat_template": apply_chat_template,
            "max_new_tokens": max_new_tokens,
            "eval_max_examples": eval_max_examples,
            "total_token": total_token,
            "depth": depth,
            "top_k": top_k,
            "threshold": threshold,
        },
        "results": datasets_summary,
    }
    _write_json(output_dir / "summary.json", top_level_summary)
    return top_level_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official upstream EAGLE-3 codebase over the repo eval suite."
    )
    parser.add_argument("--eagle-repo", default=None)
    parser.add_argument("--official-eagle3-model", default=None)
    parser.add_argument("--ea-model", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-max-examples", type=int, default=None)
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    eagle_repo = resolve_eagle_repo_path(args.eagle_repo)
    base_model, ea_model = resolve_model_pair(
        official_eagle3_model=args.official_eagle3_model,
        ea_model=args.ea_model,
        base_model=args.base_model,
    )
    run_standard_eval_suite(
        eagle_repo=eagle_repo,
        output_dir=Path(args.output_dir),
        base_model=base_model,
        ea_model=ea_model,
        dtype=_dtype_from_name(args.dtype),
        temperature=args.temperature,
        apply_chat_template=bool(args.apply_chat_template),
        max_new_tokens=args.max_new_tokens,
        eval_max_examples=args.eval_max_examples,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
