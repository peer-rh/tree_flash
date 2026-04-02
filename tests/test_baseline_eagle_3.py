from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")


ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = ROOT / "baselines"

spec = importlib.util.spec_from_file_location("baseline_eagle_3_eval", BASELINES_DIR / "eagle_3_eval.py")
assert spec is not None and spec.loader is not None
baseline_eagle_3_eval = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline_eagle_3_eval
spec.loader.exec_module(baseline_eagle_3_eval)


def test_resolve_eagle_repo_path_requires_argument_or_env(monkeypatch) -> None:
    monkeypatch.delenv("EAGLE_REPO", raising=False)
    with pytest.raises(ValueError, match="EAGLE_REPO"):
        baseline_eagle_3_eval.resolve_eagle_repo_path(None)


def test_get_upstream_ea_model_class_rejects_invalid_checkout(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="eagle.model.ea_model"):
        baseline_eagle_3_eval._get_upstream_ea_model_class(tmp_path)


def test_load_upstream_eagle3_model_imports_from_supplied_checkout(tmp_path: Path) -> None:
    eagle_repo = tmp_path / "eagle_repo"
    (eagle_repo / "eagle" / "model").mkdir(parents=True)
    (eagle_repo / "eagle" / "__init__.py").write_text("", encoding="utf-8")
    (eagle_repo / "eagle" / "model" / "__init__.py").write_text("", encoding="utf-8")
    (eagle_repo / "eagle" / "model" / "ea_model.py").write_text(
        """
class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded"

class _BaseModel:
    def parameters(self):
        return iter([])

class EaModel:
    from_pretrained_calls = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.from_pretrained_calls.append(kwargs)
        obj = cls()
        obj.base_model = _BaseModel()
        return obj

    def eval(self):
        return self

    def get_tokenizer(self):
        return _Tokenizer()
""",
        encoding="utf-8",
    )

    model, tokenizer = baseline_eagle_3_eval.load_upstream_eagle3_model(
        eagle_repo=eagle_repo,
        base_model="base-model",
        ea_model="ea-model",
        dtype=baseline_eagle_3_eval._dtype_from_name("float16"),
        total_token=11,
        depth=7,
        top_k=5,
        threshold=0.75,
    )

    assert tokenizer.decode([1]) == "decoded"
    assert model.__class__.from_pretrained_calls == [
        {
            "base_model_path": "base-model",
            "ea_model_path": "ea-model",
            "torch_dtype": baseline_eagle_3_eval._dtype_from_name("float16"),
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "total_token": 11,
            "depth": 7,
            "top_k": 5,
            "threshold": 0.75,
            "use_eagle3": True,
        }
    ]


def test_resolve_model_pair_handles_alias_and_direct_model() -> None:
    base_model, ea_model = baseline_eagle_3_eval.resolve_model_pair(
        official_eagle3_model="llama-3.1-8b-instruct",
        ea_model=None,
        base_model=None,
    )
    assert ea_model == "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    assert base_model == "meta-llama/Llama-3.1-8B-Instruct"

    with pytest.raises(ValueError, match="base model is required"):
        baseline_eagle_3_eval.resolve_model_pair(
            official_eagle3_model=None,
            ea_model="custom/eagle3-model",
            base_model=None,
        )


def test_run_standard_eval_suite_writes_outputs_and_aggregates(monkeypatch, tmp_path: Path) -> None:
    sys.modules.pop("src.eagle3_eval", None)
    dataset_order: list[str] = []

    monkeypatch.setattr(
        baseline_eagle_3_eval,
        "load_upstream_eagle3_model",
        lambda **kwargs: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        baseline_eagle_3_eval,
        "build_upstream_generate_fn",
        lambda **kwargs: (lambda *, prompt, max_new_tokens: None),
    )

    class FakeReport(SimpleNamespace):
        pass

    def fake_evaluate_prompt_suite(*, data_name, tokenizer, generate_fn, max_examples, max_new_tokens):
        del tokenizer, generate_fn, max_examples, max_new_tokens
        dataset_order.append(data_name)
        idx = len(dataset_order)
        return FakeReport(
            total_examples=idx,
            exact_matches=0,
            accepted_count=0,
            rejected_count=0,
            extra_count=idx * 10,
            missing_count=0,
            examples=[],
        )

    monkeypatch.setattr(baseline_eagle_3_eval, "evaluate_prompt_suite", fake_evaluate_prompt_suite)
    monkeypatch.setattr(baseline_eagle_3_eval, "render_eval_suite_html", lambda report: "<html>report</html>")

    summary = baseline_eagle_3_eval.run_standard_eval_suite(
        eagle_repo=tmp_path / "upstream",
        output_dir=tmp_path / "outputs",
        base_model="base-model",
        ea_model="ea-model",
        dtype=baseline_eagle_3_eval._dtype_from_name("float16"),
        temperature=0.0,
        apply_chat_template=False,
        max_new_tokens=128,
        eval_max_examples=3,
        total_token=11,
        depth=7,
        top_k=5,
        threshold=0.75,
    )

    assert dataset_order == ["humaneval", "gsm8k", "alpaca"]
    assert "src.eagle3_eval" not in sys.modules
    for data_name in baseline_eagle_3_eval.STANDARD_EVAL_DATASETS:
        dataset_dir = tmp_path / "outputs" / data_name
        assert (dataset_dir / "summary.json").is_file()
        assert (dataset_dir / "report.html").is_file()
        assert (dataset_dir / "stdout.log").is_file()

    top_level = json.loads((tmp_path / "outputs" / "summary.json").read_text(encoding="utf-8"))
    assert top_level["ea_model"] == "ea-model"
    assert top_level["base_model"] == "base-model"
    assert top_level["datasets"] == ["humaneval", "gsm8k", "alpaca"]
    assert top_level["results"]["gsm8k"]["extra_count"] == 20
    assert summary["results"]["alpaca"]["extra_count"] == 30


def test_shell_wrapper_targets_helper_module() -> None:
    wrapper = (BASELINES_DIR / "eagle-3").read_text(encoding="utf-8")
    assert "#!/usr/bin/env bash" in wrapper
    assert "uv run -m baselines.eagle_3_eval" in wrapper
