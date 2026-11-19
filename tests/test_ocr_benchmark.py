import argparse
import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    path = REPO_ROOT / "benchmarks" / "ocr_benchmark" / "run_olmocr_bench.py"
    spec = importlib.util.spec_from_file_location("run_olmocr_bench", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _AlwaysPass:
    def run(self, content):
        return True, ""


def test_select_rows_keeps_all_tests_for_each_sampled_pdf():
    bench = _load_benchmark()
    rows = [{"pdf": f"{n}.pdf", "id": f"{n}-{i}"} for n in range(5) for i in range(3)]

    first = bench.select_rows(rows, max_pdfs=2, seed=0)

    assert len({row["pdf"] for row in first}) == 2
    assert len(first) == 6
    assert first == bench.select_rows(rows, max_pdfs=2, seed=0)


def test_select_rows_negative_limit_keeps_everything():
    bench = _load_benchmark()
    rows = [{"pdf": f"{n}.pdf"} for n in range(5)]

    assert bench.select_rows(rows, max_pdfs=-1, seed=0) == rows


def test_empty_ocr_output_fails_tests_that_would_otherwise_pass():
    bench = _load_benchmark()
    result = bench.SplitResult()

    bench.score_pdf([{"type": "absent"}], "  \n", lambda row: _AlwaysPass(), result)

    assert (result.overall.passed, result.overall.total) == (0, 1)


def test_score_pdf_tallies_by_test_type():
    bench = _load_benchmark()
    result = bench.SplitResult()

    bench.score_pdf(
        [{"type": "present"}, {"type": "present"}, {"type": "order"}],
        "text",
        lambda row: _AlwaysPass(),
        result,
    )

    assert result.by_type["present"].total == 2
    assert result.by_type["order"].passed == 1
    assert result.overall.rate == 1.0


def test_failed_extraction_is_not_cached(tmp_path):
    bench = _load_benchmark()
    cache_path = tmp_path / "out" / "a.md"
    outputs = iter(["", "ocr text"])

    def extract(path, use_ocr, tier):
        return next(outputs)

    assert bench.extract_cached(Path("a.pdf"), cache_path, "basic", extract)[0] == ""
    assert not cache_path.exists()
    assert bench.extract_cached(Path("a.pdf"), cache_path, "basic", extract)[0] == "ocr text"
    assert bench.extract_cached(Path("a.pdf"), cache_path, "basic", extract) == ("ocr text", 0.0)


def test_baseline_row_is_added_only_when_the_split_has_none():
    bench = _load_benchmark()

    assert bench.baseline_row("a.pdf", [{"type": "present"}]) == {
        "pdf": "a.pdf",
        "page": 1,
        "id": "a.pdf_baseline",
        "type": "baseline",
    }
    assert bench.baseline_row("a.pdf", [{"type": "present"}, {"type": "baseline"}]) is None


def test_olmocr_score_averages_group_pass_rates_not_tests():
    bench = _load_benchmark()
    one_of_one = bench.SplitResult()
    one_of_one.by_type["present"].add(True)
    none_of_three = bench.SplitResult()
    for _ in range(3):
        none_of_three.by_type["present"].add(False)
    baseline = bench.SplitResult()
    baseline.by_type["baseline"].add(True)
    baseline.by_type["baseline"].add(False)

    score = bench.olmocr_score({"a": one_of_one, "b": none_of_three}, baseline)

    # (1/1 + 0/3 + 1/2) / 3 groups, not 2 of 6 tests.
    assert score == 0.5


def test_rerun_reuses_saved_ocr_only_for_the_same_mineru_version(tmp_path, monkeypatch):
    installed = {"mineru": "4.0.12"}
    monkeypatch.setattr(importlib.metadata, "version", installed.__getitem__)
    bench = _load_benchmark()
    tests_file = tmp_path / "split.jsonl"
    tests_file.write_text('{"pdf": "a.pdf", "page": 1, "id": "a_1", "type": "present"}\n')
    hub = types.SimpleNamespace(hf_hub_download=lambda repo, name, repo_type: str(tests_file))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    args = argparse.Namespace(max_pdfs=-1, seed=0, output_dir=tmp_path / "out", tier="standard")
    ocr_calls = []

    def extract(path, use_ocr, tier):
        ocr_calls.append(path)
        return "ocr text"

    for mineru_version in ["4.0.12", "4.0.12", "4.1.0"]:
        installed["mineru"] = mineru_version
        bench.run_split("split", args, extract, lambda row: _AlwaysPass(), bench.SplitResult())

    assert len(ocr_calls) == 2
