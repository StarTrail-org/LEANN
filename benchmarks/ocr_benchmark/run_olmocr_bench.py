#!/usr/bin/env python3
"""Score LEANN's PDF OCR path on olmOCR-Bench.

olmOCR-Bench has no full-page transcriptions. Each row is a pass/fail test on one
PDF, for example "this sentence is present" or "this text comes before that text".
So this script runs olmOCR's own tests rather than computing CER or WER, and scores
the run the way olmOCR's benchmark.py does: every PDF also gets a baseline test,
and the score is the average pass rate of the splits and the baseline group.

See README.md in this directory for setup and results.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, get_args

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_REPO = "allenai/olmOCR-bench"
DEFAULT_SPLITS = ["old_scans", "multi_column", "long_tiny_text", "headers_footers", "table_tests"]
# The math splits are opt-in because scoring them renders equations in a Playwright browser.
ALL_SPLITS = [*DEFAULT_SPLITS, "arxiv_math", "old_scans_math"]
# olmOCR adds a baseline test to every PDF that lacks one. It fails blank, repetitive,
# or wrong-script output, and it counts as its own group in the score.
BASELINE = "baseline"


class BenchTest(Protocol):
    def run(self, content: str) -> tuple[bool, str]: ...


LoadTest = Callable[[dict[str, Any]], BenchTest]
Extract = Callable[..., str]


@dataclass
class Tally:
    passed: int = 0
    total: int = 0

    def add(self, passed: bool) -> None:
        self.passed += passed
        self.total += 1

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass
class SplitResult:
    by_type: dict[str, Tally] = field(default_factory=lambda: defaultdict(Tally))
    pdfs: int = 0
    failed_pdfs: int = 0
    seconds: float = 0.0

    @property
    def overall(self) -> Tally:
        return Tally(
            passed=sum(t.passed for t in self.by_type.values()),
            total=sum(t.total for t in self.by_type.values()),
        )


def select_rows(rows: list[dict[str, Any]], max_pdfs: int, seed: int) -> list[dict[str, Any]]:
    """Keep every test row for a seeded sample of PDFs, so a run can be repeated."""
    pdfs = sorted({row["pdf"] for row in rows})
    if 0 < max_pdfs < len(pdfs):
        pdfs = random.Random(seed).sample(pdfs, max_pdfs)
    chosen = set(pdfs)
    return [row for row in rows if row["pdf"] in chosen]


def baseline_row(pdf: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the baseline test olmOCR adds to a PDF, or None if its split has one."""
    if any(row["type"] == BASELINE for row in rows):
        return None
    return {"pdf": pdf, "page": 1, "id": f"{pdf}_baseline", "type": BASELINE}


def olmocr_score(results: dict[str, SplitResult], baseline: SplitResult) -> float:
    """Average the pass rates of the splits and the baseline group, as olmOCR does."""
    rates = [
        result.overall.rate for result in [*results.values(), baseline] if result.overall.total
    ]
    return sum(rates) / len(rates) if rates else 0.0


def score_pdf(
    rows: list[dict[str, Any]],
    markdown: str,
    load_test: LoadTest,
    result: SplitResult,
) -> None:
    """Run one PDF's tests against its OCR output and add the outcomes to result."""
    for row in rows:
        # Empty output would pass every "absent" test, so a failed extraction would
        # raise the score. Count all of its tests as failed instead.
        passed = bool(markdown.strip()) and load_test(row).run(markdown)[0]
        result.by_type[row["type"]].add(passed)


def extract_cached(
    pdf_path: Path, cache_path: Path, tier: str, extract: Extract
) -> tuple[str, float]:
    """Return OCR text and the seconds spent, reusing a saved result when there is one."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), 0.0
    start = time.perf_counter()
    markdown = extract(str(pdf_path), use_ocr=True, tier=tier)
    seconds = time.perf_counter() - start
    # Failed extractions are not saved, so the next run retries them.
    if markdown.strip():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(markdown, encoding="utf-8")
    return markdown, seconds


def run_split(
    split: str,
    args: argparse.Namespace,
    extract: Extract,
    load_test: LoadTest,
    baseline: SplitResult,
) -> SplitResult:
    """OCR a split's sampled PDFs and score them, adding baseline outcomes to baseline."""
    from huggingface_hub import hf_hub_download

    tests_file = hf_hub_download(DATASET_REPO, f"bench_data/{split}.jsonl", repo_type="dataset")
    with open(tests_file, encoding="utf-8") as f:
        rows = select_rows(
            [json.loads(line) for line in f if line.strip()], args.max_pdfs, args.seed
        )

    rows_by_pdf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_pdf[row["pdf"]].append(row)

    result = SplitResult()
    for number, (pdf, pdf_rows) in enumerate(sorted(rows_by_pdf.items()), start=1):
        pdf_path = Path(
            hf_hub_download(DATASET_REPO, f"bench_data/pdfs/{pdf}", repo_type="dataset")
        )
        cache_path = args.output_dir / args.tier / f"{pdf}.md"
        markdown, seconds = extract_cached(pdf_path, cache_path, args.tier, extract)
        score_pdf(pdf_rows, markdown, load_test, result)
        if extra := baseline_row(pdf, pdf_rows):
            score_pdf([extra], markdown, load_test, baseline)
        result.pdfs += 1
        result.failed_pdfs += not markdown.strip()
        result.seconds += seconds
        print(f"[{split} {number}/{len(rows_by_pdf)}] {pdf}: {seconds:.0f}s", flush=True)
    return result


def summarize(
    results: dict[str, SplitResult], baseline: SplitResult, args: argparse.Namespace
) -> dict[str, Any]:
    from importlib.metadata import version

    return {
        "mineru_version": version("mineru"),
        "tier": args.tier,
        "max_pdfs_per_split": args.max_pdfs,
        "seed": args.seed,
        "score": olmocr_score(results, baseline),
        "baseline": {
            "passed": baseline.overall.passed,
            "total": baseline.overall.total,
            "pass_rate": baseline.overall.rate,
        },
        "splits": {
            split: {
                "pdfs": result.pdfs,
                "failed_pdfs": result.failed_pdfs,
                "ocr_seconds": round(result.seconds, 1),
                "passed": result.overall.passed,
                "total": result.overall.total,
                "pass_rate": result.overall.rate,
                "by_type": {
                    name: {"passed": tally.passed, "total": tally.total}
                    for name, tally in sorted(result.by_type.items())
                },
            }
            for split, result in results.items()
        },
    }


def parse_args(tiers: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=DEFAULT_SPLITS)
    parser.add_argument("--tier", choices=tiers, default="standard")
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=10,
        help="PDFs to sample from each split, or -1 for all of them (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "data" / "ocr_benchmark",
        help="Directory for saved OCR text and summary_<tier>.json",
    )
    return parser.parse_args()


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "apps"))
    from document_rag import (
        OCR_AVAILABLE,
        OCR_INSTALL_HINT,
        OcrTier,
        extract_pdf_with_ocr_fallback,
    )

    args = parse_args(get_args(OcrTier))
    if not OCR_AVAILABLE:
        sys.exit(f"MinerU is not installed. Run: {OCR_INSTALL_HINT}")
    try:
        from olmocr.bench.tests import load_single_test
    except ImportError as e:
        sys.exit(f"olmOCR's scorer is not available ({e}). See the setup steps in README.md.")

    baseline = SplitResult()
    results = {
        split: run_split(split, args, extract_pdf_with_ocr_fallback, load_single_test, baseline)
        for split in args.splits
    }
    summary = summarize(results, baseline, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"summary_{args.tier}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nMinerU {summary['mineru_version']}, tier={args.tier}")
    for split, row in summary["splits"].items():
        print(
            f"{split}: {row['passed']}/{row['total']} tests passed ({row['pass_rate']:.1%}), "
            f"{row['pdfs']} PDFs, {row['failed_pdfs']} failed, {row['ocr_seconds']:.0f}s of OCR"
        )
    row = summary["baseline"]
    print(f"{BASELINE}: {row['passed']}/{row['total']} tests passed ({row['pass_rate']:.1%})")
    print(f"Score (average of the groups above, as olmOCR reports it): {summary['score']:.1%}")
    print(f"Saved {summary_path}")


# MinerU starts worker processes, and on Windows they re-import this file.
if __name__ == "__main__":
    main()
