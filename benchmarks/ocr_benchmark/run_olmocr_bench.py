#!/usr/bin/env python3
"""Score LEANN's PDF OCR path on olmOCR-Bench.

olmOCR-Bench has no full-page transcriptions. Each row is a pass/fail test on one
PDF, for example "this sentence is present" or "this text comes before that text".
So this script reports the share of tests passed, using olmOCR's own scorer,
rather than CER or WER.

See README.md in this directory for setup and results.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_REPO = "allenai/olmOCR-bench"
DEFAULT_SPLITS = ["old_scans", "multi_column", "long_tiny_text", "headers_footers", "table_tests"]
# The math splits are opt-in because scoring them renders equations in a Playwright browser.
ALL_SPLITS = [*DEFAULT_SPLITS, "arxiv_math", "old_scans_math"]


class BenchTest(Protocol):
    def run(self, content: str) -> tuple[bool, str]: ...


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


def score_pdf(
    rows: list[dict[str, Any]],
    markdown: str,
    load_test: Callable[[dict[str, Any]], BenchTest],
    result: SplitResult,
) -> None:
    """Run one PDF's tests against its OCR output and add the outcomes to result."""
    for row in rows:
        # Empty output would pass every "absent" test, so a failed extraction would
        # raise the score. Count all of its tests as failed instead.
        passed = bool(markdown.strip()) and load_test(row).run(markdown)[0]
        result.by_type[row["type"]].add(passed)


def extract_cached(
    pdf_path: Path, cache_path: Path, tier: str, extract: Callable
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
    split: str, args: argparse.Namespace, extract: Callable, load_test: Callable
) -> SplitResult:
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
        result.pdfs += 1
        result.failed_pdfs += not markdown.strip()
        result.seconds += seconds
        print(f"[{split} {number}/{len(rows_by_pdf)}] {pdf}: {seconds:.0f}s", flush=True)
    return result


def summarize(results: dict[str, SplitResult], args: argparse.Namespace) -> dict[str, Any]:
    from importlib.metadata import version

    return {
        "mineru_version": version("mineru"),
        "tier": args.tier,
        "max_pdfs_per_split": args.max_pdfs,
        "seed": args.seed,
        "splits": {
            split: {
                "pdfs": result.pdfs,
                "failed_pdfs": result.failed_pdfs,
                "ocr_seconds": round(result.seconds, 1),
                "passed": result.overall.passed,
                "total": result.overall.total,
                "pass_rate": round(result.overall.rate, 4),
                "by_type": {
                    name: {"passed": tally.passed, "total": tally.total}
                    for name, tally in sorted(result.by_type.items())
                },
            }
            for split, result in results.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=DEFAULT_SPLITS)
    parser.add_argument("--tier", choices=["basic", "standard", "advanced"], default="basic")
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
        help="Directory for saved OCR text and summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sys.path.insert(0, str(REPO_ROOT / "apps"))
    from document_rag import OCR_AVAILABLE, extract_pdf_with_ocr_fallback

    if not OCR_AVAILABLE:
        sys.exit('MinerU is not installed. Run: pip install "mineru>=4.0,<5"')
    try:
        from olmocr.bench.tests import load_single_test
    except ImportError as e:
        sys.exit(f"olmOCR's scorer is not available ({e}). See the setup steps in README.md.")

    results = {
        split: run_split(split, args, extract_pdf_with_ocr_fallback, load_single_test)
        for split in args.splits
    }
    summary = summarize(results, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"summary_{args.tier}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nMinerU {summary['mineru_version']}, tier={args.tier}")
    for split, row in summary["splits"].items():
        print(
            f"{split}: {row['passed']}/{row['total']} tests passed ({row['pass_rate']:.1%}), "
            f"{row['pdfs']} PDFs, {row['failed_pdfs']} failed, {row['ocr_seconds']:.0f}s of OCR"
        )
    print(f"Saved {summary_path}")


# MinerU starts worker processes, and on Windows they re-import this file.
if __name__ == "__main__":
    main()
