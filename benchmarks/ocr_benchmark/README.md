# OCR accuracy on olmOCR-Bench

`run_olmocr_bench.py` runs LEANN's OCR path (`extract_pdf_with_ocr_fallback` in `apps/document_rag.py`, with OCR forced as `--use-ocr` does) on [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) and scores the output.

The dataset has no full-page transcriptions. Each row is a pass/fail test on one PDF, for example "this sentence is present" or "this text comes before that text". So the script runs olmOCR's own tests rather than computing CER or WER, and it scores the run the way olmOCR's `benchmark.py` does:

- Every PDF also gets olmOCR's baseline test, which fails blank, repetitive, or wrong-script output. The baseline tests form their own group.
- The score is the average pass rate of the splits and the baseline group, not the share of all tests passed.

One rule is stricter than olmOCR's. If OCR returns no text for a PDF, all of its tests count as failed, because empty output would otherwise pass every "absent" test.

## Setup

Install `olmocr` with `--no-deps`, because the scorer needs only a few of its dependencies:

```bash
pip install "mineru>=4.0,<5"
pip install --no-deps olmocr
pip install fuzzysearch rapidfuzz playwright beautifulsoup4 lxml
```

## Run

```bash
python benchmarks/ocr_benchmark/run_olmocr_bench.py --tier standard --max-pdfs 10
```

The script downloads the tests and PDFs from Hugging Face. It saves the OCR text and `summary_<tier>.json` to `benchmarks/data/ocr_benchmark/`, which git ignores. A rerun reuses saved OCR text.

`arxiv_math` and `old_scans_math` are off by default. To score them, pass them to `--splits` and run `playwright install chromium`, because their tests render equations in a browser.

## Results

These results are from a small sample of 10 PDFs per split (`--max-pdfs 10 --seed 0`), so treat them as a rough guide and not as a published score. The runs used MinerU 4.0.12 on Windows 11 with an RTX 3060 Laptop GPU (6 GB). GPU memory use peaked at 3.5 GB with `standard` and `advanced` and at 1.6 GB with `flash`, against 1.2 GB when idle.

| Group | `flash` | `basic` | `standard` | `advanced` |
|---|---|---|---|---|
| `old_scans` | 16/49 (32.7%) | 16/49 (32.7%) | 21/49 (42.9%) | 22/49 (44.9%) |
| `multi_column` | 29/36 (80.6%) | 30/36 (83.3%) | 32/36 (88.9%) | 30/36 (83.3%) |
| `long_tiny_text` | 54/61 (88.5%) | 53/61 (86.9%) | 54/61 (88.5%) | 57/61 (93.4%) |
| `headers_footers` | 21/23 (91.3%) | 21/23 (91.3%) | 23/23 (100%) | 23/23 (100%) |
| `table_tests` | 1/47 (2.1%) | 24/47 (51.1%) | 36/47 (76.6%) | 44/47 (93.6%) |
| baseline | 46/49 (93.9%) | 46/49 (93.9%) | 48/49 (98.0%) | 48/49 (98.0%) |
| **Score** | **64.8%** | **73.2%** | **82.5%** | **85.5%** |
| OCR time for 50 PDFs | 340s | 580s | 645s | 669s |

Running olmOCR's `benchmark.py` on the same OCR output gives the same score for `standard` and `advanced`. For `flash` and `basic` it gives about 1.5 points more (66.3% and 74.6%), because each returned no text for one `headers_footers` PDF, and olmOCR lets that empty output pass the PDF's "absent" tests.

`advanced` scored highest, mostly on tables, and took 4% longer than `standard`. `standard` stays the default in `apps/document_rag.py`, because it is MinerU's own default for `parse()`. `flash` was the fastest, but it writes tables as plain text instead of Markdown tables, so it fails almost every table test even though the cell text is there.

`old_scans` scores lowest. Its tests allow few or no character differences, so small misreads fail them. In the handwritten `old_scans/1.pdf`, for example, `standard` read "Desired" where a test expects "Desire".
