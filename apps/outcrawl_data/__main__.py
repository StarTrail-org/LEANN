"""Export Outcrawl search results (optionally scraped pages) to Markdown for LEANN."""

import argparse
import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .outcrawl_reader import OutcrawlReader


def document_filename(url: str) -> str:
    """Name a file after its URL: readable host/path slug plus a short hash for uniqueness."""
    parts = urlsplit(url)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{parts.netloc}{parts.path}").strip("-").lower()
    digest = hashlib.sha256(url.encode()).hexdigest()[:10]
    return f"{slug[:80].rstrip('-') or 'page'}-{digest}.md"


def write_documents(documents: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Write one UTF-8 Markdown document per URL, refreshing matching files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for document in documents:
        metadata = document["metadata"]
        path = output_dir / document_filename(metadata["url"])
        lines = [f"# {metadata['title']}", "", f"Source: {metadata['url']}", "", document["text"]]
        path.write_text("\n".join([*lines, ""]), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    """Run the Outcrawl export command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-query", required=True, help="What to search the web for")
    parser.add_argument("--max-results", type=int, default=5, help="Use 1-20 search results")
    parser.add_argument("--scrape", action="store_true", help="Scrape each result page to Markdown")
    parser.add_argument("--timeout", type=float, default=90, help="Overall timeout in seconds")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for exported Markdown"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    documents = asyncio.run(
        OutcrawlReader(timeout=args.timeout).read_web_data(
            args.search_query, args.max_results, args.scrape
        )
    )
    if not documents:
        parser.exit(1, "No nonempty web documents returned; nothing was written.\n")
    paths = write_documents(documents, args.output_dir)
    print(f"Exported {len(paths)} web documents to {args.output_dir}")


if __name__ == "__main__":
    main()
