"""
Web RAG example using the Outcrawl search and scrape API and the unified interface.

Searches the web through Outcrawl (optionally scraping each result page to
Markdown), then chunks and indexes the text in LEANN. Needs OUTCRAWL_API_KEY.

Usage:
    python -m apps.outcrawl_rag --search-query "LEANN low storage vector index" \\
        --scrape --query "How does LEANN save storage?"
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks
from llama_index.core import Document

from .outcrawl_data.outcrawl_reader import OutcrawlReader


class OutcrawlWebRAG(BaseRAGExample):
    """RAG example for web documents from Outcrawl search and scrape."""

    def __init__(self):
        super().__init__(
            name="Outcrawl Web",
            description="Search the web with Outcrawl and run RAG over the results",
            default_index_name="outcrawl_web_index",
        )

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        web_group = parser.add_argument_group("Outcrawl Web Parameters")
        web_group.add_argument("--search-query", required=True, help="What to search the web for")
        web_group.add_argument("--max-results", type=int, default=5, help="Use 1-20 search results")
        web_group.add_argument(
            "--scrape", action="store_true", help="Scrape each result page to Markdown"
        )
        web_group.add_argument(
            "--timeout", type=float, default=90, help="Overall web timeout in seconds"
        )
        web_group.add_argument(
            "--chunk-size", type=int, default=256, help="Text chunk size (default: 256)"
        )
        web_group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        print(f"Searching the web via Outcrawl: {args.search_query}")
        web_documents = await OutcrawlReader(timeout=args.timeout).read_web_data(
            args.search_query, args.max_results, args.scrape
        )
        for document in web_documents:
            print(f"  - {document['metadata']['title']} ({document['metadata']['url']})")
        documents = [
            # Keep title and URL in the text so the LLM can cite its sources.
            Document(
                text=f"{doc['metadata']['title']}\nSource: {doc['metadata']['url']}\n\n{doc['text']}",
                metadata=doc["metadata"],
            )
            for doc in web_documents
        ]
        texts = create_text_chunks(
            documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        print(f"Created {len(texts)} text chunks from {len(documents)} web documents")
        return texts


if __name__ == "__main__":
    asyncio.run(OutcrawlWebRAG().run())
