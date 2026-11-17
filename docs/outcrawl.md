# Web documents from Outcrawl

Use the opt-in reader to search the web and save source-attributed Markdown for
LEANN's document ingestion. [Outcrawl](https://outcrawl.ai) is a search and scrape
API: the reader searches, and with `--scrape` it also fetches each result page as
Markdown, so LEANN indexes whole pages instead of search snippets. It needs an
API key (create one at https://outcrawl.ai/console/keys; the free tier has 1,000
credits a month). This reader does not retry failed requests.

The search query and selected URLs are sent to Outcrawl. This is a remote data
source, unlike LEANN's local retrieval. Avoid including private data in your
search. No existing LEANN application or provider default changes.

## Install and export

From the repository root, create a separate environment for the reader (Python
3.10 or newer). It does not require the LEANN runtime or an LLM:

```bash
uv venv /tmp/leann-outcrawl
uv pip install --python /tmp/leann-outcrawl/bin/python -r apps/outcrawl_data/requirements.txt
export OUTCRAWL_API_KEY=...
/tmp/leann-outcrawl/bin/python -m apps.outcrawl_data \
  --search-query "LEANN low storage vector index" \
  --max-results 3 \
  --output-dir ./web-data
```

Add `--scrape` to scrape the selected result pages to Markdown. This uses
`https://api.outcrawl.ai/v1/search` and `https://api.outcrawl.ai/v1/scrape`. A
page that cannot be scraped keeps its search snippet and prints a warning.
`--max-results` (1-20, default 5) sets how many results are requested and kept.
`--timeout` bounds the entire operation (90 seconds by default).

Each file is named after its URL (for example
`arxiv-org-abs-2506-08276-<hash>.md`) and contains its title, source URL and
text. Repeating a search refreshes files for matching URLs; other files remain.
Use a fresh directory for a separate collection.

## Search, index, and ask in one command

Inside your LEANN environment, install the reader requirements and run the RAG
app. It searches, chunks the text, builds a LEANN index, and answers with the
configured LLM (`--llm`, see the other `apps/*_rag.py` examples):

```bash
uv pip install -r apps/outcrawl_data/requirements.txt
python -m apps.outcrawl_rag \
  --search-query "LEANN low storage vector index" \
  --scrape --query "How does LEANN save storage?"
```

Each chunk keeps the page title and source URL so answers can cite them. The
index is reused on later runs; pass `--force-rebuild` after changing the search.

## Use the exported documents

In your existing [LEANN installation](../README.md#installation), ingest the
exported directory using the normal document workflow:

```bash
leann build web-notes --docs ./web-data
leann search web-notes "How does LEANN save storage?"
```

Embedding configuration follows your existing LEANN setup. The export command
only collects documents; it does not build an index or run an agent loop.

For Python callers, `OutcrawlReader.read_web_data()` returns dictionaries with
`text` and `metadata` fields, including `url`, `title`, and `source`. Pass
`api_key=` to the constructor instead of setting `OUTCRAWL_API_KEY`.

## Reader tests

```bash
uv pip install --python /tmp/leann-outcrawl/bin/python pytest
/tmp/leann-outcrawl/bin/python -m pytest -o addopts= tests/test_outcrawl_reader.py
```
