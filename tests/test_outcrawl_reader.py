"""Exercise the real Outcrawl reader/transport against an HTTP fixture."""

import asyncio
import importlib
import json
import sys

import pytest

# The opt-in reader has its own installation; do not add its requirements to
# collection of LEANN's ordinary test suite.
httpx = pytest.importorskip("httpx")
reader = importlib.import_module("apps.outcrawl_data.outcrawl_reader")
export = importlib.import_module("apps.outcrawl_data.__main__")


def install_server(monkeypatch, hits, pages=None, status=200):
    """Install an HTTP server fixture that observes every request the reader sends."""
    requests = []
    pages = pages or {}

    def handle(request):
        requests.append(request)
        assert request.headers["User-Agent"] == reader.USER_AGENT
        assert request.headers["Authorization"] == "Bearer test-key"
        assert "cookie" not in request.headers
        payload = json.loads(request.content)
        if str(request.url) == f"{reader.API_URL}/search":
            if status != 200:
                return httpx.Response(status, json={"code": "error", "message": "nope"})
            return httpx.Response(200, json={"hits": hits, "usage": {"credits": "1"}})
        assert str(request.url) == f"{reader.API_URL}/scrape"
        assert payload["formats"] == ["markdown"] and payload["onlyMainContent"] is True
        page = pages.get(payload["url"])
        if page is None:
            return httpx.Response(502, json={"code": "page_unreachable", "message": "gone"})
        return httpx.Response(200, json={"url": payload["url"], "statusCode": 200, **page})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        reader.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs),
    )
    return requests


def hit(url, snippet="Search snippet", title="A source"):
    return {"url": url, "title": title, "snippet": snippet, "rank": 1}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OUTCRAWL_API_KEY", " test-key ")


def test_search_scrape_real_transport(monkeypatch, caplog):
    first, second = hit("https://example.com/one"), hit("https://example.com/two", "Kept snippet")
    requests = install_server(
        monkeypatch,
        [first, first, {"title": "no url"}, second, hit("https://example.com/three")],
        {first["url"]: {"markdown": "# Page\n\nScraped text", "metadata": {"title": "Page title"}}},
    )
    documents = asyncio.run(reader.OutcrawlReader().read_web_data("research web sources", 2, True))
    assert json.loads(requests[0].content) == {"query": "research web sources", "limit": 2}
    scraped = sorted(json.loads(r.content)["url"] for r in requests[1:])
    assert scraped == [first["url"], second["url"]]
    assert documents[0] == {
        "text": "# Page\n\nScraped text",
        "metadata": {"url": first["url"], "title": "Page title", "source": "outcrawl"},
    }
    assert documents[1]["text"] == "Kept snippet"
    assert documents[1]["metadata"]["title"] == "A source"
    assert "keeping its snippet" in caplog.text


def test_search_only_and_blank_snippets(monkeypatch):
    requests = install_server(
        monkeypatch, [hit("https://example.com/empty", "  "), hit("https://example.com", title="")]
    )
    documents = asyncio.run(reader.OutcrawlReader().read_web_data("web sources"))
    assert len(requests) == 1
    assert documents == [
        {
            "text": "Search snippet",
            "metadata": {
                "url": "https://example.com",
                "title": "https://example.com",
                "source": "outcrawl",
            },
        }
    ]


def test_empty_search_does_not_scrape(monkeypatch):
    requests = install_server(monkeypatch, [])
    assert asyncio.run(reader.OutcrawlReader().read_web_data("sources", scrape=True)) == []
    assert len(requests) == 1


@pytest.mark.parametrize(
    "status,reason", [(401, "invalid API key"), (402, "out of credits"), (429, "rate limited")]
)
def test_search_errors_propagate(monkeypatch, status, reason):
    install_server(monkeypatch, [], status=status)
    with pytest.raises(RuntimeError, match=f"HTTP {status}\\): {reason}"):
        asyncio.run(reader.OutcrawlReader().read_web_data("sources"))


def test_missing_hits_list_is_an_error(monkeypatch):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"results": []}))
    monkeypatch.setattr(
        reader.httpx, "AsyncClient", lambda **kwargs: client_class(transport=transport, **kwargs)
    )
    with pytest.raises(RuntimeError, match="missing its hits list"):
        asyncio.run(reader.OutcrawlReader().read_web_data("sources"))


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("OUTCRAWL_API_KEY")
    with pytest.raises(ValueError, match="OUTCRAWL_API_KEY"):
        reader.OutcrawlReader()
    assert reader.OutcrawlReader(api_key="explicit").api_key == "explicit"


@pytest.mark.parametrize("query,limit", [("", 1), (" ", 1), ("x", 0), ("x", 21)])
def test_invalid_arguments(query, limit):
    with pytest.raises(ValueError):
        asyncio.run(reader.OutcrawlReader().read_web_data(query, limit))


def test_overall_timeout(monkeypatch):
    async def stalled(*args):
        await asyncio.Event().wait()

    instance = reader.OutcrawlReader(timeout=0.01)
    monkeypatch.setattr(instance, "_read", stalled)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(instance.read_web_data("sources"))


def test_export_command_preserves_sources(monkeypatch, tmp_path, capsys):
    async def read_data(self, query, limit, scrape):
        assert (query, limit, scrape) == ("web sources", 5, False)
        return [
            {
                "text": "Useful café snippet",
                "metadata": {"url": "https://example.com/../../source", "title": "A source"},
            }
        ]

    monkeypatch.setattr(reader.OutcrawlReader, "read_web_data", read_data)
    monkeypatch.setattr(
        sys,
        "argv",
        ["outcrawl_data", "--search-query", "web sources", "--output-dir", str(tmp_path)],
    )
    export.main()
    paths = list(tmp_path.glob("*.md"))
    assert len(paths) == 1
    assert paths[0].parent == tmp_path
    assert paths[0].name.startswith("example-com-source-")
    text = paths[0].read_text()
    assert text.startswith("# A source\n\nSource: https://example.com/../../source\n")
    assert "Useful café snippet" in text
    assert "Exported 1" in capsys.readouterr().out


def test_rag_app_chunks_web_documents_with_sources(monkeypatch):
    pytest.importorskip("leann")
    pytest.importorskip("llama_index.core")
    rag = importlib.import_module("apps.outcrawl_rag")

    async def read_data(self, query, limit, scrape):
        assert (query, limit, scrape) == ("web sources", 5, True)
        return [
            {
                "text": "LEANN recomputes embeddings. " * 40,
                "metadata": {"url": "https://a.io/x", "title": "A source", "source": "outcrawl"},
            }
        ]

    monkeypatch.setattr(reader.OutcrawlReader, "read_web_data", read_data)
    app = rag.OutcrawlWebRAG()
    args = app.parser.parse_args(["--search-query", "web sources", "--scrape"])
    texts = asyncio.run(app.load_data(args))
    # Deliberately not a chunk count. When nltk's tokenizer is unavailable --
    # which is the state of the Linux CI runners, see docs/faq.md -- chunking
    # catches the error and emits the whole document as a single chunk, so how
    # many chunks come back is a property of the environment, not of this app.
    assert texts
    assert texts[0]["text"].startswith("A source\nSource: https://a.io/x")
    assert all(text["metadata"]["url"] == "https://a.io/x" for text in texts)
