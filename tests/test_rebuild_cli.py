import asyncio
import json
import pickle
from pathlib import Path

import pytest
from leann.cli import LeannCLI


def test_reconstruct_build_args_replays_stored_build_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    docs_file = docs_root / "only.md"
    docs_file.write_text("hello", encoding="utf-8")

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    meta_path = index_dir / "documents.leann.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "backend_name": "hnsw",
                "embedding_model": "text-embedding-3-small",
                "embedding_mode": "openai",
                "backend_kwargs": {
                    "graph_degree": 32,
                    "complexity": 64,
                    "num_threads": 1,
                    "is_compact": False,
                    "is_recompute": True,
                },
                "embedding_options": {
                    "base_url": "https://metadata.example/v1",
                    "api_key": "secret-should-not-be-replayed",
                    "build_prompt_template": "metadata passage: ",
                    "query_prompt_template": "metadata query: ",
                },
            }
        ),
        encoding="utf-8",
    )
    sync_config = {
        "roots": [str(docs_root)],
        "include_extensions": [".md"],
        "ignore_patterns": ["**/.*"],
        "build_config": {
            "docs": [str(docs_file.resolve())],
            "file_types": ".md,.txt",
            "include_hidden": True,
            "doc_chunk_size": 384,
            "doc_chunk_overlap": 96,
            "code_chunk_size": 640,
            "code_chunk_overlap": 80,
            "use_ast_chunking": True,
            "ast_chunk_size": 420,
            "ast_chunk_overlap": 72,
            "ast_fallback_traditional": True,
            "graph_degree": 48,
            "complexity": 96,
            "num_threads": 4,
            "compact": True,
            "recompute": False,
            "embedding_api_base": "https://stored.example/v1",
            "embedding_prompt_template": "stored passage: ",
            "query_prompt_template": "stored query: ",
        },
    }
    (index_dir / "sync_roots.json").write_text(json.dumps(sync_config), encoding="utf-8")

    reconstructed = cli._reconstruct_build_args("sample", force=True)

    assert reconstructed is not None
    assert "--embedding-api-key" not in reconstructed
    parsed = cli.create_parser().parse_args(reconstructed)
    assert parsed.docs == [str(docs_file.resolve())]
    assert parsed.file_types == ".md,.txt"
    assert parsed.include_hidden is True
    assert parsed.doc_chunk_size == 384
    assert parsed.doc_chunk_overlap == 96
    assert parsed.code_chunk_size == 640
    assert parsed.code_chunk_overlap == 80
    assert parsed.use_ast_chunking is True
    assert parsed.ast_chunk_size == 420
    assert parsed.ast_chunk_overlap == 72
    assert parsed.graph_degree == 48
    assert parsed.complexity == 96
    assert parsed.num_threads == 4
    assert parsed.compact is True
    assert parsed.recompute is False
    assert parsed.embedding_api_base == "https://stored.example/v1"
    assert parsed.embedding_prompt_template == "stored passage: "
    assert parsed.query_prompt_template == "stored query: "
    assert parsed.force is True


def test_reconstruct_build_args_falls_back_to_metadata_and_sync_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "legacy"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text(
        json.dumps(
            {
                "backend_name": "hnsw",
                "embedding_model": "facebook/contriever",
                "embedding_mode": "ollama",
                "backend_kwargs": {
                    "graph_degree": 40,
                    "complexity": 88,
                    "num_threads": 3,
                    "is_compact": True,
                    "is_recompute": False,
                },
                "embedding_options": {
                    "host": "http://ollama.example:11434",
                    "prompt_template": "passage: ",
                },
            }
        ),
        encoding="utf-8",
    )
    (index_dir / "sync_roots.json").write_text(
        json.dumps(
            {
                "roots": [str(docs_root)],
                "include_extensions": [".md", ".py"],
                "ignore_patterns": None,
            }
        ),
        encoding="utf-8",
    )

    reconstructed = cli._reconstruct_build_args("legacy")

    assert reconstructed is not None
    parsed = cli.create_parser().parse_args(reconstructed)
    assert parsed.docs == [str(docs_root)]
    assert parsed.file_types == ".md,.py"
    assert parsed.include_hidden is True
    assert parsed.graph_degree == 40
    assert parsed.complexity == 88
    assert parsed.num_threads == 3
    assert parsed.compact is True
    assert parsed.recompute is False
    assert parsed.embedding_host == "http://ollama.example:11434"
    assert parsed.embedding_prompt_template == "passage: "


def test_reconstruct_build_args_accepts_explicit_files_only(tmp_path, monkeypatch):
    # `leann build --docs a.md b.md` records the files under "files" and leaves "roots" empty.
    monkeypatch.chdir(tmp_path)
    doc = tmp_path / "notes.md"
    doc.write_text("note", encoding="utf-8")

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "files-only"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text(
        json.dumps({"backend_name": "hnsw"}), encoding="utf-8"
    )
    (index_dir / "sync_roots.json").write_text(
        json.dumps({"directories": [], "files": [str(doc)], "roots": []}),
        encoding="utf-8",
    )

    reconstructed = cli._reconstruct_build_args("files-only")

    assert reconstructed is not None
    assert cli.create_parser().parse_args(reconstructed).docs == [str(doc)]


def test_successful_full_build_persists_rebuild_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    docs_file = docs_root / "only.py"
    docs_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    cli = LeannCLI()
    args = cli.create_parser().parse_args(
        [
            "build",
            "sample",
            "--docs",
            str(docs_file),
            "--backend-name",
            "hnsw",
            "--file-types",
            ".py",
            "--include-hidden",
            "--doc-chunk-size",
            "10",
            "--doc-chunk-overlap",
            "12",
            "--code-chunk-size",
            "20",
            "--code-chunk-overlap",
            "25",
            "--use-ast-chunking",
            "--ast-chunk-size",
            "123",
            "--ast-chunk-overlap",
            "17",
            "--graph-degree",
            "44",
            "--complexity",
            "77",
            "--num-threads",
            "2",
            "--compact",
            "--no-recompute",
        ]
    )

    class FakeSynchronizer:
        def detect_changes(self):
            return {str(docs_file.resolve())}, set(), set()

        def create_snapshot(self):
            pass

    class SuccessfulBuilder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add_text(self, _text, metadata=None):
            pass

        def build_index(self, index_path):
            target_dir = Path(index_path).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "documents.leann.meta.json").write_text(
                json.dumps(
                    {
                        "backend_name": self.kwargs["backend_name"],
                        "embedding_model": self.kwargs["embedding_model"],
                        "embedding_mode": self.kwargs["embedding_mode"],
                        "backend_kwargs": {
                            "graph_degree": self.kwargs["graph_degree"],
                            "complexity": self.kwargs["complexity"],
                            "num_threads": self.kwargs["num_threads"],
                            "is_compact": self.kwargs["is_compact"],
                            "is_recompute": self.kwargs["is_recompute"],
                        },
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(cli, "_build_synchronizers", lambda *_args, **_kwargs: [FakeSynchronizer()])
    monkeypatch.setattr(
        cli,
        "load_documents",
        lambda *_args, **_kwargs: [{"text": "rebuilt", "metadata": {"file_path": str(docs_file)}}],
    )
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)
    monkeypatch.setattr("leann.cli.LeannBuilder", SuccessfulBuilder)

    asyncio.run(cli.build_index(args))

    sync_config = json.loads(
        (tmp_path / ".leann" / "indexes" / "sample" / "sync_roots.json").read_text(encoding="utf-8")
    )
    build_config = sync_config["build_config"]
    assert build_config["docs"] == [str(docs_file.resolve())]
    assert build_config["file_types"] == ".py"
    assert build_config["include_hidden"] is True
    assert build_config["doc_chunk_size"] == 10
    assert build_config["doc_chunk_overlap"] == 9
    assert build_config["code_chunk_size"] == 20
    assert build_config["code_chunk_overlap"] == 19
    assert build_config["use_ast_chunking"] is True
    assert build_config["ast_chunk_size"] == 123
    assert build_config["ast_chunk_overlap"] == 17
    assert build_config["graph_degree"] == 44
    assert build_config["complexity"] == 77
    assert build_config["num_threads"] == 2
    assert build_config["compact"] is True
    assert build_config["recompute"] is False


def test_full_rebuild_failure_preserves_existing_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    docs_file = docs_root / "only.md"
    docs_file.write_text("hello", encoding="utf-8")

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    meta_path = index_dir / "documents.leann.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "backend_name": "hnsw",
                "embedding_model": "facebook/contriever",
                "embedding_mode": "sentence-transformers",
                "backend_kwargs": {"is_compact": False, "is_recompute": True},
            }
        ),
        encoding="utf-8",
    )
    passages_file = index_dir / "documents.leann.passages.jsonl"
    passages_file.write_text(
        json.dumps({"id": "old", "text": "old text", "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    offset_file = index_dir / "documents.leann.passages.idx"
    with open(offset_file, "wb") as f:
        pickle.dump({"old": 0}, f)
    original_artifacts = {
        path: path.read_bytes() for path in (meta_path, passages_file, offset_file)
    }
    args = cli.create_parser().parse_args(
        [
            "build",
            "sample",
            "--docs",
            str(docs_file),
            "--backend-name",
            "hnsw",
            "--force",
        ]
    )
    built_paths: list[str] = []

    class FakeSynchronizer:
        def create_snapshot(self):
            pass

    class FailingBuilder:
        def __init__(self, **_kwargs):
            pass

        def add_text(self, _text, metadata=None):
            pass

        def build_index(self, index_path):
            built_paths.append(index_path)
            target_dir = Path(index_path).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "documents.leann.passages.jsonl").write_text(
                json.dumps({"id": "new", "text": "new text", "metadata": {}}) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError("simulated build failure")

    monkeypatch.setattr(cli, "_build_synchronizers", lambda *_args, **_kwargs: [FakeSynchronizer()])
    monkeypatch.setattr(
        cli,
        "load_documents",
        lambda *_args, **_kwargs: [{"text": "rebuilt", "metadata": {"file_path": str(docs_file)}}],
    )
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)
    monkeypatch.setattr("leann.cli.LeannBuilder", FailingBuilder)

    with pytest.raises(RuntimeError, match="simulated build failure"):
        asyncio.run(cli.build_index(args))

    assert built_paths
    assert built_paths[0] != str(index_dir / "documents.leann")
    assert {
        path: path.read_bytes() for path in (meta_path, passages_file, offset_file)
    } == original_artifacts
    assert not list(index_dir.parent.glob(".sample.rebuild-*"))
