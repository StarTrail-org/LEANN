import pytest
from leann.api import Fts5BM25Index


def test_fts5_bm25_matches_chinese_substrings_after_reopen(tmp_path):
    db_path = tmp_path / "passages.sqlite"
    index = Fts5BM25Index(str(db_path))
    index.fit(
        [
            {"id": "database", "text": "数据库检索系统"},
            {"id": "image", "text": "图像分类系统"},
        ]
    )
    index.close()

    reopened = Fts5BM25Index(str(db_path))
    try:
        assert [result.id for result in reopened.search("数据库")] == ["database"]
    finally:
        reopened.close()


def test_fts5_bm25_keeps_legacy_database_query_format(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    index = Fts5BM25Index(str(db_path), cjk_ngrams=False)
    index.fit([{"id": "database", "text": "database retrieval"}])
    index.close()

    reopened = Fts5BM25Index(str(db_path))
    try:
        assert [result.id for result in reopened.search("database")] == ["database"]
    finally:
        reopened.close()


@pytest.mark.parametrize("cjk_text", ["数据库", "データベース", "데이터베이스"])
@pytest.mark.parametrize("query", ["Python", "SQL"])
def test_fts5_bm25_preserves_words_adjacent_to_cjk(tmp_path, cjk_text, query):
    db_path = tmp_path / "mixed.sqlite"
    index = Fts5BM25Index(str(db_path))
    index.fit(
        [
            {"id": "mixed", "text": f"Python{cjk_text}SQL"},
            {"id": "unrelated", "text": "unrelated document"},
        ]
    )
    index.close()

    reopened = Fts5BM25Index(str(db_path))
    try:
        assert [result.id for result in reopened.search(query)] == ["mixed"]
    finally:
        reopened.close()


@pytest.mark.parametrize("query", ["Python数据库", "数据库Python", "Python数据库SQL", "2026数据库"])
def test_fts5_bm25_splits_mixed_script_query_terms(tmp_path, query):
    db_path = tmp_path / "queries.sqlite"
    index = Fts5BM25Index(str(db_path))
    index.fit(
        [
            {"id": "database", "text": "数据库检索系统"},
            {"id": "partial", "text": "数据分析"},
            {"id": "unrelated", "text": "image classification"},
        ]
    )
    index.close()

    reopened = Fts5BM25Index(str(db_path))
    try:
        # Match the CJK term independently of the Latin/number terms, while
        # still requiring all its bigrams (the partial match must be excluded).
        assert [result.id for result in reopened.search(query)] == ["database"]
    finally:
        reopened.close()
