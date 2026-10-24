"""Regression tests for comparisons of metadata stored as strings."""

from leann.metadata_filter import MetadataFilterEngine


def test_numeric_strings_compare_by_value():
    results = [
        {"id": "ten", "metadata": {"chapter": "10"}},
        {"id": "two", "metadata": {"chapter": "2"}},
        {"id": "negative", "metadata": {"chapter": "-3"}},
    ]

    matched = MetadataFilterEngine().apply_filters(results, {"chapter": {">": "2"}})

    assert [result["id"] for result in matched] == ["ten"]


def test_non_numeric_strings_keep_lexical_comparison():
    results = [
        {"id": "early", "metadata": {"date": "2024-01-01"}},
        {"id": "late", "metadata": {"date": "2025-01-01"}},
    ]

    matched = MetadataFilterEngine().apply_filters(results, {"date": {">": "2024-12-31"}})

    assert [result["id"] for result in matched] == ["late"]
