# LEANN Metadata Filtering Usage Guide

## Overview

Leann possesses metadata filtering capabilities that allow you to filter search results based on arbitrary metadata fields set during chunking. This feature enables use cases like spoiler-free book search, document filtering by date/type, code search by file type, and potentially much more.

## Basic Usage

### Adding Metadata to Your Documents

When building your index, add metadata to each text chunk:

```python
from leann.api import LeannBuilder

builder = LeannBuilder("hnsw")

# Add text with metadata
builder.add_text(
    text="Chapter 1: Alice falls down the rabbit hole",
    metadata={
        "chapter": 1,
        "character": "Alice",
        "themes": ["adventure", "curiosity"],
        "word_count": 150
    }
)

builder.build_index("alice_in_wonderland_index")
```

### Searching with Metadata Filters

Use the `metadata_filters` parameter in search calls:

```python
from leann.api import LeannSearcher

searcher = LeannSearcher("alice_in_wonderland_index")

# Search with filters
results = searcher.search(
    query="What happens to Alice?",
    top_k=10,
    metadata_filters={
        "chapter": {"<=": 5},           # Only chapters 1-5
        "spoiler_level": {"!=": "high"} # No high spoilers
    }
)
```

## Filter Syntax

### Basic Structure

```python
metadata_filters = {
    "field_name": {"operator": value},
    "another_field": {"operator": value}
}
```

### Supported Operators

#### Comparison Operators
- `"=="`: Equal to
- `"!="`: Not equal to
- `"<"`: Less than
- `"<="`: Less than or equal
- `">"`: Greater than
- `">="`: Greater than or equal

```python
# Examples
{"chapter": {"==": 1}}           # Exactly chapter 1
{"page": {">": 100}}            # Pages after 100
{"rating": {">=": 4.0}}         # Rating 4.0 or higher
{"word_count": {"<": 500}}      # Short passages
```

Numeric values stored as strings are ordered numerically (for example,
`"10" > "2"`). Other strings, including ISO dates, retain lexical ordering.

#### Membership Operators
- `"in"`: Value is in list
- `"not_in"`: Value is not in list

```python
# Examples
{"character": {"in": ["Alice", "Bob"]}}      # Alice OR Bob
{"genre": {"not_in": ["horror", "thriller"]}} # Exclude genres
{"tags": {"in": ["fiction", "adventure"]}}   # Any of these tags
```

#### String Operators
- `"contains"`: String contains substring
- `"starts_with"`: String starts with prefix
- `"ends_with"`: String ends with suffix

```python
# Examples
{"title": {"contains": "alice"}}        # Title contains "alice"
{"filename": {"ends_with": ".py"}}      # Python files
{"author": {"starts_with": "Dr."}}      # Authors with "Dr." prefix
```

#### Boolean Operators
- `"is_true"`: Field is truthy
- `"is_false"`: Field is falsy

```python
# Examples
{"is_published": {"is_true": True}}     # Published content
{"is_draft": {"is_false": False}}       # Not drafts
```

### Multiple Operators on Same Field

You can apply multiple operators to the same field (AND logic):

```python
metadata_filters = {
    "word_count": {
        ">=": 100,    # At least 100 words
        "<=": 500     # At most 500 words
    }
}
```

### Compound Filters

Multiple fields are combined with AND logic:

```python
metadata_filters = {
    "chapter": {"<=": 10},              # Up to chapter 10
    "character": {"==": "Alice"},       # About Alice
    "spoiler_level": {"!=": "high"}     # No major spoilers
}
```

## Use Case Examples

### 1. Spoiler-Free Book Search

```python
# Reader has only read up to chapter 5
def search_spoiler_free(query, max_chapter):
    return searcher.search(
        query=query,
        metadata_filters={
            "chapter": {"<=": max_chapter},
            "spoiler_level": {"in": ["none", "low"]}
        }
    )

results = search_spoiler_free("What happens to Alice?", max_chapter=5)
```

### 2. Document Management by Date

```python
# Find recent documents
recent_docs = searcher.search(
    query="project updates",
    metadata_filters={
        "date": {">=": "2024-01-01"},
        "document_type": {"==": "report"}
    }
)
```

### 3. Code Search by File Type

```python
# Search only Python files
python_code = searcher.search(
    query="authentication function",
    metadata_filters={
        "file_extension": {"==": ".py"},
        "lines_of_code": {"<": 100}
    }
)
```

### 4. Content Filtering by Audience

```python
# Age-appropriate content
family_content = searcher.search(
    query="adventure stories",
    metadata_filters={
        "age_rating": {"in": ["G", "PG"]},
        "content_warnings": {"not_in": ["violence", "adult_themes"]}
    }
)
```

### 5. Multi-Book Series Management

```python
# Search across first 3 books only
early_series = searcher.search(
    query="character development",
    metadata_filters={
        "series": {"==": "Harry Potter"},
        "book_number": {"<=": 3}
    }
)
```

## Running the Example

You can see metadata filtering in action with our spoiler-free book RAG example:

```bash
# Don't forget to set up the environment
uv venv
source .venv/bin/activate

# Set your OpenAI API key (required for embeddings, but you can update the example locally and use ollama instead)
export OPENAI_API_KEY="your-api-key-here"

# Run the spoiler-free book RAG example
uv run examples/spoiler_free_book_rag.py
```

This example demonstrates:
- Building an index with metadata (chapter numbers, characters, themes, locations)
- Searching with filters to avoid spoilers (e.g., only show results up to chapter 5)
- Different scenarios for readers at various points in the book

The example uses Alice's Adventures in Wonderland as sample data and shows how you can search for information without revealing plot points from later chapters.

## Advanced Patterns

### Custom Chunking with metadata

```python
def chunk_book_with_metadata(book_text, book_info):
    chunks = []

    for chapter_num, chapter_text in parse_chapters(book_text):
        # Extract entities, themes, etc.
        characters = extract_characters(chapter_text)
        themes = classify_themes(chapter_text)
        spoiler_level = assess_spoiler_level(chapter_text, chapter_num)

        # Create chunks with rich metadata
        for paragraph in split_paragraphs(chapter_text):
            chunks.append({
                "text": paragraph,
                "metadata": {
                    "book_title": book_info["title"],
                    "chapter": chapter_num,
                    "characters": characters,
                    "themes": themes,
                    "spoiler_level": spoiler_level,
                    "word_count": len(paragraph.split()),
                    "reading_level": calculate_reading_level(paragraph)
                }
            })

    return chunks
```

## Performance Considerations

### Efficient Filtering Strategies

1. **Post-search filtering**: Applies filters after vector search, which should be efficient for typical result sets (10-100 results).

2. **Metadata design**: Keep metadata fields simple and avoid deeply nested structures.

### Best Practices

1. **Consistent metadata schema**: Use consistent field names and value types across your documents.

2. **Reasonable metadata size**: Keep metadata reasonably sized to avoid storage overhead.

3. **Type consistency**: Use consistent data types for the same fields (e.g., always integers for chapter numbers).

4. **Index multiple granularities**: Consider chunking at different levels (paragraph, section, chapter) with appropriate metadata.

### Adding Metadata to Existing Indices

To add metadata filtering to existing indices, you'll need to rebuild them with metadata:

```python
# Read existing passages and add metadata
def add_metadata_to_existing_chunks(chunks):
    for chunk in chunks:
        # Extract or assign metadata based on content
        chunk["metadata"] = extract_metadata(chunk["text"])
    return chunks

# Rebuild index with metadata
enhanced_chunks = add_metadata_to_existing_chunks(existing_chunks)
builder = LeannBuilder("hnsw")
for chunk in enhanced_chunks:
    builder.add_text(chunk["text"], chunk["metadata"])
builder.build_index("enhanced_index")
```
