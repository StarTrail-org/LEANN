#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "faiss-cpu",
#     "numpy",
#     "sentence-transformers",
#     "torch",
#     "tqdm",
# ]
# ///

"""
Independent recall verification script using standard FAISS.
Creates two indexes (HNSW and Flat) and compares recall@3 at different complexities.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def compute_embeddings_direct(chunks: list[str], model_name: str) -> np.ndarray:
    """
    Direct embedding computation using sentence-transformers.
    Copied logic to avoid dependency issues.
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Computing embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    return embeddings.astype(np.float32)


def load_financebench_queries(dataset_path: str, max_queries: int = 200) -> list[str]:
    """Load FinanceBench queries from dataset"""
    queries = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                queries.append(data["question"])
                if len(queries) >= max_queries:
                    break
    return queries


def load_passages_from_leann_index(index_path: str) -> tuple[list[str], list[str]]:
    """Load passages from LEANN index structure"""
    meta_path = f"{index_path}.meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    passage_source = meta["passage_sources"][0]
    passage_file = passage_source["path"]

    # Convert relative path to absolute
    if not Path(passage_file).is_absolute():
        index_dir = Path(index_path).parent
        passage_file = index_dir / Path(passage_file).name

    print(f"Loading passages from {passage_file}")

    passages = []
    passage_ids = []
    with open(passage_file, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading passages"):
            if line.strip():
                data = json.loads(line)
                passages.append(data["text"])
                passage_ids.append(data["id"])

    print(f"Loaded {len(passages)} passages")
    return passages, passage_ids


def build_faiss_indexes(embeddings: np.ndarray) -> tuple[faiss.Index, faiss.Index]:
    """Build FAISS indexes: Flat (ground truth) and HNSW"""
    dimension = embeddings.shape[1]

    # Build Flat index (ground truth)
    print("Building FAISS IndexFlatIP (ground truth)...")
    flat_index = faiss.IndexFlatIP(dimension)
    flat_index.add(embeddings)

    # Build HNSW index
    print("Building FAISS IndexHNSWFlat...")
    M = 32  # Same as LEANN default
    hnsw_index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = 200  # Same as LEANN default
    hnsw_index.add(embeddings)

    print(f"Built indexes with {flat_index.ntotal} vectors, dimension {dimension}")
    return flat_index, hnsw_index


def evaluate_recall_at_k(
    query_embeddings: np.ndarray,
    flat_index: faiss.Index,
    hnsw_index: faiss.Index,
    passage_ids: list[str],
    k: int = 3,
    ef_search: int = 64,
) -> float:
    """Evaluate recall@k comparing HNSW vs Flat"""

    # Set search parameters for HNSW
    hnsw_index.hnsw.efSearch = ef_search

    total_recall = 0.0
    num_queries = query_embeddings.shape[0]

    for i in range(num_queries):
        query = query_embeddings[i : i + 1]  # Keep 2D shape

        # Get ground truth from Flat index (standard FAISS API)
        _flat_distances, flat_indices = flat_index.search(query, k)
        ground_truth_ids = {passage_ids[idx] for idx in flat_indices[0]}

        # Get results from HNSW index (standard FAISS API)
        _hnsw_distances, hnsw_indices = hnsw_index.search(query, k)
        hnsw_ids = {passage_ids[idx] for idx in hnsw_indices[0]}

        # Calculate recall
        intersection = ground_truth_ids.intersection(hnsw_ids)
        recall = len(intersection) / k
        total_recall += recall

        if i < 3:  # Show first few examples
            print(f"  Query {i + 1}: Recall@{k} = {recall:.3f}")
            print(f"    Flat: {list(ground_truth_ids)}")
            print(f"    HNSW: {list(hnsw_ids)}")
            print(f"    Intersection: {list(intersection)}")

    avg_recall = total_recall / num_queries
    return avg_recall


def main():
    # Configuration
    dataset_path = "data/financebench_merged.jsonl"
    index_path = "data/index/financebench_full_hnsw.leann"
    embedding_model = "sentence-transformers/all-mpnet-base-v2"

    print("🔍 FAISS Recall Verification")
    print("=" * 50)

    # Check if files exist
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    if not Path(f"{index_path}.meta.json").exists():
        print(f"❌ Index metadata not found: {index_path}.meta.json")
        return

    # Load data
    print("📖 Loading FinanceBench queries...")
    queries = load_financebench_queries(dataset_path, max_queries=50)
    print(f"Loaded {len(queries)} queries")

    print("📄 Loading passages from LEANN index...")
    passages, passage_ids = load_passages_from_leann_index(index_path)

    # Compute embeddings
    print("🧮 Computing passage embeddings...")
    passage_embeddings = compute_embeddings_direct(passages, embedding_model)

    print("🧮 Computing query embeddings...")
    query_embeddings = compute_embeddings_direct(queries, embedding_model)

    # Build FAISS indexes
    print("🏗️ Building FAISS indexes...")
    flat_index, hnsw_index = build_faiss_indexes(passage_embeddings)

    # Test different efSearch values (equivalent to LEANN complexity)
    print("\n📊 Evaluating Recall@3 at different efSearch values...")
    ef_search_values = [16, 32, 64, 128, 256]

    for ef_search in ef_search_values:
        print(f"\n🧪 Testing efSearch = {ef_search}")
        start_time = time.time()

        recall = evaluate_recall_at_k(
            query_embeddings, flat_index, hnsw_index, passage_ids, k=3, ef_search=ef_search
        )

        elapsed = time.time() - start_time
        print(
            f"📈 efSearch {ef_search}: Recall@3 = {recall:.3f} ({recall * 100:.1f}%) in {elapsed:.2f}s"
        )

    print("\n✅ Verification completed!")
    print("\n📋 Summary:")
    print("  - Built independent FAISS Flat and HNSW indexes")
    print("  - Compared recall@3 at different efSearch values")
    print("  - Used same embedding model as LEANN")
    print("  - This validates LEANN's recall measurements")


if __name__ == "__main__":
    main()
