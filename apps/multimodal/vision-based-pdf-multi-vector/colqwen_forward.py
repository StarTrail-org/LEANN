#!/usr/bin/env python3
"""Simple test script to test colqwen2 forward pass with a single image."""

import os
import sys
from pathlib import Path

# Add the current directory to path to import leann_multi_vector
sys.path.insert(0, str(Path(__file__).parent))

import torch
from leann_multi_vector import _embed_images, _ensure_repo_paths_importable, _load_colvision
from PIL import Image

# Ensure repo paths are importable
_ensure_repo_paths_importable(__file__)

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_test_image():
    """Create a simple test image."""
    # Create a simple RGB image (800x600)
    img = Image.new("RGB", (800, 600), color="white")
    return img


def load_test_image_from_file():
    """Try to load an image from the indexes directory if available."""
    # Try to find an existing image in the indexes directory
    indexes_dir = Path(__file__).parent / "indexes"

    # Look for images in common locations
    possible_paths = [
        indexes_dir / "vidore_fastplaid" / "images",
        indexes_dir / "colvision_large.leann.images",
        indexes_dir / "colvision.leann.images",
    ]

    for img_dir in possible_paths:
        if img_dir.exists():
            # Find first image file
            for ext in [".png", ".jpg", ".jpeg"]:
                for img_file in img_dir.glob(f"*{ext}"):
                    print(f"Loading test image from: {img_file}")
                    return Image.open(img_file)

    return None


def main():
    print("=" * 60)
    print("Testing ColQwen2 Forward Pass")
    print("=" * 60)

    # Step 1: Load or create test image
    print("\n[Step 1] Loading test image...")
    test_image = load_test_image_from_file()
    if test_image is None:
        print("No existing image found, creating a simple test image...")
        test_image = create_test_image()
    else:
        print(f"✓ Loaded image: {test_image.size} ({test_image.mode})")

    # Convert to RGB if needed
    if test_image.mode != "RGB":
        test_image = test_image.convert("RGB")
        print(f"✓ Converted to RGB: {test_image.size}")

    # Step 2: Load model
    print("\n[Step 2] Loading ColQwen2 model...")
    try:
        model_name, model, processor, device_str, _device, dtype = _load_colvision("colqwen2")
        print(f"✓ Model loaded: {model_name}")
        print(f"✓ Device: {device_str}, dtype: {dtype}")

        # Print model info
        if hasattr(model, "device"):
            print(f"✓ Model device: {model.device}")
        if hasattr(model, "dtype"):
            print(f"✓ Model dtype: {model.dtype}")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 3: Test forward pass
    print("\n[Step 3] Running forward pass...")
    try:
        # Use the _embed_images function which handles batching and forward pass
        images = [test_image]
        print(f"Processing {len(images)} image(s)...")

        doc_vecs = _embed_images(model, processor, images)

        print("✓ Forward pass completed!")
        print(f"✓ Number of embeddings: {len(doc_vecs)}")

        if len(doc_vecs) > 0:
            emb = doc_vecs[0]
            print(f"✓ Embedding shape: {emb.shape}")
            print(f"✓ Embedding dtype: {emb.dtype}")
            print("✓ Embedding stats:")
            print(f"    - Min: {emb.min().item():.4f}")
            print(f"    - Max: {emb.max().item():.4f}")
            print(f"    - Mean: {emb.mean().item():.4f}")
            print(f"    - Std: {emb.std().item():.4f}")

            # Check for NaN or Inf
            if torch.isnan(emb).any():
                print("⚠ Warning: Embedding contains NaN values!")
            if torch.isinf(emb).any():
                print("⚠ Warning: Embedding contains Inf values!")

    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
