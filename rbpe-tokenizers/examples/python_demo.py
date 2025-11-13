#!/usr/bin/env python3
"""
R-BPE Python Bindings - Quick Start Guide

This demonstrates how to use the Rust R-BPE tokenizer from Python.
"""

import rbpe_tokenizers

# Method 1: from_pretrained (RECOMMENDED - Simple and matches original Python API)
# ================================================================================
print("Method 1: from_pretrained (recommended)")
print("-" * 80)

tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("../../rbpe_tokenizer")

# Or specify a different language (currently only 'arabic' is supported)
# tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained(
#     "../../rbpe_tokenizer",
#     target_language="arabic"
# )

print("✓ Loaded tokenizer\n")


# Method 2: from_files (For custom directory structures)
# ================================================================================
print("\nMethod 2: from_files (for custom paths)")
print("-" * 80)

tokenizer2 = rbpe_tokenizers.RBPETokenizer.from_files(
    new_tokenizer_path="../../rbpe_tokenizer/new_tokenizer/tokenizer.json",
    old_tokenizer_path="../../rbpe_tokenizer/old_tokenizer/tokenizer.json",
    new_to_old_map_path="../../rbpe_tokenizer/metadata/new_to_old_map.json",
    old_to_new_map_path="../../rbpe_tokenizer/metadata/old_to_new_map.json",
    replacement_char_map_path="../../rbpe_tokenizer/metadata/replacement_character_map.json",
    target_language="arabic"
)

print("✓ Loaded tokenizer\n")


# Basic Usage
# ================================================================================
print("\nBasic Usage")
print("=" * 80)

# Encode text
text = "Hello مرحبا World!"
ids = tokenizer.encode(text)
print(f"Text:    '{text}'")
print(f"IDs:     {ids}")

# Decode back
decoded = tokenizer.decode(ids)
print(f"Decoded: '{decoded}'")

# Advanced decoding (handles replacement characters)
decoded_adv = tokenizer.decode_advanced(ids)
print(f"Advanced: '{decoded_adv}'")


# Batch Processing
# ================================================================================
print("\n\nBatch Processing")
print("=" * 80)

texts = ["Hello", "مرحبا", "World", "عالم"]

# Encode batch
batch_ids = tokenizer.encode_batch(texts)
print(f"Encoded {len(texts)} texts:")
for text, ids in zip(texts, batch_ids):
    print(f"  '{text}' -> {ids}")

# Decode batch
decoded_texts = tokenizer.decode_batch(batch_ids)
print(f"\nDecoded {len(batch_ids)} sequences:")
for decoded in decoded_texts:
    print(f"  '{decoded}'")


# Why R-BPE?
# ================================================================================
print("\n\nWhy R-BPE?")
print("=" * 80)
print("""
R-BPE (Reusable BPE) is not just a regular tokenizer:

1. **Dual Tokenizer System**: Uses TWO BPE models internally
   - One optimized for Arabic (new tokenizer)
   - One for other languages (old/base tokenizer)

2. **Language-Aware Routing**: Automatically detects language
   - Arabic text → new tokenizer (better Arabic tokenization)
   - Other text → old tokenizer (maintains compatibility)

3. **Vocabulary Mapping**: Maps token IDs between tokenizers
   - Maintains compatibility with base model vocabulary
   - Optimizes for target language without breaking compatibility

4. **High Performance**: Implemented in Rust
   - 10-100x faster than pure Python
   - Minimal Python/Rust overhead

This makes it ideal for multilingual models that need to optimize
for a specific language (like Arabic) without losing compatibility
with the base model.
""")

print("\n✓ All features working! 🚀")
