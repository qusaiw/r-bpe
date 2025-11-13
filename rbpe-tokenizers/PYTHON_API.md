# R-BPE Python API Guide

This guide shows how to use the R-BPE tokenizer from Python using the Rust-based bindings.

## Installation

```bash
# From the rbpe-tokenizers directory
maturin develop --release
```

This builds the Rust code and installs the `rbpe_tokenizers` Python package in your current environment.

## Quick Start

### Method 1: `from_pretrained()` (Recommended)

The simplest way to load a tokenizer, matching the original Python API:

```python
import rbpe_tokenizers

# Load from a directory (expects standard structure)
tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")

# Encode text
text = "Hello مرحبا World!"
ids = tokenizer.encode(text)
print(ids)  # [9906, 220, 122627, 5821, 220, 10343, 0]

# Decode back
decoded = tokenizer.decode(ids)
print(decoded)  # "Hello مرحبا World!"
```

**Expected Directory Structure:**
```
rbpe_tokenizer/
├── new_tokenizer/
│   └── tokenizer.json
├── old_tokenizer/
│   └── tokenizer.json
└── metadata/
    ├── new_to_old_map.json
    ├── old_to_new_map.json
    └── replacement_character_map.json
```

### Method 2: `from_files()` (Custom Paths)

For custom directory structures or fine-grained control:

```python
import rbpe_tokenizers

tokenizer = rbpe_tokenizers.RBPETokenizer.from_files(
    new_tokenizer_path="path/to/new_tokenizer/tokenizer.json",
    old_tokenizer_path="path/to/old_tokenizer/tokenizer.json",
    new_to_old_map_path="path/to/new_to_old_map.json",
    old_to_new_map_path="path/to/old_to_new_map.json",
    replacement_char_map_path="path/to/replacement_character_map.json",  # optional
    target_language="arabic"  # default
)
```

## API Reference

### Loading Methods

#### `RBPETokenizer.from_pretrained(pretrained_path, target_language="arabic")`

Load tokenizer from a directory with standard structure.

**Parameters:**
- `pretrained_path` (str): Path to directory containing tokenizer files
- `target_language` (str, optional): Target language for optimization. Default: "arabic"

**Returns:** `RBPETokenizer` instance

**Raises:**
- `FileNotFoundError`: If required files are missing
- `ValueError`: If unsupported language is specified

---

#### `RBPETokenizer.from_files(...)`

Load tokenizer from explicit file paths.

**Parameters:**
- `new_tokenizer_path` (str): Path to new tokenizer (Arabic-optimized)
- `old_tokenizer_path` (str): Path to old tokenizer (base model)
- `new_to_old_map_path` (str): Path to new→old ID mapping
- `old_to_new_map_path` (str): Path to old→new ID mapping
- `replacement_char_map_path` (str, optional): Path to replacement character map
- `target_language` (str, optional): Target language. Default: "arabic"

**Returns:** `RBPETokenizer` instance

### Encoding/Decoding Methods

#### `encode(text, add_special_tokens=False)`

Encode text to token IDs.

**Parameters:**
- `text` (str): Input text to encode
- `add_special_tokens` (bool, optional): Whether to add special tokens. Default: False

**Returns:** `list[int]` - List of token IDs

**Example:**
```python
ids = tokenizer.encode("Hello World")
# [9906, 4435]
```

---

#### `decode(ids, skip_special_tokens=True)`

Decode token IDs back to text (basic decoder).

**Parameters:**
- `ids` (list[int]): List of token IDs
- `skip_special_tokens` (bool, optional): Whether to skip special tokens. Default: True

**Returns:** `str` - Decoded text

**Example:**
```python
text = tokenizer.decode([9906, 4435])
# "Hello World"
```

---

#### `decode_advanced(ids, skip_special_tokens=True)`

Decode token IDs with advanced replacement character handling.

This decoder uses a sliding window approach to reconstruct UTF-8 sequences
that may have been split across multiple tokens.

**Parameters:**
- `ids` (list[int]): List of token IDs
- `skip_special_tokens` (bool, optional): Whether to skip special tokens. Default: True

**Returns:** `str` - Decoded text

**Example:**
```python
text = tokenizer.decode_advanced([9906, 4435])
# "Hello World"
```

---

#### `encode_batch(texts, add_special_tokens=False)`

Encode multiple texts in batch.

**Parameters:**
- `texts` (list[str]): List of texts to encode
- `add_special_tokens` (bool, optional): Whether to add special tokens. Default: False

**Returns:** `list[list[int]]` - List of token ID lists

**Example:**
```python
batch_ids = tokenizer.encode_batch(["Hello", "World", "مرحبا"])
# [[9906], [10343], [122627, 5821]]
```

---

#### `decode_batch(ids_batch, skip_special_tokens=True)`

Decode multiple token ID sequences in batch.

**Parameters:**
- `ids_batch` (list[list[int]]): List of token ID lists
- `skip_special_tokens` (bool, optional): Whether to skip special tokens. Default: True

**Returns:** `list[str]` - List of decoded texts

**Example:**
```python
texts = tokenizer.decode_batch([[9906], [10343], [122627, 5821]])
# ["Hello", "World", "مرحبا"]
```

## What Makes R-BPE Special?

R-BPE (Reusable BPE) is not just a regular tokenizer:

### 1. Dual Tokenizer System
- Uses **TWO BPE tokenizers** internally
- One optimized for Arabic (new tokenizer)
- One for other languages (old/base tokenizer)

### 2. Language-Aware Routing
- Automatically detects language segments
- Routes Arabic text to the new tokenizer
- Routes other text to the old tokenizer
- Maintains compatibility across languages

### 3. Vocabulary Mapping
- Maps token IDs from new tokenizer to old tokenizer space
- Ensures compatibility with base model vocabulary
- Optimizes for target language without breaking existing models

### 4. Advanced Decoding
- Handles replacement characters with sliding window
- Reconstructs UTF-8 sequences split across tokens
- Better handling of complex scripts

### 5. High Performance
- Implemented in Rust for maximum speed
- 10-100x faster than pure Python implementations
- Minimal overhead from Python/Rust boundary

## Performance

Example benchmark (1000 encode/decode operations):

```
Text length: 240 chars (mixed English/Arabic)
Total time:  0.114s
Ops/sec:    8,753
Time/op:    114.3 µs
```

## Complete Example

```python
import rbpe_tokenizers

# Load tokenizer
tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")

# Single text
text = "This is a test هذا اختبار"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
print(f"Original: {text}")
print(f"IDs: {ids}")
print(f"Decoded: {decoded}")

# Batch processing
texts = [
    "Hello مرحبا",
    "World عالم",
    "Mixed text نص مختلط"
]

# Encode all
batch_ids = tokenizer.encode_batch(texts)
for text, ids in zip(texts, batch_ids):
    print(f"{text} -> {len(ids)} tokens")

# Decode all
decoded = tokenizer.decode_batch(batch_ids)
for original, decoded_text in zip(texts, decoded):
    match = "✓" if original == decoded_text else "✗"
    print(f"{match} {decoded_text}")
```

## Error Handling

```python
import rbpe_tokenizers

try:
    # This will fail if directory doesn't exist
    tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("nonexistent")
except FileNotFoundError as e:
    print(f"Error: {e}")

try:
    # This will fail for unsupported languages
    tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained(
        "rbpe_tokenizer",
        target_language="french"  # Not supported yet
    )
except ValueError as e:
    print(f"Error: {e}")
```

## Comparison with Original Python Implementation

### Similarities
- Same `from_pretrained()` API
- Same directory structure expectations
- Compatible token IDs and outputs

### Differences
- **Performance**: Rust implementation is 10-100x faster
- **Target Language**: Must specify explicitly (defaults to "arabic")
- **API Simplification**: Focused on core tokenization features

### Migration Guide

If you're migrating from the original Python implementation:

**Before (Python):**
```python
from rbpe import RBPETokenizer

tokenizer = RBPETokenizer.from_pretrained("rbpe_tokenizer")
ids = tokenizer.encode("Hello World")
text = tokenizer.decode(ids)
```

**After (Rust):**
```python
import rbpe_tokenizers

tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")
ids = tokenizer.encode("Hello World")
text = tokenizer.decode(ids)
```

The API is almost identical! The main difference is the import statement.

## Supported Languages

Currently supported target languages:
- `"arabic"` (default)

More languages can be added by extending the Unicode range checker in the Rust code.

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Rust API guide
- [examples/python_demo.py](examples/python_demo.py) - Full Python example
- [test_python_bindings.py](../test_python_bindings.py) - Comprehensive test suite
