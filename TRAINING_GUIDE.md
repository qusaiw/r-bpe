# R-BPE Training Guide

## Current Status

The codebase has been refactored to separate training (Python) from runtime (Rust).

### What Changed

**BEFORE** (Old Architecture):
- Training created tokenizer using `DynamicTokenizer` (Python runtime)
- `mapping_tokenizer.py` was used at both training AND runtime
- Slower performance (pure Python)

**AFTER** (New Architecture):
- Training creates tokenizer files + copies Rust wrapper
- `mapping_tokenizer.py` is ONLY used during training to create mapping files
- Runtime uses Rust implementation (11x faster)

## Training a New Tokenizer

### Option 1: Using Config File

```bash
# Create config.yaml
cat > my_config.yaml << EOL
model_id: "meta-llama/Llama-3.1-8B"
training_data_dir: ./data/my_training_data
target_language_scripts:
  - arabic
preserved_languages_scripts:
  - latin
  - greek
min_reusable_count: 20000
clean_data: true
apply_rbpe_arabic_norm: true
hf_token: YOUR_HF_TOKEN_HERE
EOL

# Train tokenizer
rbpe create-tokenizer --config my_config.yaml --output_dir ./my_new_tokenizer
```

### Option 2: Using Python API

```python
from rbpe import RBPETokenizer

# Create factory
factory = RBPETokenizer(
    model_id="meta-llama/Llama-3.1-8B",
    training_data_dir="./data/my_training_data",
    target_language_scripts=["arabic"],
    preserved_languages_scripts=["latin", "greek"],
    min_reusable_count=20000,
    clean_data=True,
    hf_token="YOUR_HF_TOKEN_HERE",
    apply_rbpe_arabic_norm=True
)

# Train and save
tokenizer = factory.prepare()
tokenizer.save_pretrained("./my_new_tokenizer")
```

### What Gets Created

After training, you'll have:

```
my_new_tokenizer/
├── new_tokenizer/
│   ├── tokenizer.json           # Target language tokenizer
│   └── special_tokens_map.json
├── old_tokenizer/
│   ├── tokenizer.json           # Base model tokenizer  
│   └── special_tokens_map.json
├── metadata/
│   ├── new_to_old_map.json      # ID mappings (new → old)
│   ├── old_to_new_map.json      # ID mappings (old → new)
│   ├── replacement_character_map.json
│   ├── token_id_language_map.json
│   └── vocabulary_languages.txt
├── tokenization_rbpe.py         # Rust wrapper (auto-copied)
├── tokenizer_config.json        # HuggingFace config
└── special_tokens_map.json
```

## Loading the Trained Tokenizer

### Using AutoTokenizer (Recommended)

```python
from transformers import AutoTokenizer

# Load with Rust backend
tokenizer = AutoTokenizer.from_pretrained(
    "./my_new_tokenizer",
    trust_remote_code=True  # Required to load tokenization_rbpe.py
)

# Use like any HF tokenizer
ids = tokenizer.encode("Hello مرحبا World")
text = tokenizer.decode(ids, skip_special_tokens=True)

# Batch processing
batch = tokenizer(
    ["Hello", "مرحبا", "World"],
    padding=True,
    return_tensors="pt"
)
```

### Using Direct Rust API (Maximum Performance)

```python
import rbpe_tokenizers

# Load directly
tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("./my_new_tokenizer")

# Encode/decode
ids = tokenizer.encode("Hello مرحبا", add_special_tokens=False)
text = tokenizer.decode(ids, skip_special_tokens=True)

# Batch operations
batch_ids = tokenizer.encode_batch(["Hello", "مرحبا", "World"])
batch_texts = tokenizer.decode_batch(batch_ids)
```

## Migrating Old Tokenizers

If you have a tokenizer trained with the OLD code (before the refactoring):

### Check if Your Tokenizer is Old

```python
import json

with open("path/to/tokenizer_config.json") as f:
    config = json.load(f)
    
# Old tokenizer has these keys:
is_old = "custom_tokenizer_config" in config and "mapping_tokenizer" in config
print(f"Is old format: {is_old}")
```

### Migrate to New Format

Unfortunately, you need to **retrain** the tokenizer with the new code. The old format used `DynamicTokenizer` which is no longer supported.

```bash
# Retrain with same config
rbpe create-tokenizer --config original_config.yaml --output_dir ./new_format_tokenizer
```

## Understanding the Files

### Required for Rust Runtime

These files MUST exist for the Rust tokenizer to work:

- `new_tokenizer/tokenizer.json` - Target language BPE model
- `old_tokenizer/tokenizer.json` - Base model BPE  
- `metadata/new_to_old_map.json` - ID mapping
- `metadata/old_to_new_map.json` - Reverse ID mapping
- `tokenization_rbpe.py` - HuggingFace wrapper

### Optional Metadata

These are for reference/debugging only:

- `metadata/replacement_character_map.json` - Character replacements
- `metadata/token_id_language_map.json` - Token classifications
- `metadata/vocabulary_languages.txt` - Language statistics

## Troubleshooting

### "Rust R-BPE tokenizer not available"

```bash
# Build Rust bindings
cd rbpe-tokenizers
maturin develop --release
```

### "trust_remote_code required"

Always use `trust_remote_code=True` when loading:

```python
tokenizer = AutoTokenizer.from_pretrained("path", trust_remote_code=True)
```

### Training Fails with Import Error

Make sure both packages are installed:

```bash
# Install Python package
pip install -e .

# Build Rust bindings
cd rbpe-tokenizers
maturin develop --release
```

### Tokenizer Loads but Uses Old Python Code

Check if `tokenization_rbpe.py` exists in the tokenizer directory:

```bash
ls -la my_tokenizer/tokenization_rbpe.py
```

If missing, the tokenizer was trained with old code and needs to be retrained.

## Performance Comparison

| Operation | Old (Python) | New (Rust) | Speedup |
|-----------|--------------|------------|---------|
| Single encode | ~1ms | ~90µs | 11x |
| Batch (100) | ~100ms | ~9ms | 11x |
| Decode | ~1ms | ~90µs | 11x |

## Next Steps

1. Train a new tokenizer with the updated code
2. Test with your use case
3. Compare performance with old tokenizer
4. Deploy with confidence!
