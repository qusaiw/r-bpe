# R-BPE Architecture

## Overview

R-BPE uses a **two-phase architecture** that separates training (Python) from runtime (Rust) for optimal performance and maintainability.

## Phase 1: Training (Python)

The training phase is implemented in Python (`src/rbpe/`) and handles:

1. **Token Classification** (`token_classifier.py`)
   - Analyzes base tokenizer vocabulary
   - Identifies reusable tokens by language/script
   - Determines which tokens can be repurposed for target language

2. **Data Cleaning** (`data_cleaner.py`)
   - Removes non-target language content from training data
   - Ensures new tokenizer trains only on target language

3. **BPE Training** (`bpe_tokenizer_trainer.py`)
   - Trains new BPE tokenizer on cleaned data
   - Creates vocabulary optimized for target language

4. **Mapping Creation** (`mapping_tokenizer.py`)
   - Creates bidirectional ID mappings (new ↔ old)
   - Generates replacement character map
   - Used ONLY during training, not at runtime

5. **Saving** (`rbpe_tokenizer.py`)
   - Saves both tokenizers + metadata
   - Copies `tokenization_rbpe.py` wrapper for HuggingFace compatibility
   - Creates `tokenizer_config.json` with AutoTokenizer integration

## Phase 2: Runtime (Rust)

The runtime phase is implemented in Rust (`rbpe-tokenizers/`) for maximum performance:

### Core Components

1. **Fast Tokenizer** (`fast_tokenizer.rs`)
   - Main entry point for tokenization
   - Loads both tokenizers and mappings
   - Orchestrates the entire tokenization pipeline

2. **Pre-Tokenizer** (`pretokenizer.rs`)
   - Language-aware text splitting
   - Unicode range checking for target language detection
   - Splits text into segments by language

3. **Model** (`model.rs`)
   - Dual tokenizer system (new + old)
   - Routes segments to appropriate tokenizer
   - Applies ID mappings to maintain vocabulary space

4. **Decoder** (`decoder.rs`)
   - Basic decoding: Maps IDs back to text
   - Advanced decoding: Handles replacement characters with sliding window
   - Reconstructs UTF-8 sequences split across tokens

5. **Normalizer** (`normalizer.rs`)
   - Optional Unicode normalization
   - Arabic-specific normalizations (optional)

6. **Python Bindings** (`python_bindings.rs`)
   - PyO3-based bindings for Python access
   - Exposes RBPETokenizer class to Python
   - High-performance bridge with minimal overhead

### HuggingFace Integration

**tokenization_rbpe.py** (`src/rbpe/tokenization_rbpe.py`)
- HuggingFace PreTrainedTokenizer wrapper
- Delegates all operations to Rust backend
- Enables `AutoTokenizer.from_pretrained()` usage
- Full compatibility with HF ecosystem (Trainer, pipelines, etc.)

## Data Flow

### Training Flow
```
User Config
    ↓
RBPETokenizer (Factory)
    ↓
TokenClassifier → Analyze base vocabulary
    ↓
DataCleaner → Clean training data
    ↓
BPETokenizerTrainer → Train new tokenizer
    ↓
MappingTokenizer → Create ID mappings
    ↓
Save Directory Structure:
    new_tokenizer/
    old_tokenizer/
    metadata/
    tokenization_rbpe.py
    tokenizer_config.json
```

### Runtime Flow
```
Input Text
    ↓
[Rust RBPEFastTokenizer]
    ↓
Normalizer (optional)
    ↓
PreTokenizer → Split by language
    ↓
For each segment:
    ├─ Target language? → New Tokenizer → Map to old vocab space
    └─ Other language?  → Old Tokenizer → Use directly
    ↓
Merge token IDs
    ↓
Return: List[u32]
```

### Loading Flow
```
AutoTokenizer.from_pretrained("path", trust_remote_code=True)
    ↓
Loads tokenization_rbpe.py
    ↓
RBPETokenizer.__init__
    ↓
Calls rbpe_tokenizers.RBPETokenizer.from_pretrained (Rust)
    ↓
Loads:
    - new_tokenizer/tokenizer.json
    - old_tokenizer/tokenizer.json
    - metadata/*.json
    ↓
Returns HF-compatible tokenizer with Rust backend
```

## File Organization

### Python Package (`src/rbpe/`)

**Training Components** (used only during training):
- `rbpe_tokenizer.py` - Main factory class
- `token_classifier.py` - Token classification
- `data_cleaner.py` - Data cleaning
- `bpe_tokenizer_trainer.py` - BPE training
- `mapping_tokenizer.py` - Mapping creation (training-time only!)
- `cli.py` - Command-line interface
- `utils/` - Unicode handling, data readers

**Runtime Component** (copied to saved tokenizer):
- `tokenization_rbpe.py` - HuggingFace wrapper (uses Rust backend)

**Data**:
- `data/unicode/` - Unicode data files
- `data/unicode_derived/` - Generated language maps
- `data/arabic_chars_rbpe_norm.pickle` - Arabic normalization data

### Rust Package (`rbpe-tokenizers/`)

**Core Implementation**:
- `lib.rs` - Library entry point
- `fast_tokenizer.rs` - Main tokenizer
- `pretokenizer.rs` - Language-aware splitting
- `model.rs` - Dual tokenizer + routing
- `decoder.rs` - Decoding logic
- `normalizer.rs` - Text normalization
- `tokenizer.rs` - Tokenizer traits
- `hf_builder.rs` - HuggingFace integration helpers
- `python_bindings.rs` - PyO3 bindings
- `utils/` - Mappings, Unicode ranges

**Build System**:
- `Cargo.toml` - Rust dependencies
- `pyproject.toml` - Python package metadata (maturin)

## Key Design Decisions

### 1. Why Two Phases?

- **Training in Python**: Leverages HuggingFace ecosystem for training
- **Runtime in Rust**: Achieves 11x speed improvement with no accuracy loss
- **Best of both worlds**: Easy training, fast inference

### 2. Why Keep MappingTokenizer in Python?

- Only used during training to create mapping files
- No need for Rust version since it's not performance-critical
- Simplifies codebase maintenance

### 3. Why Copy tokenization_rbpe.py?

- Enables `AutoTokenizer.from_pretrained()` with `trust_remote_code=True`
- Makes tokenizer fully self-contained
- Works offline once downloaded

### 4. Why Dual Tokenizers?

- New tokenizer: Optimized for target language (better token efficiency)
- Old tokenizer: Handles other languages (maintains model compatibility)
- ID mapping: Keeps everything in original vocabulary space (no model changes needed)

## Performance Characteristics

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Single encode | ~1ms | ~90µs | 11x |
| Batch encode (100 texts) | ~100ms | ~9ms | 11x |
| Decode | ~1ms | ~90µs | 11x |
| Throughput | 6,000/s | 66,000/s | 11x |

## Dependencies

**Python Training**:
- transformers
- datasets
- tokenizers
- torch
- tqdm
- PyYAML

**Rust Runtime**:
- tokenizers (Rust crate)
- serde / serde_json
- pyo3 (for Python bindings)
- unicode-normalization

## Development Workflow

### Adding a New Feature

1. **Training-time feature** → Add to Python (`src/rbpe/`)
2. **Runtime feature** → Add to Rust (`rbpe-tokenizers/src/`)
3. **Both phases** → Coordinate changes in both

### Testing

```bash
# Test training
python -c "from rbpe import RBPETokenizer; ..."

# Test Rust directly
cd rbpe-tokenizers && cargo test

# Test Python bindings
cd rbpe-tokenizers && maturin develop --release
python test_python_bindings.py

# Test HF integration
python test_autotokenizer.py
python test_hf_ecosystem.py
```

### Building

```bash
# Build Rust bindings
cd rbpe-tokenizers
maturin develop --release  # Development
maturin build --release     # Production wheel

# Install Python package
pip install -e .
```

## Migration Notes

**Old Architecture** (deprecated):
- Had Python runtime tokenizer (`dynamic_tokenizer.py`, runtime `mapping_tokenizer.py`)
- Slower performance
- Complex class hierarchy

**New Architecture** (current):
- Rust runtime with Python training
- 11x faster
- Cleaner separation of concerns
- `dynamic_tokenizer.py` - REMOVED
- `mapping_tokenizer.py` - Kept only for training-time mapping creation

**To update existing code**:
```python
# OLD (deprecated)
from rbpe import RBPETokenizer
tokenizer = RBPETokenizer.from_pretrained("path")

# NEW (recommended)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("path", trust_remote_code=True)
```

## Future Enhancements

- Support for more target languages
- Dynamic language detection (multiple target languages)
- Streaming tokenization for very large texts
- WASM compilation for browser usage
- Additional normalization options
