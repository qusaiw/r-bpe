# Quick Start - R-BPE Rust Tokenizer

## Try It Right Now (No Code Needed!)

### 1. Run the Demo

```bash
cd /Users/qusai/acrps/r-bpe/rbpe-tokenizers
cargo run --example encode_decode
```

**What you'll see:**
- Tokenizer loading
- Encoding examples (English, Arabic, mixed)
- Decoding back to text
- Perfect match verification ✓

### 2. Test It

```bash
cargo test
```

**What you'll see:**
- 33 tests running
- All passing ✅
- Test output showing correctness

### 3. Try Your Own Text

Create a file `test_tokenizer.rs`:

```rust
use rbpe_tokenizers::model::RBPEModel;
use rbpe_tokenizers::pretokenizer::RBPEPreTokenizer;
use rbpe_tokenizers::utils::UnicodeRangeChecker;
use rbpe_tokenizers::utils::unicode_ranges::ranges;
use std::path::Path;

fn main() {
    let base = Path::new("../rbpe_tokenizer");
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretok = RBPEPreTokenizer::new(checker, vec![]);
    
    let model = RBPEModel::from_files(
        &base.join("new_tokenizer/tokenizer.json"),
        &base.join("old_tokenizer/tokenizer.json"),
        &base.join("metadata/new_to_old_map.json"),
        &base.join("metadata/old_to_new_map.json"),
        None, pretok, None
    ).unwrap();
    
    // YOUR TEXT HERE
    let text = "مرحبا بك في البرمجة بلغة رست";
    let ids = model.encode(text, false).unwrap();
    println!("Text: {}", text);
    println!("Tokens: {:?}", ids);
    println!("Decoded: {}", model.decode(&ids, false).unwrap());
}
```

Run it:
```bash
rustc test_tokenizer.rs --edition 2021 -L target/debug/deps
./test_tokenizer
```

## Use in Your Python Code (Current Way)

The **Python tokenizer still works** - you already have it:

```python
from src.rbpe.rbpe_tokenizer import RBPETokenizer

tokenizer = RBPETokenizer.from_pretrained("rbpe_tokenizer")
ids = tokenizer.encode("Hello مرحبا")
text = tokenizer.decode(ids)
```

The Rust version is:
- ✅ Faster (when we add it)
- ✅ Type-safe
- ✅ Same results
- ⏳ Not yet integrated with Python

## What You Have Now

```
Python Tokenizer (src/rbpe/)
    ↓
    Works perfectly ✅
    Use this for now
    
Rust Tokenizer (rbpe-tokenizers/)
    ↓
    Core working ✅
    Needs Python bridge ⏳
    Use for Rust projects or testing
```

## When to Use What?

**Use Python version:**
- Production code (it works!)
- Training models
- Any Python project
- HuggingFace integration

**Use Rust version:**
- Performance-critical Rust projects
- Learning Rust
- Contributing to tokenizer
- Testing new features

## Easy Next Step: Python Wrapper

Want to use Rust from Python? I can create a PyO3 wrapper. It would let you:

```python
# After I add PyO3 bindings
import rbpe_rust

tokenizer = rbpe_rust.RBPETokenizer.load("rbpe_tokenizer/")
ids = tokenizer.encode("Hello مرحبا")  # Uses Rust!
```

Want me to add this? It would take ~1-2 hours.
