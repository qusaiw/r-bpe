# R-BPE Recipe: Install, Train, Use

A simple guide to get started with R-BPE tokenizer.

## 1. Installation

### Prerequisites
```bash
# Python 3.7+
python --version

# Rust (for high-performance tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Install R-BPE
```bash
# Clone repository
git clone <repository-url>
cd r-bpe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python package
pip install -e .

# Build Rust tokenizer (for high performance)
cd rbpe-tokenizers
maturin develop --release
cd ..
```

Verify installation:
```bash
python -c "from transformers import AutoTokenizer; print('✓ Installation successful')"
```

---

## 2. Training a New R-BPE Tokenizer

### Option A: Using CLI (Recommended)

```bash
rbpe create-tokenizer \
  --model_id meta-llama/Llama-3.1-8B \
  --training_data_dir ./data/arabic_corpus \
  --output_dir ./my_rbpe_tokenizer \
  --target_language_scripts arabic \
  --preserved_languages_scripts latin greek \
  --hf_token YOUR_HUGGINGFACE_TOKEN \
  --min_reusable_count 20000
```

### Option B: Using Python API

```python
from rbpe import RBPETokenizer

# Create and train tokenizer
tokenizer_factory = RBPETokenizer(
    model_id='meta-llama/Llama-3.1-8B',
    training_data_dir='./data/arabic_corpus',
    target_language_scripts=['arabic'],
    preserved_languages_scripts=['latin', 'greek'],
    min_reusable_count=20000,
    hf_token='YOUR_HUGGINGFACE_TOKEN'
)

# Train and prepare
tokenizer = tokenizer_factory.prepare()

# Save
tokenizer.save_pretrained('./my_rbpe_tokenizer')
```

### Option C: Using Config File

Create `config.yaml`:
```yaml
model_id: meta-llama/Llama-3.1-8B
training_data_dir: ./data/arabic_corpus
output_dir: ./my_rbpe_tokenizer
target_language_scripts:
  - arabic
preserved_languages_scripts:
  - latin
  - greek
min_reusable_count: 20000
hf_token: YOUR_HUGGINGFACE_TOKEN
clean_data: true
```

Run:
```bash
rbpe create-tokenizer --config config.yaml
```

### Training Data Format

Your training data should be plain text files:
```
data/arabic_corpus/
├── file1.txt
├── file2.txt
└── file3.txt
```

Each file contains Arabic text (one sentence per line or continuous text).

---

## 3. Using R-BPE Tokenizer

### Method 1: With AutoTokenizer (Recommended - HuggingFace Compatible)

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./my_rbpe_tokenizer",
    trust_remote_code=True
)

# Encode text
text = "Hello مرحبا World عالم"
ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")

# Decode
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")

# Batch processing
texts = ["Hello", "مرحبا", "World"]
batch = tokenizer(texts, padding=True, return_tensors="pt")
print(f"Batch shape: {batch['input_ids'].shape}")
```

### Method 2: Direct Rust API (Maximum Performance)

```python
import rbpe_tokenizers

# Load tokenizer
tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("./my_rbpe_tokenizer")

# Encode
ids = tokenizer.encode("Hello مرحبا World")

# Decode
text = tokenizer.decode(ids)

# Batch
batch_ids = tokenizer.encode_batch(["Hello", "مرحبا", "World"])
```

### Method 3: Original Python API

```python
from rbpe import RBPETokenizer

# Load tokenizer
tokenizer = RBPETokenizer.from_pretrained("./my_rbpe_tokenizer")

# Use like any tokenizer
encoded = tokenizer("Hello مرحبا World")
decoded = tokenizer.decode(encoded['input_ids'])
```

---

## 4. Using with HuggingFace Datasets

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./my_rbpe_tokenizer",
    trust_remote_code=True
)

# Load dataset
dataset = load_dataset("your-dataset")

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

---

## 5. Using with HuggingFace Trainer

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "./my_rbpe_tokenizer",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained("your-model")

# Prepare dataset
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
    batched=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train
trainer.train()
```

---

## 6. Pre-trained R-BPE Tokenizer

If you want to use an existing trained R-BPE tokenizer (skip training):

```python
from transformers import AutoTokenizer

# Use the provided tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./rbpe_tokenizer",  # Path to pre-trained tokenizer
    trust_remote_code=True
)

# Start using immediately
ids = tokenizer.encode("Hello مرحبا World")
```

---

## 7. Common Use Cases

### A. Tokenize Single Text
```python
text = "This is a test هذا اختبار"
ids = tokenizer.encode(text)
```

### B. Tokenize Batch
```python
texts = ["Hello", "مرحبا", "World", "عالم"]
batch = tokenizer(texts, padding=True, return_tensors="pt")
```

### C. Tokenize Dataset
```python
dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True),
    batched=True
)
```

### D. Save and Load
```python
# Save
tokenizer.save_pretrained("./saved_tokenizer")

# Load
tokenizer = AutoTokenizer.from_pretrained(
    "./saved_tokenizer",
    trust_remote_code=True
)
```

### E. Get Vocabulary Info
```python
# Vocabulary size
print(f"Vocab size: {tokenizer.vocab_size}")

# Special tokens
print(f"BOS: {tokenizer.bos_token}")
print(f"EOS: {tokenizer.eos_token}")
print(f"PAD: {tokenizer.pad_token}")
```

---

## 8. Performance Tips

### Use Rust Backend
Always use the Rust backend for best performance:
```bash
cd rbpe-tokenizers
maturin develop --release
```

### Batch Processing
Process in batches for efficiency:
```python
# Good: Batch processing
tokenizer(texts, padding=True, return_tensors="pt")

# Slow: One at a time
[tokenizer(text) for text in texts]
```

### Use `batched=True` with Datasets
```python
dataset.map(tokenize_function, batched=True)  # Fast
dataset.map(tokenize_function, batched=False)  # Slow
```

---

## 9. Troubleshooting

### "Rust R-BPE tokenizer is required"
```bash
cd rbpe-tokenizers
maturin develop --release
```

### "trust_remote_code required"
Add `trust_remote_code=True`:
```python
AutoTokenizer.from_pretrained("path", trust_remote_code=True)
```

### "New tokenizer not found"
Make sure directory structure is correct:
```
my_rbpe_tokenizer/
├── tokenization_rbpe.py
├── tokenizer_config.json
├── new_tokenizer/
├── old_tokenizer/
└── metadata/
```

### Import Error
```bash
pip install transformers datasets torch
```

---

## 10. Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install -e . && cd rbpe-tokenizers && maturin develop --release` |
| Train | `rbpe create-tokenizer --config config.yaml` |
| Load | `AutoTokenizer.from_pretrained("path", trust_remote_code=True)` |
| Encode | `tokenizer.encode("text")` |
| Decode | `tokenizer.decode(ids)` |
| Batch | `tokenizer(texts, padding=True, return_tensors="pt")` |
| Save | `tokenizer.save_pretrained("path")` |

---

## Complete Example: End-to-End

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Load tokenizer (pre-trained or your own)
tokenizer = AutoTokenizer.from_pretrained(
    "./rbpe_tokenizer",
    trust_remote_code=True
)

# 2. Load dataset
dataset = load_dataset("your-dataset")

# 3. Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# 5. Setup training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# 6. Train
trainer.train()

# 7. Save
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
```

---

## That's It!

Three simple steps:
1. **Install**: `pip install -e . && cd rbpe-tokenizers && maturin develop --release`
2. **Train**: `rbpe create-tokenizer --config config.yaml` (or use pre-trained)
3. **Use**: `AutoTokenizer.from_pretrained("path", trust_remote_code=True)`

For more details, see:
- [SESSION_6_SUMMARY.md](SESSION_6_SUMMARY.md) - Complete documentation
- [QUICKSTART_AUTOTOKENIZER.md](QUICKSTART_AUTOTOKENIZER.md) - Quick reference
- [README.md](README.md) - Full project documentation
