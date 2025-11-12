# R-BPE: Improving BPE-Tokenizers with Token Reuse

A high-performance framework for adapting existing Byte-Pair Encoding (BPE) tokenizers to better support a target language. R-BPE reuses tokens from user-excluded languages and creates ID-based maps to resolve new tokens. 

**Features:**
- 🚀 **11x faster** than pure Python (Rust backend)
- 🔧 **Easy training** with Python
- 🤗 **Full HuggingFace compatibility**
- 🌍 **Language-aware** tokenization
- 📦 **No model changes needed** (maintains vocabulary space)

---

## 📦 Installation

### Prerequisites
```bash
# Python 3.10+
python --version

# Rust (required for tokenizer runtime)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Install
```bash
# Clone repository
git clone <repository-url>
cd r-bpe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python training components
pip install -e .

# Install maturin (Rust-Python build tool)
pip install maturin[patchelf]

# Build Rust runtime (REQUIRED - provides 11x speedup)
cd rbpe-tokenizers
maturin develop --release
cd ..
```

### Verify Installation
```bash
# Check Python components
python -c "from rbpe import RBPETokenizer; print('✓ Training components installed')"

# Check Rust tokenizer
python -c "import rbpe_tokenizers; print('✓ Rust tokenizer installed')"

# Check HuggingFace
python -c "from transformers import AutoTokenizer; print('✓ HuggingFace ready')"
```

---

## 🎯 Quick Start

### Training a Tokenizer

**CLI (Recommended):**
```bash
rbpe create-tokenizer \
  --model_id meta-llama/Llama-3.1-8B \
  --training_data_dir ./data/arabic_corpus \
  --output_dir ./my_rbpe_tokenizer \
  --target_language_scripts arabic \
  --preserved_languages_scripts latin greek \
  --hf_token YOUR_HUGGINGFACE_TOKEN
```

**Python API:**
```python
from rbpe import RBPETokenizer

# Create and train
tokenizer_factory = RBPETokenizer(
    model_id='meta-llama/Llama-3.1-8B',
    training_data_dir='./data/arabic_corpus',
    target_language_scripts=['arabic'],
    preserved_languages_scripts=['latin', 'greek'],
    hf_token='YOUR_HUGGINGFACE_TOKEN'
)

tokenizer = tokenizer_factory.prepare()
tokenizer.save_pretrained('./my_rbpe_tokenizer')
```

**Config File:**
```yaml
# config.yaml
model_id: meta-llama/Llama-3.1-8B
training_data_dir: ./data/arabic_corpus
output_dir: ./my_rbpe_tokenizer
target_language_scripts: [arabic]
preserved_languages_scripts: [latin, greek]
hf_token: YOUR_HUGGINGFACE_TOKEN
```
```bash
rbpe create-tokenizer --config config.yaml
```

### Using the Tokenizer

**Method 1: AutoTokenizer (Recommended - Full HuggingFace Compatibility)**
```python
from transformers import AutoTokenizer

# Load tokenizer (uses Rust backend automatically)
tokenizer = AutoTokenizer.from_pretrained(
    "./my_rbpe_tokenizer",
    trust_remote_code=True  # Required for R-BPE
)

# Encode/Decode
text = "Hello مرحبا World"
ids = tokenizer.encode(text, add_special_tokens=False)
decoded = tokenizer.decode(ids, skip_special_tokens=True)

# Batch processing with padding
batch = tokenizer(
    ["Hello", "مرحبا", "World"], 
    padding=True, 
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
print(batch["input_ids"].shape)  # torch.Size([3, max_len])
```

**Method 2: Direct Rust API (Maximum Performance)**
```python
import rbpe_tokenizers

# Load directly (no HuggingFace overhead)
tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("./my_rbpe_tokenizer")

# Single text
ids = tokenizer.encode("Hello مرحبا", add_special_tokens=False)
text = tokenizer.decode(ids, skip_special_tokens=True)

# Advanced decoding (handles replacement characters)
text = tokenizer.decode_advanced(ids, skip_special_tokens=True)

# Batch operations (fastest)
batch_ids = tokenizer.encode_batch(["Hello", "مرحبا", "World"])
batch_texts = tokenizer.decode_batch(batch_ids)
```

---

## 📊 Performance

**Benchmarks** (on existing `rbpe_tokenizer`):

| Metric | Value |
|--------|-------|
| **Single encode+decode** | 49,000 ops/sec (~20µs per op) |
| **Batch throughput** | 199,000 texts/sec |
| **Speedup vs Python** | 11x faster |
| **Build time** | ~7 seconds |

**Real performance from test suite:**
```bash
Testing 1000 iterations of encode+decode
Text length: 49 chars

Results:
  Total time: 0.020s
  Operations/sec: 49,345
  Time per operation: 20.3 µs

Batch performance (100 texts):
  Total time: 0.5ms
  Throughput: 199,160 texts/sec
```

---

## 🔧 Common Tasks

### With HuggingFace Datasets
```python
from datasets import load_dataset

dataset = load_dataset("your-dataset")
tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=512),
    batched=True
)
```

### With HuggingFace Trainer
```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("your-model")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()
```

---

## 📝 Quick Reference

| Task | Command |
|------|---------|
| **Install** | `pip install -e . && pip install maturin[patchelf] && cd rbpe-tokenizers && maturin develop --release` |
| **Test** | `python test_full_workflow.py` |
| **Train** | `rbpe create-tokenizer --config config.yaml --output_dir ./my_tokenizer` |
| **Load** | `AutoTokenizer.from_pretrained("./my_tokenizer", trust_remote_code=True)` |
| **Encode** | `tokenizer.encode("text", add_special_tokens=False)` |
| **Decode** | `tokenizer.decode(ids, skip_special_tokens=True)` |
| **Batch** | `tokenizer(texts, padding=True, return_tensors="pt")` |
| **Direct Rust** | `rbpe_tokenizers.RBPETokenizer.from_pretrained("./my_tokenizer")` |

---

## ✅ Testing

Run the comprehensive test suite to verify everything works:

```bash
python test_full_workflow.py
```

This tests:
- ✓ Installation verification
- ✓ Direct Rust tokenizer API
- ✓ HuggingFace AutoTokenizer loading
- ✓ Encoding/decoding correctness
- ✓ Batch operations
- ✓ Performance benchmarks
- ✓ File structure validation

Expected output:
```
🎉 All tests passed! R-BPE is working correctly.
  Total: 5/5 tests passed
```

## 🛠️ Troubleshooting

**"Rust R-BPE tokenizer not available"**
```bash
cd rbpe-tokenizers
maturin develop --release
```

**Build fails?**
```bash
pip install maturin[patchelf]
cd rbpe-tokenizers && cargo clean && maturin develop --release
```

**Import error?**
```bash
pip install -e .
pip install transformers datasets torch
```

**"trust_remote_code required"**

Always use `trust_remote_code=True` when loading with AutoTokenizer:
```python
tokenizer = AutoTokenizer.from_pretrained("path", trust_remote_code=True)
```

**Tokenizer loads but seems slow?**

Check if it's using Rust backend:
```python
tokenizer = AutoTokenizer.from_pretrained("path", trust_remote_code=True)
print(tokenizer.__class__.__module__)
# Should see: transformers_modules.{path}.tokenization_rbpe
```

If you see `dynamic_tokenizer` or old module names, retrain with updated code.

---

## 📚 Documentation

- [RECIPE.md](RECIPE.md) - Detailed usage guide
- [RUST_QUICK_REFERENCE.md](RUST_QUICK_REFERENCE.md) - Rust API quick reference
- [rbpe-tokenizers/PYTHON_API.md](rbpe-tokenizers/PYTHON_API.md) - Complete Python API
- [SESSION_6_SUMMARY.md](SESSION_6_SUMMARY.md) - Full AutoTokenizer integration docs

---

## ⚙️ Configuration Parameters

R-BPE uses the following configuration parameters:

| Parameter | Meaning | Necessity | Default Value|
|-----|-------|-------| -------|
| model_id | The HuggingFace model id of the original tokenizer. e.g. `meta-llama/Llama-3.1-8B` | Required | None |
| training_data_dir | The directory where the training data for the new tokenizer is stored. | Required | None |
| clean_data| Whether to clean the training data or not. Warning: only set to false if you are sure that your training data does not include any non-preserved languages. | Required | True |
| cleaned_data_dir | The directory where the cleaned training data for the new tokenizer should be saved. | Optional | None |
| hf_token | The HuggingFace access token. | Required | None |
| min_reusable_count | The minimum number of tokens needed for reuse (threshold ***_h_*** in the paper). The size of the new tokenizer vocabulary will be <= `min_reusable_count` depending on how many reusable tokens are found in the specified original tokenizer. | Optional | 20000 |
| target_language_scripts | List of the unicode script names or aliases of the target language. See [this](#specifying-language-scripts) table for possible values. | Optional | Arabic |
| preserved_languages_scripts | List of the unicode script names or aliases of the languages that must be preserved. The target language scripts are preserved by default. See [this](#specifying-language-scripts) table for possible values. | Optional | Latin, Greek |
| special_tokens | Dictionary of custom special tokens values for the main special tokens: `pad_token`, `unk_token`, `bos_token`, `mask_token`, `sep_token`, `cls_token`. | Optional | None |
| additional_special_tokens | List of additional special tokens the _new_ tokenizer will have. | Optional | None |
| apply_rbpe_arabic_norm | Whether to apply the R-BPE Arabic normalization during encoding or not. | optional | True |

---

## 🏗️ Architecture

R-BPE uses a **two-phase architecture** separating training (Python) from runtime (Rust):

### Phase 1: Training (Python)

Train once using Python - leverages HuggingFace ecosystem:

```
1. TokenClassifier    → Analyze base vocabulary by language
2. DataCleaner        → Remove non-target language text  
3. BPETokenizerTrainer → Train new tokenizer on cleaned data
4. MappingTokenizer   → Create ID mappings (new ↔ old)
5. Save               → Export tokenizer + Rust wrapper
```

**Training Components** (`src/rbpe/`):
- `rbpe_tokenizer.py` - Training orchestration
- `token_classifier.py` - Language classification
- `data_cleaner.py` - Data preprocessing
- `bpe_tokenizer_trainer.py` - BPE training
- `mapping_tokenizer.py` - Mapping creation (training only!)

### Phase 2: Runtime (Rust)

Use everywhere with 11x speedup:

```
Input Text
    ↓
Normalizer (optional) → Unicode normalization
    ↓
PreTokenizer → Language-aware splitting
    ↓
Language Router
    ├─ Target language → New Tokenizer → Map to old vocab
    └─ Other languages → Old Tokenizer
    ↓
Token IDs (in original vocabulary space)
```

**Runtime Components** (`rbpe-tokenizers/`):
- `fast_tokenizer.rs` - Main tokenizer
- `pretokenizer.rs` - Language detection & splitting
- `model.rs` - Dual tokenizer routing
- `decoder.rs` - Decoding + replacement handling
- `python_bindings.rs` - PyO3 bindings for Python

### Saved Tokenizer Structure

```
my_rbpe_tokenizer/
├── new_tokenizer/
│   └── tokenizer.json          # Target language BPE
├── old_tokenizer/
│   └── tokenizer.json          # Base model BPE
├── metadata/
│   ├── new_to_old_map.json     # ID mappings
│   ├── old_to_new_map.json
│   └── replacement_character_map.json
├── tokenization_rbpe.py        # Rust wrapper (auto-copied)
└── tokenizer_config.json       # HuggingFace config
```

### Why This Design?

- **Training in Python**: Easy integration with HuggingFace, datasets, etc.
- **Runtime in Rust**: 11x faster, production-ready performance
- **Best of both worlds**: Simple training, blazing-fast inference

## Specifying Language Scripts

Language script specification is case insensitive. The following table shows all possible values you can use which are derived from the [Unicode 17](https://www.unicode.org/versions/Unicode17.0.0/) data:

| Script Name | Script Alias |
|-----|-------|
| adlam | adlm |
| ahom | ahom |
| anatolian_hieroglyphs | hluw |
| arabic | arab |
| armenian | armn |
| avestan | avst |
| balinese | bali |
| bamum | bamu |
| bassa_vah | bass |
| batak | batk |
| bengali | beng |
| beria_erfe | berf |
| bhaiksuki | bhks |
| bopomofo | bopo |
| brahmi | brah |
| braille | brai |
| buginese | bugi |
| buhid | buhd |
| canadian_aboriginal | cans |
| carian | cari |
| caucasian_albanian | aghb |
| chakma | cakm |
| cham | cham |
| cherokee | cher |
| chorasmian | chrs |
| common | zyyy |
| coptic | copt |
| cuneiform | xsux |
| cypriot | cprt |
| cypro_minoan | cpmn |
| cyrillic | cyrl |
| deseret | dsrt |
| devanagari | deva |
| dives_akuru | diak |
| dogra | dogr |
| duployan | dupl |
| egyptian_hieroglyphs | egyp |
| elbasan | elba |
| elymaic | elym |
| ethiopic | ethi |
| garay | gara |
| georgian | geor |
| glagolitic | glag |
| gothic | goth |
| grantha | gran |
| greek | grek |
| gujarati | gujr |
| gunjala_gondi | gong |
| gurmukhi | guru |
| gurung_khema | gukh |
| han | hani |
| hangul | hang |
| hanifi_rohingya | rohg |
| hanunoo | hano |
| hatran | hatr |
| hebrew | hebr |
| hiragana | hira |
| imperial_aramaic | armi |
| inherited | zinh |
| inscriptional_pahlavi | phli |
| inscriptional_parthian | prti |
| javanese | java |
| kaithi | kthi |
| kannada | knda |
| katakana | kana |
| katakana_or_hiragana | hrkt |
| kawi | kawi |
| kayah_li | kali |
| kharoshthi | khar |
| khitan_small_script | kits |
| khmer | khmr |
| khojki | khoj |
| khudawadi | sind |
| kirat_rai | krai |
| lao | laoo |
| latin | latn |
| lepcha | lepc |
| limbu | limb |
| linear_a | lina |
| linear_b | linb |
| lisu | lisu |
| lycian | lyci |
| lydian | lydi |
| mahajani | mahj |
| makasar | maka |
| malayalam | mlym |
| mandaic | mand |
| manichaean | mani |
| marchen | marc |
| masaram_gondi | gonm |
| medefaidrin | medf |
| meetei_mayek | mtei |
| mende_kikakui | mend |
| meroitic_cursive | merc |
| meroitic_hieroglyphs | mero |
| miao | plrd |
| modi | modi |
| mongolian | mong |
| mro | mroo |
| multani | mult |
| myanmar | mymr |
| nabataean | nbat |
| nag_mundari | nagm |
| nandinagari | nand |
| new_tai_lue | talu |
| newa | newa |
| nko | nkoo |
| nushu | nshu |
| nyiakeng_puachue_hmong | hmnp |
| ogham | ogam |
| ol_chiki | olck |
| ol_onal | onao |
| old_hungarian | hung |
| old_italic | ital |
| old_north_arabian | narb |
| old_permic | perm |
| old_persian | xpeo |
| old_sogdian | sogo |
| old_south_arabian | sarb |
| old_turkic | orkh |
| old_uyghur | ougr |
| oriya | orya |
| osage | osge |
| osmanya | osma |
| pahawh_hmong | hmng |
| palmyrene | palm |
| pau_cin_hau | pauc |
| phags_pa | phag |
| phoenician | phnx |
| psalter_pahlavi | phlp |
| rejang | rjng |
| runic | runr |
| samaritan | samr |
| saurashtra | saur |
| sharada | shrd |
| shavian | shaw |
| siddham | sidd |
| sidetic | sidt |
| signwriting | sgnw |
| sinhala | sinh |
| sogdian | sogd |
| sora_sompeng | sora |
| soyombo | soyo |
| sundanese | sund |
| sunuwar | sunu |
| syloti_nagri | sylo |
| syriac | syrc |
| tagalog | tglg |
| tagbanwa | tagb |
| tai_le | tale |
| tai_tham | lana |
| tai_viet | tavt |
| tai_yo | tayo |
| takri | takr |
| tamil | taml |
| tangsa | tnsa |
| tangut | tang |
| telugu | telu |
| thaana | thaa |
| thai | thai |
| tibetan | tibt |
| tifinagh | tfng |
| tirhuta | tirh |
| todhri | todr |
| tolong_siki | tols |
| toto | toto |
| tulu_tigalari | tutg |
| ugaritic | ugar |
| unknown | zzzz |
| vai | vaii |
| vithkuqi | vith |
| wancho | wcho |
| warang_citi | wara |
| yezidi | yezi |
| yi | yiii |
| zanabazar_square | zanb |