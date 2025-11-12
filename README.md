# R-BPE: Improving BPE-Tokenizers with Token Reuse

A lightweight framework for adapting existing Byte-Pair Encoding (BPE) tokenizers to better support a target language. R-BPE reuses tokens from user-excluded languages and creates ID-based maps to resolve new tokens. Fully compatible with HuggingFace ecosystem.

---

## 📦 Installation

### Prerequisites
```bash
# Python 3.7+
python --version

# Rust (for high-performance tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Install
```bash
# Clone and setup
git clone <repository-url>
cd r-bpe
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python package
pip install -e .

# Build Rust tokenizer (11x faster, required)
cd rbpe-tokenizers
maturin develop --release
cd ..
```

Verify:
```bash
python -c "from transformers import AutoTokenizer; print('✓ Ready')"
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

**Method 1: AutoTokenizer (HuggingFace Compatible)**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "./my_rbpe_tokenizer",
    trust_remote_code=True
)

# Encode/Decode
ids = tokenizer.encode("Hello مرحبا World")
text = tokenizer.decode(ids)

# Batch processing
batch = tokenizer(["Hello", "مرحبا"], padding=True, return_tensors="pt")
```

**Method 2: Direct Rust API (Maximum Performance)**
```python
import rbpe_tokenizers

tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("./my_rbpe_tokenizer")
ids = tokenizer.encode("Hello مرحبا", add_special_tokens=False)
text = tokenizer.decode(ids, skip_special_tokens=True)
batch_ids = tokenizer.encode_batch(["Hello", "مرحبا", "World"])
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Speed** | 11x faster than Python |
| **Throughput** | 66,000 texts/sec |
| **Accuracy** | 100% parity with Python |
| **Build time** | ~7 seconds |

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
| **Install** | `pip install -e . && cd rbpe-tokenizers && maturin develop --release` |
| **Train** | `rbpe create-tokenizer --config config.yaml` |
| **Load** | `AutoTokenizer.from_pretrained("path", trust_remote_code=True)` |
| **Encode** | `tokenizer.encode("text")` |
| **Decode** | `tokenizer.decode(ids)` |
| **Batch** | `tokenizer(texts, padding=True, return_tensors="pt")` |

---

## 🛠️ Troubleshooting

**Build fails?**
```bash
pip install maturin
cd rbpe-tokenizers && cargo clean && maturin develop --release
```

**Import error?**
```bash
pip install transformers datasets torch
```

**"trust_remote_code required"**
```python
AutoTokenizer.from_pretrained("path", trust_remote_code=True)
```

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

```
Input Text
    ↓
Normalizer (optional) → Normalize Unicode
    ↓
PreTokenizer → Split by language
    ↓
Language Router → Arabic vs Others
    ↓
├─ Arabic → New Tokenizer → Map IDs
└─ Others → Old Tokenizer
    ↓
Token IDs (in old vocabulary space)
```

**Process:**
1. Classify vocabulary tokens by language via `TokenClassifier`
2. Clean training data using `DataCleaner`
3. Train new BPE tokenizer with `BPETokenizerTrainer`
4. Create mappings between original and new tokenizer with `MappingTokenizer`
5. Return final `RBPETokenizer` adapted to target language

**Required Directory Structure:**
```
your_tokenizer/
├── new_tokenizer/tokenizer.json
├── old_tokenizer/tokenizer.json
└── metadata/
    ├── new_to_old_map.json
    ├── old_to_new_map.json
    └── replacement_character_map.json  # optional
```

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