# Python Implementation Analysis

## Overview
The R-BPE tokenizer uses a dual-tokenizer approach with runtime language detection and ID mapping.

## Core Components

### 1. MappingTokenizer.encode() Flow

```
Input Text
    ↓
├─→ Split by special tokens (_split_by_new_specials)
│   - Identifies new special tokens using regex
│   - Returns segments: [(text, is_always_new)]
│
├─→ For each segment:
│   │
│   ├─→ If is_always_new:
│   │   - Encode with new_tokenizer
│   │   - Map new_ids → old_ids
│   │
│   └─→ Else:
│       ├─→ Apply normalization (if enabled)
│       │
│       ├─→ Segment by language (_segment_input_text)
│       │   - Char-by-char Unicode range check
│       │   - Whitespace stays with current segment
│       │   - Returns: [(segment_text, is_target)]
│       │
│       └─→ For each language segment:
│           ├─→ If is_target: new_tokenizer.encode() → map to old_ids
│           └─→ Else: old_tokenizer.encode() → already old_ids
│
└─→ Return concatenated old_ids
```

### 2. Language Segmentation (_segment_input_text)

**Algorithm:**
```python
segments = []
current_segment = []
is_current_target = None

for char in text:
    is_char_target = _is_target_input(char)  # Unicode range check
    
    if char.isspace():
        current_segment.append(char)  # Whitespace joins current segment
        continue
    
    if is_current_target is None:  # First non-whitespace
        is_current_target = is_char_target
        current_segment.append(char)
        continue
    
    if is_char_target != is_current_target:  # Language switch
        segments.append((''.join(current_segment), is_current_target))
        current_segment = [char]
        is_current_target = is_char_target
    else:
        current_segment.append(char)

# Add final segment
if current_segment:
    segments.append((''.join(current_segment), is_current_target))

return segments
```

**Key behaviors:**
- Whitespace inherits the language of the current segment
- Switches segments when language changes
- Uses Unicode code point ranges for detection

### 3. Unicode Target Detection (_is_target_input)

```python
def _is_target_input(self, text):
    for char in text:
        code_point = ord(char)
        if any(start <= code_point <= end for start, end in self.target_language_scripts_ranges):
            return True
    return False
```

**Example ranges for Arabic:**
- Arabic: U+0600 to U+06FF
- Arabic Supplement: U+0750 to U+077F
- Arabic Extended: U+08A0 to U+08FF
- etc.

### 4. Unicode Normalization

Located in: `src/rbpe/utils/unicode_normalizer.py`

**Key transformations:**
- Character replacements for Arabic script
- Handled before segmentation (when enabled)
- Applied only to non-special-token segments

### 5. Decode Flow

```
Input IDs (old vocab space)
    ↓
├─→ Group by mapping status:
│   - Mapped: ID exists in old_to_new_map
│   - Unmapped: ID does not exist in old_to_new_map
│
├─→ Create segments: [(id_list, is_mapped)]
│
├─→ For each segment:
│   ├─→ If mapped: old_ids → new_ids → new_tokenizer.decode()
│   └─→ Else: old_tokenizer.decode()
│
└─→ Concatenate decoded segments
```

### 6. Replacement Character Handling (Complex Decode)

**Problem:** 
- UTF-8 characters may be split across multiple token IDs
- Direct decode produces � (replacement character)

**Solution:** Sliding window approach

```python
def _find_optimal_window(ids, start_idx, current_segment, current_is_mapped):
    # Try windows of size 1-4 tokens
    for window_size in range(1, min(5, len(ids) - start_idx + 1)):
        test_window = ids[start_idx:start_idx+window_size]
        test_segment = current_segment + test_window
        
        # Try decoding with appropriate tokenizer
        decoded = decode_with_tokenizer(test_segment)
        
        if "�" not in decoded:
            return window_size, decoded, is_mapped
    
    return None  # Could not resolve
```

**Used when:**
- Initial decode contains �
- Token is in replacement_character_map
- Token is a byte-level token (ID > last_special && ID <= 256 + last_special)

## Data Structures

### Vocabulary Maps
```python
new_to_old_map: Dict[int, int]  # new_token_id → old_token_id
old_to_new_map: Dict[int, int]  # old_token_id → new_token_id
```

### Token Classification
```python
token_id_language_map = {
    'arabic': {
        'tokens': [id1, id2, ...],
        'ranges': [(start, end), ...]
    },
    'latin': { ... },
    # ... other languages
}
```

### Reusable Tokens
```python
reusable_token_ids = [
    # Token IDs from old tokenizer that can be reused
    # Excludes: special tokens (0-8) and initial bytes (9-263)
]
```

## Key Constants

```python
SPECIAL_TOKEN_THRESHOLD = 8
BYTE_TOKEN_START = 9
BYTE_TOKEN_END = 263
```

## Critical Invariants

1. **ID Space:** All final IDs are in old tokenizer's vocab space
2. **Mapping Coverage:** All new_tokenizer IDs have mapping to old IDs
3. **Segment Continuity:** Whitespace never creates new segments
4. **Decode Reversibility:** encode(text) → decode() should recover original (modulo normalization)

## Performance Characteristics

**Bottlenecks (Python):**
- Character-by-character iteration in segmentation
- Multiple tokenizer calls (old + new)
- Dict lookups for ID mapping
- Replacement character window search

**Rust Optimization Opportunities:**
- Zero-copy string slicing
- Vectorized Unicode range checks
- Inline small mappings
- Pre-computed lookup tables
