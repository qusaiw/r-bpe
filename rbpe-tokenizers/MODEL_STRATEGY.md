# R-BPE Model Implementation Strategy

## Problem Statement

We need to create a custom tokenization model that:
1. Segments text by language (already done in PreTokenizer)
2. Routes each segment to the appropriate BPE tokenizer (new or old)
3. Maps new tokenizer IDs back to old tokenizer IDs
4. Returns token IDs in the old vocabulary space

## Architecture Options

### Option A: Dual BPE Wrapper (RECOMMENDED)
**Approach:** Load both BPE models, route based on segment metadata

**Pros:**
- Uses proven BPE implementations
- Lower complexity
- Easier to maintain
- Can leverage existing tokenizer.json files

**Cons:**
- Two models in memory
- Slightly more complex routing logic

**Implementation:**
```rust
struct RBPEModel {
    new_model: BPE,        // For target language
    old_model: BPE,        // For other languages
    vocab_mapper: VocabMapper,
}

fn tokenize(text, segments) {
    for segment in segments {
        if segment.is_target {
            new_ids = new_model.tokenize(segment.text)
            old_ids = vocab_mapper.map_new_to_old(new_ids)
        } else {
            old_ids = old_model.tokenize(segment.text)
        }
        all_ids.extend(old_ids)
    }
}
```

### Option B: Custom BPE Implementation
**Approach:** Implement BPE from scratch with segment awareness

**Pros:**
- Single model
- Could be more memory efficient
- Full control over algorithm

**Cons:**
- High complexity
- Need to reimplement proven BPE algorithm
- More potential for bugs
- Harder to maintain

## Decision: Option A

We'll use Option A (Dual BPE Wrapper) because:
1. Lower risk - uses battle-tested BPE implementations
2. Faster development - leverage existing code
3. Easier debugging - can test each model independently
4. Better maintainability

## Implementation Plan

### Phase 1: Rust Tokenizer Integration
Load tokenizers using the `tokenizers` crate:

```rust
use tokenizers::Tokenizer;

let new_tokenizer = Tokenizer::from_file("new_tokenizer/tokenizer.json")?;
let old_tokenizer = Tokenizer::from_file("old_tokenizer/tokenizer.json")?;
```

### Phase 2: Encoding Pipeline
```rust
pub fn encode(&self, text: &str) -> Vec<u32> {
    // 1. Normalize (if enabled)
    let normalized = self.normalizer.normalize(text);
    
    // 2. Pre-tokenize (segment by language)
    let segments = self.pretokenizer.pre_tokenize(&normalized);
    
    // 3. Tokenize each segment
    let mut all_ids = Vec::new();
    for segment in segments {
        if segment.is_special_token {
            // Special tokens: use old tokenizer
            let ids = self.old_tokenizer.encode(segment.text, false)?;
            all_ids.extend(ids.get_ids());
        } else if segment.is_target {
            // Target language: use new tokenizer, map to old IDs
            let ids = self.new_tokenizer.encode(segment.text, false)?;
            let mapped = self.vocab_mapper.map_new_to_old(ids.get_ids());
            all_ids.extend(mapped);
        } else {
            // Non-target: use old tokenizer directly
            let ids = self.old_tokenizer.encode(segment.text, false)?;
            all_ids.extend(ids.get_ids());
        }
    }
    
    all_ids
}
```

### Phase 3: Decoding Pipeline
```rust
pub fn decode(&self, ids: &[u32]) -> String {
    // 1. Group IDs by mapping status
    let segments = self.group_ids_by_mapping(ids);
    
    // 2. Decode each segment
    let mut decoded_parts = Vec::new();
    for (segment_ids, is_mapped) in segments {
        if is_mapped {
            // Map back to new IDs and decode with new tokenizer
            let new_ids = self.vocab_mapper.map_old_to_new(&segment_ids);
            let text = self.new_tokenizer.decode(&new_ids, false)?;
            decoded_parts.push(text);
        } else {
            // Decode directly with old tokenizer
            let text = self.old_tokenizer.decode(&segment_ids, false)?;
            decoded_parts.push(text);
        }
    }
    
    decoded_parts.join("")
}
```

## File Structure

```
src/
├── model.rs           # RBPEModel implementation
├── tokenizer.rs       # High-level RBPETokenizer wrapper
└── decoder.rs         # Advanced decoding logic (replacement chars)
```

## Testing Strategy

1. **Unit tests**: Test each component
2. **Integration tests**: Compare with Python implementation
3. **Property tests**: Roundtrip encode/decode
4. **Benchmark tests**: Performance vs Python

## Performance Considerations

1. **Memory**: Two BPE models (~50MB each = 100MB total)
2. **Speed**: Segment routing overhead is minimal
3. **Optimization**: Can share common data structures where possible

## Next Steps

1. Implement basic RBPEModel struct
2. Add encode() method
3. Add basic decode() method
4. Test with real tokenizer files
5. Add advanced decode() with replacement character handling
6. Benchmark performance
