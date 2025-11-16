//! R-BPE Model - Dual BPE tokenization with language-aware routing
//!
//! This module implements the core R-BPE model that uses two BPE tokenizers
//! (new and old) and routes text segments based on language detection.

use crate::decoder::RBPEDecoder;
use crate::normalizer::RBPENormalizer;
use crate::pretokenizer::{RBPEPreTokenizer, Segment};
use crate::utils::VocabMapper;
use tokenizers::Tokenizer;
use thiserror::Error;

/// R-BPE Model errors
#[derive(Error, Debug)]
pub enum RBPEError {
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    #[error("Decoder error: {0}")]
    DecoderError(#[from] crate::decoder::DecoderError),
    
    #[error("Other error: {0}")]
    Other(String),
}

/// R-BPE Model
///
/// Combines:
/// - Normalizer (optional Unicode normalization)
/// - PreTokenizer (language-aware segmentation)
/// - Dual BPE models (new for target language, old for others)
/// - Vocabulary mapper (new IDs -> old IDs)
pub struct RBPEModel {
    /// New tokenizer (optimized for target language, e.g., Arabic)
    new_tokenizer: Tokenizer,
    
    /// Old tokenizer (original vocabulary)
    old_tokenizer: Tokenizer,
    
    /// Vocabulary mapper for ID conversion
    vocab_mapper: VocabMapper,
    
    /// Pre-tokenizer for language segmentation
    pretokenizer: RBPEPreTokenizer,
    
    /// Optional normalizer
    normalizer: Option<RBPENormalizer>,
}

impl RBPEModel {
    /// Create a new R-BPE model
    pub fn new(
        new_tokenizer: Tokenizer,
        old_tokenizer: Tokenizer,
        vocab_mapper: VocabMapper,
        pretokenizer: RBPEPreTokenizer,
        normalizer: Option<RBPENormalizer>,
    ) -> Self {
        Self {
            new_tokenizer,
            old_tokenizer,
            vocab_mapper,
            pretokenizer,
            normalizer,
        }
    }

    /// Load from tokenizer files and metadata
    pub fn from_files(
        new_tokenizer_path: &std::path::Path,
        old_tokenizer_path: &std::path::Path,
        new_to_old_map_path: &std::path::Path,
        old_to_new_map_path: &std::path::Path,
        replacement_char_path: Option<&std::path::Path>,
        pretokenizer: RBPEPreTokenizer,
        normalizer: Option<RBPENormalizer>,
    ) -> Result<Self, RBPEError> {
        let new_tokenizer = Tokenizer::from_file(new_tokenizer_path)?;
        let old_tokenizer = Tokenizer::from_file(old_tokenizer_path)?;
        let vocab_mapper = VocabMapper::from_json_files(
            new_to_old_map_path,
            old_to_new_map_path,
            replacement_char_path,
        ).map_err(|e| RBPEError::Other(e.to_string()))?;

        Ok(Self::new(
            new_tokenizer,
            old_tokenizer,
            vocab_mapper,
            pretokenizer,
            normalizer,
        ))
    }

    /// Encode text to token IDs (in old vocabulary space)
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, RBPEError> {
        // 1. Apply normalization if enabled
        let normalized_text = if let Some(ref normalizer) = self.normalizer {
            normalizer.normalize(text)
        } else {
            text.to_string()
        };

        // 2. Pre-tokenize into language-aware segments
        let segments = self.pretokenizer.pre_tokenize(&normalized_text);

        // 3. Encode each segment and collect IDs
        let mut all_ids = Vec::new();

        for segment in segments {
            let segment_ids = self.encode_segment(&segment, add_special_tokens)?;
            all_ids.extend(segment_ids);
        }

        Ok(all_ids)
    }

    /// Encode a single segment
    fn encode_segment(&self, segment: &Segment, add_special_tokens: bool) -> Result<Vec<u32>, RBPEError> {
        if segment.is_special_token {
            // Special tokens: encode with old tokenizer
            let encoding = self.old_tokenizer.encode(segment.text.clone(), add_special_tokens)?;
            Ok(encoding.get_ids().to_vec())
        } else if segment.is_target {
            // Target language: encode with new tokenizer, then map to old IDs
            let encoding = self.new_tokenizer.encode(segment.text.clone(), add_special_tokens)?;
            let new_ids = encoding.get_ids();
            Ok(self.vocab_mapper.map_new_to_old(new_ids))
        } else {
            // Non-target language: encode with old tokenizer
            let encoding = self.old_tokenizer.encode(segment.text.clone(), add_special_tokens)?;
            Ok(encoding.get_ids().to_vec())
        }
    }

    /// Decode token IDs back to text (basic version without replacement character handling)
    /// 
    /// This is a faster version that doesn't handle replacement characters.
    /// For production use, prefer `decode()` which automatically handles edge cases.
    pub fn decode_basic(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, RBPEError> {
        // Group IDs by whether they're mapped
        let segments = self.group_ids_by_mapping(ids);

        // Decode each segment
        let mut decoded_parts = Vec::new();
        for (segment_ids, is_mapped) in segments {
            if is_mapped {
                // Map back to new IDs and decode with new tokenizer
                let new_ids = self.vocab_mapper.map_old_to_new(&segment_ids);
                let text = self.new_tokenizer.decode(&new_ids, skip_special_tokens)?;
                decoded_parts.push(text);
            } else {
                // Decode directly with old tokenizer
                let text = self.old_tokenizer.decode(&segment_ids, skip_special_tokens)?;
                decoded_parts.push(text);
            }
        }

        Ok(decoded_parts.join(""))
    }

    /// Decode token IDs back to text with automatic replacement character handling
    /// 
    /// This is the recommended decode method that matches Python's MappingTokenizer behavior.
    /// It automatically detects and handles replacement characters using the advanced decoder
    /// when needed, falling back to fast basic decode when possible.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, RBPEError> {
        self.decode_advanced(ids, skip_special_tokens)
    }

    /// Advanced decode with replacement character handling using sliding window
    /// 
    /// This method handles UTF-8 byte sequences that may be split across multiple tokens.
    /// If the basic decode produces replacement characters (�), it uses a sliding window
    /// algorithm to find optimal token groupings.
    pub fn decode_advanced(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, RBPEError> {
        // Try basic decode first
        let basic_decoded = self.decode_basic(ids, skip_special_tokens)?;
        
        // If no replacement characters, return early (fast path)
        if !basic_decoded.contains('�') {
            return Ok(basic_decoded);
        }

        // Use sliding window algorithm for replacement characters (slow path)
        // Get last special token ID from old tokenizer (assuming 128000 for Llama-based tokenizers)
        let old_last_special_token_id = 128000;
        
        // Clone tokenizers only when we need advanced decoding (rare case)
        let decoder = RBPEDecoder::new(
            self.new_tokenizer.clone(),
            self.old_tokenizer.clone(),
            self.vocab_mapper.clone(),
            old_last_special_token_id,
        );
        
        Ok(decoder.decode(ids, skip_special_tokens)?)
    }

    /// Group consecutive IDs by whether they're mapped
    fn group_ids_by_mapping(&self, ids: &[u32]) -> Vec<(Vec<u32>, bool)> {
        let mut segments = Vec::new();
        let mut current_segment = Vec::new();
        let mut current_is_mapped: Option<bool> = None;

        for &id in ids {
            let is_mapped = self.vocab_mapper.is_mapped(id);

            if let Some(was_mapped) = current_is_mapped {
                if is_mapped != was_mapped {
                    // Mapping status changed, start new segment
                    if !current_segment.is_empty() {
                        segments.push((current_segment.clone(), was_mapped));
                        current_segment.clear();
                    }
                }
            }

            current_segment.push(id);
            current_is_mapped = Some(is_mapped);
        }

        // Add final segment
        if !current_segment.is_empty() {
            if let Some(is_mapped) = current_is_mapped {
                segments.push((current_segment, is_mapped));
            }
        }

        segments
    }

    /// Get the vocabulary mapper
    pub fn vocab_mapper(&self) -> &VocabMapper {
        &self.vocab_mapper
    }

    /// Get the pre-tokenizer
    pub fn pretokenizer(&self) -> &RBPEPreTokenizer {
        &self.pretokenizer
    }

    /// Get the normalizer
    pub fn normalizer(&self) -> Option<&RBPENormalizer> {
        self.normalizer.as_ref()
    }
    
    /// Get token ID for a given token string
    /// 
    /// This queries the old tokenizer's vocabulary (R-BPE output space).
    /// Returns None if the token is not in the vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.old_tokenizer.token_to_id(token)
    }
    
    /// Get token string for a given token ID
    /// 
    /// This queries the old tokenizer's vocabulary (R-BPE output space).
    /// Returns None if the ID is not in the vocabulary.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.old_tokenizer.id_to_token(id)
    }
    
    /// Get the vocabulary size
    /// 
    /// Returns the size of the old tokenizer's vocabulary (R-BPE output space).
    pub fn vocab_size(&self) -> usize {
        self.old_tokenizer.get_vocab_size(false) // false = don't include added tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::unicode_ranges::ranges;
    use crate::utils::UnicodeRangeChecker;

    // We'll need actual tokenizer files to test, so these are placeholder tests
    // Real tests will be in integration tests

    #[test]
    fn test_group_ids_by_mapping() {
        use std::collections::HashMap;

        // Create a simple vocab mapper
        let mut new_to_old = HashMap::new();
        new_to_old.insert(0, 0);
        new_to_old.insert(1, 100);
        new_to_old.insert(2, 200);

        let mut old_to_new = HashMap::new();
        old_to_new.insert(0, 0);
        old_to_new.insert(100, 1);
        old_to_new.insert(200, 2);

        let vocab_mapper = VocabMapper::new(new_to_old, old_to_new, HashMap::new(), vec![]);

        // Create minimal model components (we won't actually use the tokenizers in this test)
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);

        // Create dummy tokenizers (these won't be used in this test)
        // In real tests, we'd load actual tokenizer files
        let new_tok = Tokenizer::from_file("../rbpe_tokenizer/new_tokenizer/tokenizer.json").unwrap();
        let old_tok = Tokenizer::from_file("../rbpe_tokenizer/old_tokenizer/tokenizer.json").unwrap();

        let model = RBPEModel::new(new_tok, old_tok, vocab_mapper, pretokenizer, None);

        // Test grouping: [mapped, mapped, unmapped, unmapped, mapped]
        let ids = vec![100, 200, 5, 10, 100];
        let groups = model.group_ids_by_mapping(&ids);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], (vec![100, 200], true));
        assert_eq!(groups[1], (vec![5, 10], false));
        assert_eq!(groups[2], (vec![100], true));
    }
}
