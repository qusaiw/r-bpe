//! High-level tokenizer wrapper for HuggingFace compatibility
//!
//! This module provides a wrapper around the R-BPE model that can be
//! serialized to tokenizer.json format compatible with HuggingFace's
//! AutoTokenizer.from_pretrained().

use crate::model::RBPEModel;
use crate::normalizer::RBPENormalizer;
use crate::pretokenizer::RBPEPreTokenizer;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// High-level R-BPE tokenizer compatible with HuggingFace
pub struct RBPETokenizer {
    /// The underlying R-BPE model
    model: RBPEModel,
    
    /// Padding configuration
    padding: Option<PaddingConfig>,
    
    /// Truncation configuration
    truncation: Option<TruncationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingConfig {
    pub strategy: String, // "longest", "max_length", "do_not_pad"
    pub direction: String, // "right", "left"
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    pub pad_token: String,
    pub max_length: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationConfig {
    pub max_length: usize,
    pub strategy: String, // "longest_first", "only_first", "only_second"
    pub stride: usize,
    pub direction: String, // "right", "left"
}

impl RBPETokenizer {
    /// Create a new R-BPE tokenizer
    pub fn new(model: RBPEModel) -> Self {
        Self {
            model,
            padding: None,
            truncation: None,
        }
    }
    
    /// Load from files
    pub fn from_files(
        new_tokenizer_path: &Path,
        old_tokenizer_path: &Path,
        new_to_old_map_path: &Path,
        old_to_new_map_path: &Path,
        replacement_char_path: Option<&Path>,
        pretokenizer: RBPEPreTokenizer,
        normalizer: Option<RBPENormalizer>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = RBPEModel::from_files(
            new_tokenizer_path,
            old_tokenizer_path,
            new_to_old_map_path,
            old_to_new_map_path,
            replacement_char_path,
            pretokenizer,
            normalizer,
        )?;
        
        Ok(Self::new(model))
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        Ok(self.model.encode(text, add_special_tokens)?)
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        Ok(self.model.decode(ids, skip_special_tokens)?)
    }
    
    /// Decode with advanced replacement character handling
    pub fn decode_advanced(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        Ok(self.model.decode_advanced(ids, skip_special_tokens)?)
    }
    
    /// Enable padding
    pub fn enable_padding(&mut self, config: PaddingConfig) {
        self.padding = Some(config);
    }
    
    /// Disable padding
    pub fn disable_padding(&mut self) {
        self.padding = None;
    }
    
    /// Enable truncation
    pub fn enable_truncation(&mut self, config: TruncationConfig) {
        self.truncation = Some(config);
    }
    
    /// Disable truncation
    pub fn disable_truncation(&mut self) {
        self.truncation = None;
    }
    
    /// Get the underlying model
    pub fn model(&self) -> &RBPEModel {
        &self.model
    }
    
    /// Get vocab size (returns old tokenizer vocab size)
    pub fn vocab_size(&self) -> usize {
        // Delegate to the underlying model which queries the actual tokenizer
        self.model.vocab_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::UnicodeRangeChecker;
    use crate::utils::unicode_ranges::ranges;
    
    fn create_test_tokenizer() -> RBPETokenizer {
        let base_path = Path::new("../rbpe_tokenizer");
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
        
        RBPETokenizer::from_files(
            &base_path.join("new_tokenizer/tokenizer.json"),
            &base_path.join("old_tokenizer/tokenizer.json"),
            &base_path.join("metadata/new_to_old_map.json"),
            &base_path.join("metadata/old_to_new_map.json"),
            Some(&base_path.join("metadata/replacement_character_map.json")),
            pretokenizer,
            None,
        ).expect("Failed to create tokenizer")
    }
    
    #[test]
    fn test_encode_decode() {
        let tokenizer = create_test_tokenizer();
        
        let text = "Hello مرحبا World";
        let ids = tokenizer.encode(text, false).expect("Encoding failed");
        let decoded = tokenizer.decode(&ids, false).expect("Decoding failed");
        
        assert_eq!(decoded.trim(), text);
    }
    
    #[test]
    fn test_padding_config() {
        let mut tokenizer = create_test_tokenizer();
        
        let config = PaddingConfig {
            strategy: "max_length".to_string(),
            direction: "right".to_string(),
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_token: "<pad>".to_string(),
            max_length: Some(512),
        };
        
        tokenizer.enable_padding(config);
        assert!(tokenizer.padding.is_some());
        
        tokenizer.disable_padding();
        assert!(tokenizer.padding.is_none());
    }
}
