//! FastTokenizer-compatible wrapper for R-BPE
//!
//! This module provides a wrapper around RBPEModel that implements
//! the tokenizers crate's API, making it compatible with HuggingFace's
//! FastTokenizer interface while preserving all R-BPE logic.

use crate::model::RBPEModel;
use tokenizers::Encoding;
use std::path::Path;

/// FastTokenizer-compatible R-BPE wrapper
/// 
/// This struct wraps the RBPEModel and provides the standard tokenizers crate API
/// while internally using all R-BPE logic (dual tokenizers, mappings, language routing).
pub struct RBPEFastTokenizer {
    /// The underlying R-BPE model with all custom logic
    model: RBPEModel,
    
    /// Whether to add special tokens by default
    add_special_tokens: bool,
    
    /// Whether to add BOS token (configured from tokenizer_config.json)
    add_bos_token: bool,
    
    /// Whether to add EOS token (configured from tokenizer_config.json)
    add_eos_token: bool,
    
    /// BOS token ID (from old tokenizer vocabulary)
    bos_token_id: Option<u32>,
    
    /// EOS token ID (from old tokenizer vocabulary)
    eos_token_id: Option<u32>,
}

impl RBPEFastTokenizer {
    /// Create a new FastTokenizer from an existing RBPEModel
    pub fn new(model: RBPEModel) -> Self {
        // Get BOS and EOS token IDs from the old tokenizer
        let bos_token_id = model.token_to_id("<BOS_TOKEN>");
        let eos_token_id = model.token_to_id("<EOS_TOKEN>");
        
        Self {
            model,
            add_special_tokens: true,
            add_bos_token: true,  // Default to true
            add_eos_token: false, // Default to false (matches most configs)
            bos_token_id,
            eos_token_id,
        }
    }
    
    /// Create from file paths
    pub fn from_files(
        new_tokenizer_path: &Path,
        old_tokenizer_path: &Path,
        new_to_old_map_path: &Path,
        old_to_new_map_path: &Path,
        replacement_char_path: Option<&Path>,
        pretokenizer: crate::pretokenizer::RBPEPreTokenizer,
        normalizer: Option<crate::normalizer::RBPENormalizer>,
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
    
    /// Encode text to token IDs using R-BPE logic
    pub fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut ids = self.model.encode(text, add_special_tokens)?;
        
        // Post-process to handle BOS/EOS tokens according to config
        if add_special_tokens {
            // The model.encode already adds BOS via the tokenizer's post-processor
            // We need to handle the EOS token manually if add_eos_token is true
            if self.add_eos_token {
                if let Some(eos_id) = self.eos_token_id {
                    // Add EOS token at the end
                    ids.push(eos_id);
                }
            }
        }
        
        Ok(ids)
    }
    
    /// Decode token IDs to text using R-BPE logic (basic)
    pub fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        Ok(self.model.decode(ids, skip_special_tokens)?)
    }
    
    /// Decode token IDs to text using R-BPE advanced decoder
    pub fn decode_ids_advanced(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        Ok(self.model.decode_advanced(ids, skip_special_tokens)?)
    }
    
    /// Set whether to add special tokens
    pub fn set_add_special_tokens(&mut self, add: bool) {
        self.add_special_tokens = add;
    }
    
    /// Set whether to add BOS token
    pub fn set_add_bos_token(&mut self, add: bool) {
        self.add_bos_token = add;
    }
    
    /// Set whether to add EOS token
    pub fn set_add_eos_token(&mut self, add: bool) {
        self.add_eos_token = add;
    }
    
    /// Get whether BOS token is added
    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }
    
    /// Get whether EOS token is added
    pub fn add_eos_token(&self) -> bool {
        self.add_eos_token
    }
    
    /// Get the underlying model
    pub fn model(&self) -> &RBPEModel {
        &self.model
    }
    
    /// Encode with full Encoding object (for tokenizers crate compatibility)
    pub fn encode_to_encoding(&self, text: &str) -> Result<Encoding, Box<dyn std::error::Error>> {
        // Get token IDs using R-BPE logic
        let ids = self.encode_text(text, self.add_special_tokens)?;
        
        // Create token strings by decoding each ID individually
        let tokens: Vec<String> = ids.iter()
            .map(|&id| {
                // Decode single token
                self.model.decode(&[id], false)
                    .unwrap_or_else(|_| format!("<unk:{}>", id))
            })
            .collect();
        
        // Create Encoding
        let encoding = Encoding::new(
            ids.clone(),
            vec![0; ids.len()], // type_ids - all 0 for single sequence
            tokens,
            vec![], // words - not tracking word boundaries
            vec![(0, text.len()); ids.len()], // offsets - rough estimate
            vec![], // special_tokens_mask
            vec![], // attention_mask will be set automatically
            Vec::new(), // overflowing tokens
            AHashMap::new(), // additional data
        );
        
        Ok(encoding)
    }
    
    /// Batch encode multiple texts
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Encoding>, Box<dyn std::error::Error>> {
        texts.iter()
            .map(|text| self.encode_to_encoding(text))
            .collect()
    }
    
    /// Get token ID for a given token string
    /// 
    /// Returns None if the token is not in the vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.token_to_id(token)
    }
    
    /// Get token string for a given token ID
    /// 
    /// Returns None if the ID is not in the vocabulary.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.id_to_token(id)
    }
    
    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }
}

use ahash::AHashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::UnicodeRangeChecker;
    use crate::utils::unicode_ranges::ranges;
    
    fn create_test_tokenizer() -> RBPEFastTokenizer {
        let base_path = Path::new("../rbpe_tokenizer");
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = crate::pretokenizer::RBPEPreTokenizer::new(checker, vec![]);
        
        RBPEFastTokenizer::from_files(
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
        let ids = tokenizer.encode_text(text, false).expect("Encoding failed");
        let decoded = tokenizer.decode_ids(&ids, false).expect("Decoding failed");
        
        assert_eq!(decoded.trim(), text);
    }
    
    #[test]
    fn test_encoding_object() {
        let tokenizer = create_test_tokenizer();
        
        let text = "Hello World";
        let encoding = tokenizer.encode_to_encoding(text).expect("Encoding failed");
        
        assert!(!encoding.get_ids().is_empty());
        assert_eq!(encoding.get_ids().len(), encoding.get_tokens().len());
    }
    
    #[test]
    fn test_batch_encode() {
        let tokenizer = create_test_tokenizer();
        
        let texts = vec!["Hello", "مرحبا", "World"];
        let encodings = tokenizer.encode_batch(&texts).expect("Batch encoding failed");
        
        assert_eq!(encodings.len(), 3);
        for encoding in encodings {
            assert!(!encoding.get_ids().is_empty());
        }
    }
}
