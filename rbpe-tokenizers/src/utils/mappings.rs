//! Token ID mapping utilities
//!
//! This module contains the vocabulary mapping logic between old and new tokenizers.
//! The R-BPE tokenizer uses two vocabularies:
//! - New tokenizer: Optimized for target language (e.g., Arabic)
//! - Old tokenizer: Original vocabulary
//!
//! At runtime, we encode with the appropriate tokenizer based on language,
//! then map new IDs back to old IDs so the model sees consistent vocabulary.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Bidirectional vocabulary mapping between new and old tokenizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabMapper {
    /// Map from new tokenizer ID to old tokenizer ID
    pub new_to_old: HashMap<u32, u32>,
    
    /// Map from old tokenizer ID to new tokenizer ID (inverse)
    pub old_to_new: HashMap<u32, u32>,
    
    /// Token IDs that contain replacement characters (�)
    /// Maps ID to the decoded text (for debugging/handling)
    pub replacement_character_ids: HashMap<u32, String>,
    
    /// Common token IDs (present in both vocabularies with same meaning)
    pub common_token_ids: Vec<u32>,
}

impl VocabMapper {
    /// Create a new vocabulary mapper
    pub fn new(
        new_to_old: HashMap<u32, u32>,
        old_to_new: HashMap<u32, u32>,
        replacement_character_ids: HashMap<u32, String>,
        common_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            new_to_old,
            old_to_new,
            replacement_character_ids,
            common_token_ids,
        }
    }

    /// Load vocabulary mapper from JSON files
    pub fn from_json_files(
        new_to_old_path: &Path,
        old_to_new_path: &Path,
        replacement_char_path: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load new_to_old map
        let new_to_old_file = std::fs::File::open(new_to_old_path)?;
        let new_to_old_str: HashMap<String, u32> = serde_json::from_reader(new_to_old_file)?;
        let new_to_old: HashMap<u32, u32> = new_to_old_str
            .into_iter()
            .filter_map(|(k, v)| k.parse::<u32>().ok().map(|k_int| (k_int, v)))
            .collect();

        // Load old_to_new map
        let old_to_new_file = std::fs::File::open(old_to_new_path)?;
        let old_to_new_str: HashMap<String, u32> = serde_json::from_reader(old_to_new_file)?;
        let old_to_new: HashMap<u32, u32> = old_to_new_str
            .into_iter()
            .filter_map(|(k, v)| k.parse::<u32>().ok().map(|k_int| (k_int, v)))
            .collect();

        // Load replacement character map if provided
        let replacement_character_ids = if let Some(path) = replacement_char_path {
            let file = std::fs::File::open(path)?;
            let map_str: HashMap<String, String> = serde_json::from_reader(file)?;
            map_str
                .into_iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|k_int| (k_int, v)))
                .collect()
        } else {
            HashMap::new()
        };

        // Compute common token IDs (IDs present in both maps that map to themselves)
        let common_token_ids: Vec<u32> = old_to_new
            .iter()
            .filter_map(|(&old_id, &new_id)| {
                if new_to_old.get(&new_id) == Some(&old_id) {
                    Some(old_id)
                } else {
                    None
                }
            })
            .collect();

        Ok(Self::new(
            new_to_old,
            old_to_new,
            replacement_character_ids,
            common_token_ids,
        ))
    }

    /// Map a new tokenizer ID to old tokenizer ID
    #[inline]
    pub fn new_to_old_id(&self, new_id: u32) -> Option<u32> {
        self.new_to_old.get(&new_id).copied()
    }

    /// Map an old tokenizer ID to new tokenizer ID
    #[inline]
    pub fn old_to_new_id(&self, old_id: u32) -> Option<u32> {
        self.old_to_new.get(&old_id).copied()
    }

    /// Check if an old tokenizer ID is mapped (exists in new tokenizer)
    #[inline]
    pub fn is_mapped(&self, old_id: u32) -> bool {
        self.old_to_new.contains_key(&old_id)
    }

    /// Check if an ID contains replacement characters
    #[inline]
    pub fn has_replacement_char(&self, old_id: u32) -> bool {
        self.replacement_character_ids.contains_key(&old_id)
    }

    /// Check if an ID is a common token
    #[inline]
    pub fn is_common_token(&self, old_id: u32) -> bool {
        self.common_token_ids.binary_search(&old_id).is_ok()
    }

    /// Map a sequence of new IDs to old IDs
    pub fn map_new_to_old(&self, new_ids: &[u32]) -> Vec<u32> {
        new_ids
            .iter()
            .filter_map(|&id| self.new_to_old_id(id))
            .collect()
    }

    /// Map a sequence of old IDs to new IDs
    pub fn map_old_to_new(&self, old_ids: &[u32]) -> Vec<u32> {
        old_ids
            .iter()
            .filter_map(|&id| self.old_to_new_id(id))
            .collect()
    }

    /// Get statistics about the mapping
    pub fn stats(&self) -> MapperStats {
        MapperStats {
            new_vocab_size: self.new_to_old.len(),
            old_vocab_size: self.old_to_new.len(),
            mapped_tokens: self.new_to_old.len(),
            common_tokens: self.common_token_ids.len(),
            replacement_char_tokens: self.replacement_character_ids.len(),
        }
    }
}

/// Statistics about vocabulary mapping
#[derive(Debug, Clone)]
pub struct MapperStats {
    pub new_vocab_size: usize,
    pub old_vocab_size: usize,
    pub mapped_tokens: usize,
    pub common_tokens: usize,
    pub replacement_char_tokens: usize,
}

impl MapperStats {
    pub fn print(&self) {
        println!("Vocabulary Mapping Statistics:");
        println!("  New tokenizer vocab size: {}", self.new_vocab_size);
        println!("  Old tokenizer vocab size: {}", self.old_vocab_size);
        println!("  Mapped tokens: {}", self.mapped_tokens);
        println!("  Common tokens: {}", self.common_tokens);
        println!("  Replacement char tokens: {}", self.replacement_char_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mapper() -> VocabMapper {
        let mut new_to_old = HashMap::new();
        new_to_old.insert(0, 0);   // Special token
        new_to_old.insert(1, 100); // Mapped token
        new_to_old.insert(2, 200); // Mapped token
        new_to_old.insert(3, 3);   // Common token

        let mut old_to_new = HashMap::new();
        old_to_new.insert(0, 0);   // Special token
        old_to_new.insert(100, 1); // Mapped token
        old_to_new.insert(200, 2); // Mapped token
        old_to_new.insert(3, 3);   // Common token
        old_to_new.insert(500, 1); // Old token that maps to new

        let mut replacement_chars = HashMap::new();
        replacement_chars.insert(999, "�".to_string());

        let common_tokens = vec![0, 3];

        VocabMapper::new(new_to_old, old_to_new, replacement_chars, common_tokens)
    }

    #[test]
    fn test_new_to_old_mapping() {
        let mapper = create_test_mapper();
        assert_eq!(mapper.new_to_old_id(1), Some(100));
        assert_eq!(mapper.new_to_old_id(2), Some(200));
        assert_eq!(mapper.new_to_old_id(3), Some(3));
        assert_eq!(mapper.new_to_old_id(999), None);
    }

    #[test]
    fn test_old_to_new_mapping() {
        let mapper = create_test_mapper();
        assert_eq!(mapper.old_to_new_id(100), Some(1));
        assert_eq!(mapper.old_to_new_id(200), Some(2));
        assert_eq!(mapper.old_to_new_id(3), Some(3));
        assert_eq!(mapper.old_to_new_id(999), None);
    }

    #[test]
    fn test_is_mapped() {
        let mapper = create_test_mapper();
        assert!(mapper.is_mapped(100));
        assert!(mapper.is_mapped(200));
        assert!(mapper.is_mapped(3));
        assert!(!mapper.is_mapped(999));
    }

    #[test]
    fn test_has_replacement_char() {
        let mapper = create_test_mapper();
        assert!(mapper.has_replacement_char(999));
        assert!(!mapper.has_replacement_char(100));
    }

    #[test]
    fn test_is_common_token() {
        let mapper = create_test_mapper();
        assert!(mapper.is_common_token(0));
        assert!(mapper.is_common_token(3));
        assert!(!mapper.is_common_token(100));
    }

    #[test]
    fn test_map_sequence_new_to_old() {
        let mapper = create_test_mapper();
        let new_ids = vec![1, 2, 3];
        let old_ids = mapper.map_new_to_old(&new_ids);
        assert_eq!(old_ids, vec![100, 200, 3]);
    }

    #[test]
    fn test_map_sequence_old_to_new() {
        let mapper = create_test_mapper();
        let old_ids = vec![100, 200, 3];
        let new_ids = mapper.map_old_to_new(&old_ids);
        assert_eq!(new_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_map_with_missing_ids() {
        let mapper = create_test_mapper();
        let new_ids = vec![1, 999, 2]; // 999 doesn't exist
        let old_ids = mapper.map_new_to_old(&new_ids);
        assert_eq!(old_ids, vec![100, 200]); // 999 is filtered out
    }

    #[test]
    fn test_stats() {
        let mapper = create_test_mapper();
        let stats = mapper.stats();
        assert_eq!(stats.new_vocab_size, 4);
        assert_eq!(stats.old_vocab_size, 5);
        assert_eq!(stats.mapped_tokens, 4);
        assert_eq!(stats.common_tokens, 2);
        assert_eq!(stats.replacement_char_tokens, 1);
    }
}
