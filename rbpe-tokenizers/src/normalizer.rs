//! Unicode normalization for R-BPE
//!
//! This module implements the R-BPE Arabic normalization logic,
//! which applies character replacements based on a normalization map.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// R-BPE Unicode Normalizer
///
/// Applies character normalization according to loaded mapping rules.
/// Only normalizes characters within multi-character tokens to preserve
/// single-character semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RBPENormalizer {
    /// Map from Unicode code point to replacement code point(s)
    /// Value is Vec because some characters expand to multiple characters
    normalization_map: HashMap<u32, Vec<u32>>,
}

impl RBPENormalizer {
    /// Create a new normalizer with the given normalization map
    pub fn new(normalization_map: HashMap<u32, Vec<u32>>) -> Self {
        Self { normalization_map }
    }

    /// Load normalizer from JSON file
    pub fn from_json_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let map_str: HashMap<String, Vec<u32>> = serde_json::from_reader(file)?;
        
        // Convert string keys to u32
        let normalization_map = map_str
            .into_iter()
            .filter_map(|(k, v)| k.parse::<u32>().ok().map(|k_int| (k_int, v)))
            .collect();
        
        Ok(Self::new(normalization_map))
    }

    /// Create an empty normalizer (no normalization applied)
    pub fn identity() -> Self {
        Self {
            normalization_map: HashMap::new(),
        }
    }

    /// Normalize a single character
    #[inline]
    fn normalize_char(&self, ch: char) -> Option<&[u32]> {
        self.normalization_map.get(&(ch as u32)).map(|v| v.as_slice())
    }

    /// Normalize text according to R-BPE rules:
    /// - Whitespace and single characters are kept as-is
    /// - Multi-character tokens have normalization applied
    pub fn normalize(&self, text: &str) -> String {
        if self.normalization_map.is_empty() {
            return text.to_string();
        }

        let mut result = String::with_capacity(text.len());
        let mut current_token = String::new();
        let mut is_whitespace_token = false;

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !is_whitespace_token && !current_token.is_empty() {
                    // Flush non-whitespace token
                    result.push_str(&self.normalize_token(&current_token));
                    current_token.clear();
                }
                current_token.push(ch);
                is_whitespace_token = true;
            } else {
                if is_whitespace_token && !current_token.is_empty() {
                    // Flush whitespace token without normalization
                    result.push_str(&current_token);
                    current_token.clear();
                }
                current_token.push(ch);
                is_whitespace_token = false;
            }
        }

        // Flush final token
        if !current_token.is_empty() {
            if is_whitespace_token {
                result.push_str(&current_token);
            } else {
                result.push_str(&self.normalize_token(&current_token));
            }
        }

        result
    }

    /// Normalize a single token
    /// Single-character tokens and whitespace are kept as-is
    fn normalize_token(&self, token: &str) -> String {
        // Single character or whitespace - no normalization
        if token.chars().count() <= 1 || token.chars().all(|c| c.is_whitespace()) {
            return token.to_string();
        }

        let mut result = String::with_capacity(token.len());
        
        for ch in token.chars() {
            if let Some(replacements) = self.normalize_char(ch) {
                // Apply replacement
                for &code_point in replacements {
                    if let Some(replacement_char) = char::from_u32(code_point) {
                        result.push(replacement_char);
                    } else {
                        // Invalid code point, keep original
                        result.push(ch);
                    }
                }
            } else {
                // No replacement, keep original
                result.push(ch);
            }
        }

        result
    }

    /// Get the number of normalization rules
    pub fn num_rules(&self) -> usize {
        self.normalization_map.len()
    }

    /// Check if normalizer has any rules
    pub fn is_empty(&self) -> bool {
        self.normalization_map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_normalizer() {
        let normalizer = RBPENormalizer::identity();
        assert_eq!(normalizer.normalize("Hello مرحبا"), "Hello مرحبا");
    }

    #[test]
    fn test_single_char_not_normalized() {
        let mut map = HashMap::new();
        map.insert('ٯ' as u32, vec!['ق' as u32]); // ٯ -> ق
        let normalizer = RBPENormalizer::new(map);
        
        // Single character - should not be normalized
        assert_eq!(normalizer.normalize("ٯ"), "ٯ");
        
        // Multi-character - should be normalized
        assert_eq!(normalizer.normalize("ٯا"), "قا");
    }

    #[test]
    fn test_whitespace_preserved() {
        let mut map = HashMap::new();
        map.insert('ٯ' as u32, vec!['ق' as u32]);
        let normalizer = RBPENormalizer::new(map);
        
        assert_eq!(normalizer.normalize("  "), "  ");
        assert_eq!(normalizer.normalize("\t\n"), "\t\n");
        assert_eq!(normalizer.normalize("ٯا ٯا"), "قا قا");
    }

    #[test]
    fn test_multi_char_replacement() {
        let mut map = HashMap::new();
        // ٷ -> ؤُ (multi-character replacement)
        map.insert(0x0677, vec![0x0624, 0x064F]);
        let normalizer = RBPENormalizer::new(map);
        
        // Single char - no normalization
        assert_eq!(normalizer.normalize("ٷ"), "ٷ");
        
        // Multi-char token - apply normalization
        let input = "اٷا"; // includes ٷ
        let expected = "اؤُا"; // ٷ becomes ؤُ
        assert_eq!(normalizer.normalize(input), expected);
    }

    #[test]
    fn test_arabic_normalization_samples() {
        let mut map = HashMap::new();
        map.insert(0x066F, vec![0x0642]); // ٯ -> ق
        map.insert(0x0672, vec![0x0623]); // ٲ -> أ
        map.insert(0x06A9, vec![0x0643]); // ک -> ك
        
        let normalizer = RBPENormalizer::new(map);
        
        // Test various cases
        assert_eq!(normalizer.normalize("مٯحبا"), "مقحبا");
        assert_eq!(normalizer.normalize("ٲکرم"), "أكرم");
    }

    #[test]
    fn test_mixed_text() {
        let mut map = HashMap::new();
        map.insert(0x06A9, vec![0x0643]); // ک -> ك
        
        let normalizer = RBPENormalizer::new(map);
        
        // Mixed Arabic/English
        assert_eq!(
            normalizer.normalize("Hello کتاب world"),
            "Hello كتاب world"
        );
    }
}
