//! Pre-tokenization with language-aware segmentation
//!
//! This module implements the R-BPE pre-tokenizer that segments text
//! based on language scripts (e.g., Arabic vs. Latin).

use crate::utils::UnicodeRangeChecker;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// A text segment with its language classification
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub text: String,
    pub is_target: bool,
    pub is_special_token: bool,
}

/// R-BPE Pre-tokenizer
///
/// Segments text based on:
/// 1. Special tokens (if configured)
/// 2. Target language detection (via Unicode ranges)
/// 3. Whitespace handling (whitespace joins current segment)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RBPEPreTokenizer {
    /// Unicode range checker for target language detection
    target_language_checker: UnicodeRangeChecker,
    
    /// Optional regex for special tokens
    #[serde(skip)]
    special_tokens_regex: Option<Regex>,
    
    /// Special token patterns (stored for serialization)
    special_tokens: Vec<String>,
}

impl RBPEPreTokenizer {
    /// Create a new pre-tokenizer
    pub fn new(
        target_language_checker: UnicodeRangeChecker,
        special_tokens: Vec<String>,
    ) -> Self {
        let special_tokens_regex = if !special_tokens.is_empty() {
            // Sort by length (descending) to match longest first
            let mut sorted_tokens = special_tokens.clone();
            sorted_tokens.sort_by(|a, b| b.len().cmp(&a.len()));
            
            // Escape and join with |
            let pattern = sorted_tokens
                .iter()
                .map(|t| regex::escape(t))
                .collect::<Vec<_>>()
                .join("|");
            
            Regex::new(&format!("({})", pattern)).ok()
        } else {
            None
        };

        Self {
            target_language_checker,
            special_tokens_regex,
            special_tokens,
        }
    }

    /// Segment text by special tokens first
    fn split_by_special_tokens<'a>(&self, text: &'a str) -> Vec<(&'a str, bool)> {
        if let Some(ref regex) = self.special_tokens_regex {
            let mut segments = Vec::new();
            let mut last_end = 0;

            for mat in regex.find_iter(text) {
                // Add text before match (non-special)
                if mat.start() > last_end {
                    segments.push((&text[last_end..mat.start()], false));
                }
                // Add special token
                segments.push((mat.as_str(), true));
                last_end = mat.end();
            }

            // Add remaining text
            if last_end < text.len() {
                segments.push((&text[last_end..], false));
            }

            segments
        } else {
            vec![(text, false)]
        }
    }

    /// Segment a single piece of text by language
    fn segment_by_language(&self, text: &str) -> Vec<Segment> {
        let mut segments = Vec::new();
        let mut current_segment = String::new();
        let mut is_current_target: Option<bool> = None;

        for ch in text.chars() {
            let is_char_target = self.target_language_checker.is_target(ch);

            // Whitespace joins current segment
            if ch.is_whitespace() {
                current_segment.push(ch);
                continue;
            }

            // First non-whitespace character
            if is_current_target.is_none() {
                is_current_target = Some(is_char_target);
                current_segment.push(ch);
                continue;
            }

            // Language switch
            if is_char_target != is_current_target.unwrap() {
                if !current_segment.is_empty() {
                    segments.push(Segment {
                        text: current_segment.clone(),
                        is_target: is_current_target.unwrap(),
                        is_special_token: false,
                    });
                }
                current_segment.clear();
                current_segment.push(ch);
                is_current_target = Some(is_char_target);
            } else {
                current_segment.push(ch);
            }
        }

        // Add final segment
        if !current_segment.is_empty() {
            if let Some(is_target) = is_current_target {
                segments.push(Segment {
                    text: current_segment,
                    is_target,
                    is_special_token: false,
                });
            } else {
                // Whitespace-only text - treat as non-target (use old tokenizer)
                segments.push(Segment {
                    text: current_segment,
                    is_target: false,
                    is_special_token: false,
                });
            }
        }

        segments
    }

    /// Pre-tokenize text into segments
    pub fn pre_tokenize(&self, text: &str) -> Vec<Segment> {
        let mut all_segments = Vec::new();

        // First split by special tokens
        for (segment_text, is_special) in self.split_by_special_tokens(text) {
            if is_special {
                // Special tokens are their own segment
                all_segments.push(Segment {
                    text: segment_text.to_string(),
                    is_target: false, // Special tokens use old tokenizer
                    is_special_token: true,
                });
            } else {
                // Regular text - segment by language
                all_segments.extend(self.segment_by_language(segment_text));
            }
        }

        all_segments
    }

    /// Get the target language checker
    pub fn target_language_checker(&self) -> &UnicodeRangeChecker {
        &self.target_language_checker
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::unicode_ranges::ranges;

    #[test]
    fn test_pure_arabic() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);

        let segments = pretokenizer.pre_tokenize("مرحبا");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "مرحبا");
        assert_eq!(segments[0].is_target, true);
        assert_eq!(segments[0].is_special_token, false);
    }

    #[test]
    fn test_pure_english() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);

        let segments = pretokenizer.pre_tokenize("Hello World");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello World");
        assert_eq!(segments[0].is_target, false);
    }

    #[test]
    fn test_mixed_text() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);

        let segments = pretokenizer.pre_tokenize("Hello مرحبا World");
        assert_eq!(segments.len(), 3);
        
        assert_eq!(segments[0].text, "Hello ");
        assert_eq!(segments[0].is_target, false);
        
        assert_eq!(segments[1].text, "مرحبا ");  // Whitespace stays with segment
        assert_eq!(segments[1].is_target, true);
        
        assert_eq!(segments[2].text, "World");
        assert_eq!(segments[2].is_target, false);
    }

    #[test]
    fn test_whitespace_joining() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);

        // Whitespace at start stays with first segment
        let segments = pretokenizer.pre_tokenize("  مرحبا");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "  مرحبا");
        assert_eq!(segments[0].is_target, true);

        // Whitespace in middle joins appropriate segment
        let segments = pretokenizer.pre_tokenize("Hello   World");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello   World");
        assert_eq!(segments[0].is_target, false);
    }

    #[test]
    fn test_special_tokens() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let special_tokens = vec!["<|start|>".to_string(), "<|end|>".to_string()];
        let pretokenizer = RBPEPreTokenizer::new(checker, special_tokens);

        let segments = pretokenizer.pre_tokenize("<|start|>مرحبا<|end|>");
        assert_eq!(segments.len(), 3);
        
        assert_eq!(segments[0].text, "<|start|>");
        assert_eq!(segments[0].is_special_token, true);
        
        assert_eq!(segments[1].text, "مرحبا");
        assert_eq!(segments[1].is_target, true);
        assert_eq!(segments[1].is_special_token, false);
        
        assert_eq!(segments[2].text, "<|end|>");
        assert_eq!(segments[2].is_special_token, true);
    }

    #[test]
    fn test_special_tokens_with_text() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let special_tokens = vec!["<extra_id_1>".to_string(), "<extra_id_10>".to_string()];
        let pretokenizer = RBPEPreTokenizer::new(checker, special_tokens);

        // Should match longest first
        let segments = pretokenizer.pre_tokenize("Hello <extra_id_10> World");
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[1].text, "<extra_id_10>");
        assert_eq!(segments[1].is_special_token, true);
    }

    #[test]
    fn test_arabic_english_multiple_switches() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);

        let segments = pretokenizer.pre_tokenize("كتاب book مجلة magazine");
        assert_eq!(segments.len(), 4);
        
        assert_eq!(segments[0].text, "كتاب ");  // Whitespace stays with segment
        assert_eq!(segments[0].is_target, true);
        
        assert_eq!(segments[1].text, "book ");  // Whitespace stays with segment
        assert_eq!(segments[1].is_target, false);
        
        assert_eq!(segments[2].text, "مجلة ");  // Whitespace stays with segment
        assert_eq!(segments[2].is_target, true);
        
        assert_eq!(segments[3].text, "magazine");
        assert_eq!(segments[3].is_target, false);
    }

    #[test]
    fn test_empty_string() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);

        let segments = pretokenizer.pre_tokenize("");
        assert_eq!(segments.len(), 0);
    }

    #[test]
    fn test_only_whitespace() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);

        let segments = pretokenizer.pre_tokenize("   \t\n  ");
        // Whitespace with no non-whitespace characters creates one segment (non-target)
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].is_target, false);
        assert_eq!(segments[0].text, "   \t\n  ");
    }
}
