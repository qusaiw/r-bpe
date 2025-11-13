//! Unicode range detection for language segmentation
//!
//! This module provides utilities for detecting whether characters belong to
//! target language scripts based on Unicode code point ranges.

use serde::{Deserialize, Serialize};

/// A Unicode code point range (inclusive)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnicodeRange {
    pub start: u32,
    pub end: u32,
}

impl UnicodeRange {
    /// Create a new Unicode range
    pub fn new(start: u32, end: u32) -> Self {
        assert!(start <= end, "Invalid range: start must be <= end");
        Self { start, end }
    }

    /// Check if a code point is within this range
    #[inline]
    pub fn contains(&self, code_point: u32) -> bool {
        code_point >= self.start && code_point <= self.end
    }
}

/// Checker for target language detection based on Unicode ranges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnicodeRangeChecker {
    ranges: Vec<UnicodeRange>,
}

impl UnicodeRangeChecker {
    /// Create a new checker with the given Unicode ranges
    pub fn new(ranges: Vec<UnicodeRange>) -> Self {
        Self { ranges }
    }

    /// Check if a character belongs to any of the target language ranges
    #[inline]
    pub fn is_target(&self, ch: char) -> bool {
        let code_point = ch as u32;
        self.ranges.iter().any(|range| range.contains(code_point))
    }

    /// Check if any character in the text belongs to target language ranges
    pub fn contains_target(&self, text: &str) -> bool {
        text.chars().any(|ch| self.is_target(ch))
    }

    /// Get all ranges
    pub fn ranges(&self) -> &[UnicodeRange] {
        &self.ranges
    }
}

/// Common Unicode ranges for various scripts
pub mod ranges {
    use super::UnicodeRange;

    // Arabic script ranges
    pub const ARABIC: UnicodeRange = UnicodeRange { start: 0x0600, end: 0x06FF };
    pub const ARABIC_SUPPLEMENT: UnicodeRange = UnicodeRange { start: 0x0750, end: 0x077F };
    pub const ARABIC_EXTENDED_A: UnicodeRange = UnicodeRange { start: 0x08A0, end: 0x08FF };
    pub const ARABIC_EXTENDED_B: UnicodeRange = UnicodeRange { start: 0x0870, end: 0x089F };
    pub const ARABIC_PRESENTATION_FORMS_A: UnicodeRange = UnicodeRange { start: 0xFB50, end: 0xFDFF };
    pub const ARABIC_PRESENTATION_FORMS_B: UnicodeRange = UnicodeRange { start: 0xFE70, end: 0xFEFF };

    // Latin script ranges
    pub const BASIC_LATIN: UnicodeRange = UnicodeRange { start: 0x0000, end: 0x007F };
    pub const LATIN_1_SUPPLEMENT: UnicodeRange = UnicodeRange { start: 0x0080, end: 0x00FF };
    pub const LATIN_EXTENDED_A: UnicodeRange = UnicodeRange { start: 0x0100, end: 0x017F };
    pub const LATIN_EXTENDED_B: UnicodeRange = UnicodeRange { start: 0x0180, end: 0x024F };

    // Greek ranges
    pub const GREEK: UnicodeRange = UnicodeRange { start: 0x0370, end: 0x03FF };
    pub const GREEK_EXTENDED: UnicodeRange = UnicodeRange { start: 0x1F00, end: 0x1FFF };

    /// Get all Arabic script ranges
    pub fn all_arabic() -> Vec<UnicodeRange> {
        vec![
            ARABIC,
            ARABIC_SUPPLEMENT,
            ARABIC_EXTENDED_A,
            ARABIC_EXTENDED_B,
            ARABIC_PRESENTATION_FORMS_A,
            ARABIC_PRESENTATION_FORMS_B,
        ]
    }

    /// Get all Latin script ranges
    pub fn all_latin() -> Vec<UnicodeRange> {
        vec![
            BASIC_LATIN,
            LATIN_1_SUPPLEMENT,
            LATIN_EXTENDED_A,
            LATIN_EXTENDED_B,
        ]
    }

    /// Get all Greek script ranges
    pub fn all_greek() -> Vec<UnicodeRange> {
        vec![GREEK, GREEK_EXTENDED]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unicode_range_contains() {
        let range = UnicodeRange::new(0x0600, 0x06FF);
        assert!(range.contains(0x0600));
        assert!(range.contains(0x0650));
        assert!(range.contains(0x06FF));
        assert!(!range.contains(0x05FF));
        assert!(!range.contains(0x0700));
    }

    #[test]
    fn test_arabic_detection() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        
        // Arabic characters
        assert!(checker.is_target('ا')); // Arabic letter Alef
        assert!(checker.is_target('ب')); // Arabic letter Beh
        assert!(checker.is_target('م')); // Arabic letter Meem
        
        // Non-Arabic characters
        assert!(!checker.is_target('a'));
        assert!(!checker.is_target('A'));
        assert!(!checker.is_target('1'));
        assert!(!checker.is_target(' '));
    }

    #[test]
    fn test_mixed_text() {
        let checker = UnicodeRangeChecker::new(ranges::all_arabic());
        
        assert!(checker.contains_target("مرحبا Hello"));
        assert!(checker.contains_target("Hello مرحبا"));
        assert!(!checker.contains_target("Hello World"));
        assert!(checker.contains_target("مرحبا"));
    }

    #[test]
    fn test_latin_detection() {
        let checker = UnicodeRangeChecker::new(ranges::all_latin());
        
        assert!(checker.is_target('a'));
        assert!(checker.is_target('Z'));
        assert!(checker.is_target('é'));
        assert!(!checker.is_target('ا'));
    }
}
