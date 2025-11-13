//! Integration tests for R-BPE tokenizer

use rbpe_tokenizers::model::RBPEModel;
use rbpe_tokenizers::normalizer::RBPENormalizer;
use rbpe_tokenizers::pretokenizer::RBPEPreTokenizer;
use rbpe_tokenizers::utils::UnicodeRangeChecker;
use rbpe_tokenizers::utils::unicode_ranges::ranges;
use std::path::Path;

#[test]
fn test_load_model_from_files() {
    let base_path = Path::new("../rbpe_tokenizer");
    
    // Create pre-tokenizer
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    // Load normalizer
    let normalizer_path = Path::new("arabic_normalization_map.json");
    let normalizer = if normalizer_path.exists() {
        Some(RBPENormalizer::from_json_file(normalizer_path).unwrap())
    } else {
        None
    };
    
    // Load model
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        normalizer,
    ).expect("Failed to load model");
    
    // Test encoding
    let text = "Hello مرحبا World";
    let ids = model.encode(text, false).expect("Encoding failed");
    
    println!("Encoded '{}' to {} tokens", text, ids.len());
    assert!(!ids.is_empty(), "Encoding should produce tokens");
    
    // Test decoding
    let decoded = model.decode(&ids, false).expect("Decoding failed");
    println!("Decoded back to: '{}'", decoded);
    
    // Check that we get reasonable output (may not be exact due to normalization)
    assert!(decoded.contains("Hello") || decoded.contains("مرحبا"),
            "Decoded text should contain some of the original content");
}

#[test]
fn test_encode_pure_arabic() {
    let base_path = Path::new("../rbpe_tokenizer");
    
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        None,
        pretokenizer,
        None,
    ).expect("Failed to load model");
    
    let text = "مرحبا";
    let ids = model.encode(text, false).expect("Encoding failed");
    
    println!("Encoded pure Arabic '{}' to {} tokens: {:?}", text, ids.len(), ids);
    assert!(!ids.is_empty(), "Should produce tokens");
    
    let decoded = model.decode(&ids, false).expect("Decoding failed");
    println!("Decoded: '{}'", decoded);
}

#[test]
fn test_encode_pure_english() {
    let base_path = Path::new("../rbpe_tokenizer");
    
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        None,
        pretokenizer,
        None,
    ).expect("Failed to load model");
    
    let text = "Hello World";
    let ids = model.encode(text, false).expect("Encoding failed");
    
    println!("Encoded English '{}' to {} tokens: {:?}", text, ids.len(), ids);
    assert!(!ids.is_empty(), "Should produce tokens");
    
    let decoded = model.decode(&ids, false).expect("Decoding failed");
    println!("Decoded: '{}'", decoded);
    assert_eq!(decoded.trim(), text.trim(), "Decoded text should match original");
}

#[test]
fn test_compare_with_python() {
    let base_path = Path::new("../rbpe_tokenizer");
    
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        None,
        pretokenizer,
        None,
    ).expect("Failed to load model");
    
    // Test case 1: Pure English
    let text1 = "Hello World";
    let ids1 = model.encode(text1, false).expect("Encoding failed");
    println!("Rust - Text: '{}', IDs: {:?}", text1, ids1);
    // Python gives: [9906, 4435]
    assert_eq!(ids1.len(), 2, "Should produce 2 tokens for 'Hello World'");
    
    // Test case 2: Pure Arabic
    let text2 = "مرحبا";
    let ids2 = model.encode(text2, false).expect("Encoding failed");
    println!("Rust - Text: '{}', IDs: {:?}", text2, ids2);
    // Python gives: [122627, 5821]
    assert_eq!(ids2.len(), 2, "Should produce 2 tokens for 'مرحبا'");
    
    // Test case 3: Mixed
    let text3 = "Hello مرحبا World";
    let ids3 = model.encode(text3, false).expect("Encoding failed");
    println!("Rust - Text: '{}', IDs: {:?}", text3, ids3);
    // Python gives: [9906, 220, 122627, 5821, 220, 10343]
    assert_eq!(ids3.len(), 6, "Should produce 6 tokens for mixed text");
    
    // Verify decoding
    let decoded1 = model.decode(&ids1, false).expect("Decoding failed");
    let decoded2 = model.decode(&ids2, false).expect("Decoding failed");
    let decoded3 = model.decode(&ids3, false).expect("Decoding failed");
    
    println!("Decoded 1: '{}'", decoded1);
    println!("Decoded 2: '{}'", decoded2);
    println!("Decoded 3: '{}'", decoded3);
    
    assert_eq!(decoded1.trim(), text1);
    assert_eq!(decoded3.trim(), text3);
}

#[test]
fn test_advanced_decoder_with_basic_text() {
    let base_path = Path::new("../rbpe_tokenizer");
    
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        None,
    ).expect("Failed to load model");
    
    // Test case 1: Pure English (should work the same with basic or advanced)
    let text1 = "Hello World";
    let ids1 = model.encode(text1, false).expect("Encoding failed");
    
    let basic_decoded = model.decode(&ids1, false).expect("Basic decoding failed");
    let advanced_decoded = model.decode_advanced(&ids1, false).expect("Advanced decoding failed");
    
    println!("Basic:    '{}'", basic_decoded);
    println!("Advanced: '{}'", advanced_decoded);
    
    assert_eq!(basic_decoded.trim(), text1);
    assert_eq!(advanced_decoded.trim(), text1);
    assert_eq!(basic_decoded, advanced_decoded, "Basic and advanced decoders should produce same output for simple text");
}

#[test]
fn test_advanced_decoder_with_arabic() {
    let base_path = Path::new("../rbpe_tokenizer");
    
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        None,
    ).expect("Failed to load model");
    
    // Test case 2: Pure Arabic
    let text2 = "مرحبا بك في عالم البرمجة";
    let ids2 = model.encode(text2, false).expect("Encoding failed");
    
    let basic_decoded = model.decode(&ids2, false).expect("Basic decoding failed");
    let advanced_decoded = model.decode_advanced(&ids2, false).expect("Advanced decoding failed");
    
    println!("Original: '{}'", text2);
    println!("Basic:    '{}'", basic_decoded);
    println!("Advanced: '{}'", advanced_decoded);
    
    // Both should decode without replacement characters
    assert!(!basic_decoded.contains('�'), "Basic decoder should not produce replacement characters");
    assert!(!advanced_decoded.contains('�'), "Advanced decoder should not produce replacement characters");
}

#[test]
fn test_advanced_decoder_mixed_content() {
    let base_path = Path::new("../rbpe_tokenizer");
    
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        None,
    ).expect("Failed to load model");
    
    // Test mixed content with numbers and punctuation
    let test_cases = vec![
        "Hello مرحبا World",
        "The year is 2024 and السنة هي ٢٠٢٤",
        "This is a test: هذا اختبار!",
        "مرحبا! Hello! 你好!",
    ];
    
    for text in test_cases {
        println!("\nTesting: '{}'", text);
        let ids = model.encode(text, false).expect("Encoding failed");
        println!("Encoded to {} tokens", ids.len());
        
        let basic_decoded = model.decode(&ids, false).expect("Basic decoding failed");
        let advanced_decoded = model.decode_advanced(&ids, false).expect("Advanced decoding failed");
        
        println!("Basic:    '{}'", basic_decoded);
        println!("Advanced: '{}'", advanced_decoded);
        
        // Advanced decoder should handle any replacement characters better
        let basic_has_replacement = basic_decoded.contains('�');
        let advanced_has_replacement = advanced_decoded.contains('�');
        
        if basic_has_replacement {
            println!("  ⚠️  Basic decoder produced replacement characters");
        }
        if advanced_has_replacement {
            println!("  ⚠️  Advanced decoder produced replacement characters");
        }
        
        // Advanced should have fewer or equal replacement characters
        assert!(
            advanced_decoded.matches('�').count() <= basic_decoded.matches('�').count(),
            "Advanced decoder should not produce more replacement characters than basic decoder"
        );
    }
}

#[test]
fn test_empty_and_edge_cases() {
    let base_path = Path::new("../rbpe_tokenizer");
    
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        None,
    ).expect("Failed to load model");
    
    // Test edge cases
    let edge_cases = vec![
        "",                    // Empty string
        " ",                   // Single space
        "  ",                  // Multiple spaces
        "\n",                  // Newline
        "\t",                  // Tab
        "a",                   // Single char
        "ا",                   // Single Arabic char
    ];
    
    for text in edge_cases {
        println!("\nTesting edge case: {:?}", text);
        let ids = model.encode(text, false).expect("Encoding failed");
        
        if ids.is_empty() {
            println!("  Empty encoding");
            continue;
        }
        
        let basic_decoded = model.decode(&ids, false).expect("Basic decoding failed");
        let advanced_decoded = model.decode_advanced(&ids, false).expect("Advanced decoding failed");
        
        println!("  Original:  {:?}", text);
        println!("  Basic:     {:?}", basic_decoded);
        println!("  Advanced:  {:?}", advanced_decoded);
        
        // Both should produce valid output
        assert!(!basic_decoded.is_empty() || ids.is_empty(), "Decoder should produce output for non-empty input");
    }
}
