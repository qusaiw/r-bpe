//! Python parity tests - verify exact token-by-token matching with Python implementation
//!
//! This test suite loads reference outputs from the Python tokenizer and verifies
//! that the Rust implementation produces identical results.

use rbpe_tokenizers::model::RBPEModel;
use rbpe_tokenizers::pretokenizer::RBPEPreTokenizer;
use rbpe_tokenizers::utils::UnicodeRangeChecker;
use rbpe_tokenizers::utils::unicode_ranges::ranges;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Deserialize, Serialize)]
struct TestCase {
    name: String,
    input: String,
    token_ids: Vec<u32>,
    num_tokens: usize,
    basic_decoded: String,
    advanced_decoded: String,
    has_replacement_basic: bool,
    has_replacement_advanced: bool,
    matches_input: bool,
}

fn load_test_cases() -> Vec<TestCase> {
    let json_path = Path::new("tests/python_reference_outputs.json");
    let json_content = std::fs::read_to_string(json_path)
        .expect("Failed to read python_reference_outputs.json");
    serde_json::from_str(&json_content)
        .expect("Failed to parse python_reference_outputs.json")
}

fn create_model() -> RBPEModel {
    let base_path = Path::new("../rbpe_tokenizer");
    
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        None,
    ).expect("Failed to load model")
}

#[test]
fn test_all_python_reference_cases() {
    let model = create_model();
    let test_cases = load_test_cases();
    
    println!("\nRunning {} Python parity tests:", test_cases.len());
    println!("{}", "=".repeat(80));
    
    let mut passed = 0;
    let mut failed = 0;
    let mut failures = Vec::new();
    
    for test_case in &test_cases {
        print!("\n{}: ", test_case.name);
        
        // Encode
        let rust_ids = model.encode(&test_case.input, false)
            .expect("Encoding failed");
        
        // Compare token IDs
        let ids_match = rust_ids == test_case.token_ids;
        
        // Decode with basic method
        let rust_basic_decoded = model.decode(&rust_ids, false)
            .expect("Basic decoding failed");
        
        // Decode with advanced method
        let rust_advanced_decoded = model.decode_advanced(&rust_ids, false)
            .expect("Advanced decoding failed");
        
        // Compare decodings
        let basic_decode_match = rust_basic_decoded == test_case.basic_decoded;
        let advanced_decode_match = rust_advanced_decoded == test_case.advanced_decoded;
        
        if ids_match && basic_decode_match && advanced_decode_match {
            println!("✓ PASS");
            passed += 1;
        } else {
            println!("✗ FAIL");
            failed += 1;
            
            let mut failure_msg = format!("Test '{}' failed:\n", test_case.name);
            
            if !ids_match {
                failure_msg.push_str(&format!(
                    "  Token IDs mismatch:\n    Python:  {:?}\n    Rust:    {:?}\n",
                    test_case.token_ids, rust_ids
                ));
            }
            
            if !basic_decode_match {
                failure_msg.push_str(&format!(
                    "  Basic decode mismatch:\n    Python:  '{}'\n    Rust:    '{}'\n",
                    test_case.basic_decoded, rust_basic_decoded
                ));
            }
            
            if !advanced_decode_match {
                failure_msg.push_str(&format!(
                    "  Advanced decode mismatch:\n    Python:  '{}'\n    Rust:    '{}'\n",
                    test_case.advanced_decoded, rust_advanced_decoded
                ));
            }
            
            failures.push(failure_msg);
        }
    }
    
    println!("\n{}", "=".repeat(80));
    println!("Results: {} passed, {} failed", passed, failed);
    
    if !failures.is_empty() {
        println!("\nFailure details:");
        for failure in &failures {
            println!("{}", failure);
        }
    }
    
    assert_eq!(failed, 0, "Some tests failed - see details above");
}

#[test]
fn test_specific_cases_detailed() {
    let model = create_model();
    let test_cases = load_test_cases();
    
    // Test a few specific cases with detailed output
    let interesting_cases = vec!["pure_english_short", "pure_arabic_short", "mixed_simple"];
    
    println!("\nDetailed testing of specific cases:");
    println!("{}", "=".repeat(80));
    
    for case_name in interesting_cases {
        let test_case = test_cases.iter()
            .find(|tc| tc.name == case_name)
            .expect(&format!("Test case '{}' not found", case_name));
        
        println!("\nTest: {}", test_case.name);
        println!("Input: '{}'", test_case.input);
        
        // Encode
        let rust_ids = model.encode(&test_case.input, false)
            .expect("Encoding failed");
        
        println!("\nToken IDs:");
        println!("  Python: {:?}", test_case.token_ids);
        println!("  Rust:   {:?}", rust_ids);
        println!("  Match:  {}", if rust_ids == test_case.token_ids { "✓" } else { "✗" });
        
        // Decode
        let rust_decoded = model.decode(&rust_ids, false)
            .expect("Decoding failed");
        
        println!("\nDecoded:");
        println!("  Python: '{}'", test_case.basic_decoded);
        println!("  Rust:   '{}'", rust_decoded);
        println!("  Match:  {}", if rust_decoded == test_case.basic_decoded { "✓" } else { "✗" });
        
        assert_eq!(rust_ids, test_case.token_ids, "Token IDs should match");
        assert_eq!(rust_decoded, test_case.basic_decoded, "Decoded text should match");
    }
}

#[test]
fn test_encode_statistics() {
    let model = create_model();
    let test_cases = load_test_cases();
    
    println!("\nEncoding statistics:");
    println!("{}", "=".repeat(80));
    
    let mut total_python_tokens = 0;
    let mut total_rust_tokens = 0;
    let mut exact_matches = 0;
    
    for test_case in &test_cases {
        let rust_ids = model.encode(&test_case.input, false)
            .expect("Encoding failed");
        
        total_python_tokens += test_case.num_tokens;
        total_rust_tokens += rust_ids.len();
        
        if rust_ids.len() == test_case.num_tokens && rust_ids == test_case.token_ids {
            exact_matches += 1;
        }
    }
    
    println!("Total test cases: {}", test_cases.len());
    println!("Python total tokens: {}", total_python_tokens);
    println!("Rust total tokens: {}", total_rust_tokens);
    println!("Exact matches: {}/{}", exact_matches, test_cases.len());
    
    let match_rate = (exact_matches as f64 / test_cases.len() as f64) * 100.0;
    println!("Match rate: {:.1}%", match_rate);
    
    assert_eq!(
        exact_matches, test_cases.len(),
        "All test cases should match exactly"
    );
}

#[test]
fn test_decode_statistics() {
    let model = create_model();
    let test_cases = load_test_cases();
    
    println!("\nDecoding statistics:");
    println!("{}", "=".repeat(80));
    
    let mut basic_matches = 0;
    let mut advanced_matches = 0;
    let mut perfect_roundtrips = 0;
    
    for test_case in &test_cases {
        let rust_ids = model.encode(&test_case.input, false)
            .expect("Encoding failed");
        
        let rust_basic = model.decode(&rust_ids, false)
            .expect("Basic decode failed");
        let rust_advanced = model.decode_advanced(&rust_ids, false)
            .expect("Advanced decode failed");
        
        if rust_basic == test_case.basic_decoded {
            basic_matches += 1;
        }
        
        if rust_advanced == test_case.advanced_decoded {
            advanced_matches += 1;
        }
        
        if rust_advanced.trim() == test_case.input.trim() {
            perfect_roundtrips += 1;
        }
    }
    
    println!("Total test cases: {}", test_cases.len());
    println!("Basic decode matches: {}/{}", basic_matches, test_cases.len());
    println!("Advanced decode matches: {}/{}", advanced_matches, test_cases.len());
    println!("Perfect roundtrips: {}/{}", perfect_roundtrips, test_cases.len());
    
    let basic_rate = (basic_matches as f64 / test_cases.len() as f64) * 100.0;
    let advanced_rate = (advanced_matches as f64 / test_cases.len() as f64) * 100.0;
    
    println!("Basic match rate: {:.1}%", basic_rate);
    println!("Advanced match rate: {:.1}%", advanced_rate);
    
    assert_eq!(
        basic_matches, test_cases.len(),
        "All basic decodes should match"
    );
    assert_eq!(
        advanced_matches, test_cases.len(),
        "All advanced decodes should match"
    );
}
