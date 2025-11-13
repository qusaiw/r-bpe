//! Example: Complete encode/decode cycle with R-BPE tokenizer

use rbpe_tokenizers::model::RBPEModel;
use rbpe_tokenizers::normalizer::RBPENormalizer;
use rbpe_tokenizers::pretokenizer::RBPEPreTokenizer;
use rbpe_tokenizers::utils::UnicodeRangeChecker;
use rbpe_tokenizers::utils::unicode_ranges::ranges;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("R-BPE Tokenizer - Rust Implementation Demo");
    println!("==========================================\n");
    
    // Setup paths
    let base_path = Path::new("../rbpe_tokenizer");
    
    // Create pre-tokenizer with Arabic detection
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    // Load normalizer (optional)
    let normalizer_path = Path::new("arabic_normalization_map.json");
    let normalizer = if normalizer_path.exists() {
        println!("✓ Loading Arabic normalizer");
        Some(RBPENormalizer::from_json_file(normalizer_path)?)
    } else {
        println!("  (Normalizer not found, skipping)");
        None
    };
    
    // Load R-BPE model
    println!("✓ Loading R-BPE model...");
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        normalizer,
    )?;
    
    // Display stats
    let stats = model.vocab_mapper().stats();
    println!("✓ Model loaded successfully!\n");
    println!("Vocabulary Statistics:");
    println!("  New vocab size: {} tokens", stats.new_vocab_size);
    println!("  Old vocab size: {} tokens (mapped)", stats.old_vocab_size);
    println!("  Common tokens: {}", stats.common_tokens);
    println!();
    
    // Test cases
    let test_cases = vec![
        "Hello World",
        "مرحبا",
        "Hello مرحبا World",
        "كتاب book مجلة magazine",
    ];
    
    println!("Testing Encode/Decode:");
    println!("{}", "=".repeat(70));
    
    for text in test_cases {
        println!("\nInput:  '{}'", text);
        
        // Encode
        let ids = model.encode(text, false)?;
        println!("Tokens: {} tokens", ids.len());
        println!("IDs:    {:?}", &ids[..ids.len().min(10)]);
        if ids.len() > 10 {
            println!("        ... ({} more)", ids.len() - 10);
        }
        
        // Decode
        let decoded = model.decode(&ids, false)?;
        println!("Output: '{}'", decoded);
        
        // Verify
        if decoded.trim() == text.trim() {
            println!("✓ Perfect match!");
        } else if decoded.contains(text) || text.contains(&decoded) {
            println!("✓ Partial match (normalization may have occurred)");
        } else {
            println!("⚠ Output differs (may be due to normalization)");
        }
    }
    
    println!("\n{}", "=".repeat(70));
    println!("✓ Demo complete!");
    
    Ok(())
}
