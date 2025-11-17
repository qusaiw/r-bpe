//! Example: Demonstrating the advanced decoder with replacement character handling

use rbpe_tokenizers::model::RBPEModel;
use rbpe_tokenizers::pretokenizer::RBPEPreTokenizer;
use rbpe_tokenizers::utils::UnicodeRangeChecker;
use rbpe_tokenizers::utils::unicode_ranges::ranges;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("R-BPE Advanced Decoder Demo");
    println!("============================\n");
    
    // Setup paths
    let base_path = Path::new("../rbpe_tokenizer");
    
    // Create pre-tokenizer with Arabic detection
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    // Load R-BPE model
    println!("Loading R-BPE model...");
    let model = RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        None,
    )?;
    println!("✓ Model loaded\n");
    
    // Test cases - focusing on mixed content and edge cases
    let test_cases = vec![
        ("Pure English", "Hello World! How are you today?"),
        ("Pure Arabic", "مرحبا! كيف حالك اليوم؟"),
        ("Mixed Languages", "Hello مرحبا World عالم!"),
        ("With Numbers", "The year is 2024 السنة ٢٠٢٤"),
        ("With Punctuation", "Hello! مرحبا؟ World... عالم!"),
        ("Code-Switching", "This is a test هذا اختبار متعدد اللغات multilingual"),
        ("Special Characters", "Email: test@example.com البريد: مثال@نطاق.com"),
    ];
    
    println!("Comparing Basic vs Advanced Decoder:");
    println!("{}", "=".repeat(80));
    
    for (label, text) in test_cases {
        println!("\n{}", label);
        println!("{}", "-".repeat(80));
        println!("Input: {}", text);
        
        // Encode
        let ids = model.encode(text, false)?;
        println!("\nEncoded to {} tokens", ids.len());
        
        // Basic decode (fast but may have replacement chars)
        let basic_decoded = model.decode_basic(&ids, false)?;
        let basic_has_replacement = basic_decoded.contains('�');
        
        // Standard decode (with automatic replacement handling)
        let decoded = model.decode(&ids, false)?;
        let has_replacement = decoded.contains('�');
        
        println!("\nBasic Decoder (fast path):");
        println!("  Output: {}", basic_decoded);
        if basic_has_replacement {
            let count = basic_decoded.matches('�').count();
            println!("  ⚠️  Contains {} replacement character(s)", count);
        } else {
            println!("  ✓ No replacement characters");
        }
        
        println!("\nStandard Decoder (recommended):");
        println!("  Output: {}", decoded);
        if has_replacement {
            let count = decoded.matches('�').count();
            println!("  ⚠️  Contains {} replacement character(s)", count);
        } else {
            println!("  ✓ No replacement characters");
        }
        
        // Compare
        if basic_decoded == decoded {
            println!("\n✓ Both decoders produced identical output");
        } else {
            println!("\n⚠️  Decoders produced different outputs");
            if !basic_has_replacement && !has_replacement {
                println!("   (Both valid, but using different decoding paths)");
            } else if basic_has_replacement && !has_replacement {
                println!("   ✓ Standard decoder successfully handled replacement characters!");
            }
        }
        
        // Verify match with original
        if decoded.trim() == text.trim() {
            println!("✓ Perfect match with original input");
        } else {
            println!("⚠️  Output differs from input (may be due to tokenization artifacts)");
        }
    }
    
    println!("\n{}", "=".repeat(80));
    
    // Demonstrate sliding window algorithm
    println!("\nSliding Window Algorithm Benefits:");
    println!("{}", "-".repeat(80));
    println!("The advanced decoder uses a sliding window algorithm to handle cases where");
    println!("UTF-8 byte sequences are split across multiple tokens. This can happen when:");
    println!("  1. Multi-byte Unicode characters are tokenized as individual bytes");
    println!("  2. Token boundaries don't align with character boundaries");
    println!("  3. Language transitions cause unusual tokenization patterns");
    println!();
    println!("The algorithm tries windows of 1-4 tokens to find complete UTF-8 sequences,");
    println!("testing both the new and old tokenizers to find the best decoding.");
    
    println!("\n✓ Demo complete!");
    
    Ok(())
}
