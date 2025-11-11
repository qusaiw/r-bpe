//! Example: Using R-BPE with FastTokenizer-compatible API
//!
//! This example demonstrates how R-BPE works as a FastTokenizer-compatible
//! tokenizer while maintaining all its custom logic (dual tokenizers, mappings,
//! language routing).

use rbpe_tokenizers::RBPEFastTokenizer;
use rbpe_tokenizers::pretokenizer::RBPEPreTokenizer;
use rbpe_tokenizers::utils::UnicodeRangeChecker;
use rbpe_tokenizers::utils::unicode_ranges::ranges;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("R-BPE FastTokenizer Demo");
    println!("{}", "=".repeat(80));
    
    // Setup
    let base_path = Path::new("../rbpe_tokenizer");
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    println!("\nLoading R-BPE tokenizer...");
    let tokenizer = RBPEFastTokenizer::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        None,
    )?;
    
    println!("✓ Tokenizer loaded with full R-BPE logic!");
    println!("  - Dual tokenizers (new + old)");
    println!("  - Language-aware routing");
    println!("  - Vocabulary mapping");
    println!("  - Advanced decoding");
    
    // Test cases
    let test_cases = vec![
        ("Pure English", "Hello World!"),
        ("Pure Arabic", "مرحبا بالعالم!"),
        ("Mixed", "Hello مرحبا World عالم!"),
        ("Code-switching", "This is a test هذا اختبار"),
    ];
    
    println!("\n{}", "=".repeat(80));
    println!("Testing FastTokenizer API with R-BPE Logic:");
    println!("{}", "=".repeat(80));
    
    for (name, text) in &test_cases {
        println!("\n{}", name);
        println!("{}", "-".repeat(80));
        println!("Input: {}", text);
        
        // Encode using R-BPE logic
        let ids = tokenizer.encode_text(text, false)?;
        println!("\nEncoding (R-BPE does its magic internally):");
        println!("  Tokens: {}", ids.len());
        println!("  IDs:    {:?}", &ids[..ids.len().min(10)]);
        if ids.len() > 10 {
            println!("          ... ({} more)", ids.len() - 10);
        }
        
        // Decode using R-BPE basic decoder
        let decoded_basic = tokenizer.decode_ids(&ids, false)?;
        println!("\nBasic Decode:");
        println!("  Output: {}", decoded_basic);
        println!("  Match:  {}", if decoded_basic.trim() == *text { "✓" } else { "✗" });
        
        // Decode using R-BPE advanced decoder  
        let decoded_advanced = tokenizer.decode_ids_advanced(&ids, false)?;
        println!("\nAdvanced Decode:");
        println!("  Output: {}", decoded_advanced);
        println!("  Match:  {}", if decoded_advanced.trim() == *text { "✓" } else { "✗" });
        
        // Show that it creates Encoding objects
        let encoding = tokenizer.encode_to_encoding(text)?;
        println!("\nEncoding Object (tokenizers crate compatible):");
        println!("  IDs:     {:?}", &encoding.get_ids()[..encoding.get_ids().len().min(5)]);
        println!("  Tokens:  {:?}", &encoding.get_tokens()[..encoding.get_tokens().len().min(5)]);
        println!("  Length:  {}", encoding.len());
    }
    
    // Demonstrate batch encoding
    println!("\n{}", "=".repeat(80));
    println!("Batch Encoding:");
    println!("{}", "=".repeat(80));
    
    let batch_texts = vec!["Hello", "مرحبا", "World", "عالم"];
    let encodings = tokenizer.encode_batch(&batch_texts)?;
    
    println!("\nEncoded {} texts in batch:", batch_texts.len());
    for (i, (text, encoding)) in batch_texts.iter().zip(encodings.iter()).enumerate() {
        println!("  {}. '{}' -> {} tokens", i + 1, text, encoding.len());
    }
    
    // Show what makes R-BPE special
    println!("\n{}", "=".repeat(80));
    println!("What makes this R-BPE (not just a regular tokenizer):");
    println!("{}", "=".repeat(80));
    println!("
1. **Dual Tokenizer System**
   - Uses TWO BPE tokenizers internally (new + old)
   - Routes text segments to appropriate tokenizer based on language
   
2. **Language-Aware Segmentation**
   - Pre-tokenizes by detecting language (Arabic vs. other)
   - Each segment goes to the right tokenizer
   
3. **Vocabulary Mapping**
   - New tokenizer IDs are mapped back to old tokenizer space
   - Maintains compatibility while optimizing for target language
   
4. **Advanced Decoding**
   - Handles replacement characters with sliding window
   - Reconstructs UTF-8 sequences split across tokens
   
5. **Optimized for Arabic**
   - New tokenizer has 16K vocab focused on Arabic
   - Old tokenizer has 128K vocab for everything else
   - Best of both worlds!
");
    
    println!("{}", "=".repeat(80));
    println!("✓ All R-BPE logic works through FastTokenizer API!");
    println!("\nKey Point: This is NOT just wrapping tokenizer.json.");
    println!("All the custom R-BPE logic runs in Rust code!");
    
    Ok(())
}
