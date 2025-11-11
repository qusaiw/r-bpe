//! Simple performance test - measure throughput of encoding/decoding

use rbpe_tokenizers::model::RBPEModel;
use rbpe_tokenizers::pretokenizer::RBPEPreTokenizer;
use rbpe_tokenizers::utils::UnicodeRangeChecker;
use rbpe_tokenizers::utils::unicode_ranges::ranges;
use std::path::Path;
use std::time::Instant;

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

fn main() {
    println!("R-BPE Tokenizer Performance Test");
    println!("{}", "=".repeat(80));
    
    let model = create_model();
    
    let medium_english = "The quick brown fox jumps over the lazy dog. ".repeat(5);
    let medium_arabic = "مرحبا! كيف حالك اليوم؟ ".repeat(5);
    let long_mixed = "This is mixed text هذا نص مختلط ".repeat(20);
    
    let test_cases: Vec<(&str, &str, usize)> = vec![
        ("Short English", "Hello World", 1000),
        ("Short Arabic", "مرحبا", 1000),
        ("Mixed", "Hello مرحبا World", 1000),
        ("Medium English", &medium_english, 500),
        ("Medium Arabic", &medium_arabic, 500),
        ("Long Mixed", &long_mixed, 100),
    ];
    
    println!("\nEncoding Performance:");
    println!("{}", "-".repeat(80));
    
    for (name, text, iterations) in &test_cases {
        let start = Instant::now();
        for _ in 0..*iterations {
            let _ = model.encode(text, false).unwrap();
        }
        let elapsed = start.elapsed();
        
        let avg_micros = elapsed.as_micros() as f64 / *iterations as f64;
        let tokens_per_sec = (*iterations as f64 / elapsed.as_secs_f64()) as u64;
        
        println!(
            "{:20} {:6} iterations: {:8.2} µs/op, {:7} ops/sec",
            name, iterations, avg_micros, tokens_per_sec
        );
    }
    
    println!("\nDecoding Performance (Basic):");
    println!("{}", "-".repeat(80));
    
    for (name, text, iterations) in &test_cases {
        let ids = model.encode(text, false).unwrap();
        
        let start = Instant::now();
        for _ in 0..*iterations {
            let _ = model.decode(&ids, false).unwrap();
        }
        let elapsed = start.elapsed();
        
        let avg_micros = elapsed.as_micros() as f64 / *iterations as f64;
        let tokens_per_sec = (*iterations as f64 / elapsed.as_secs_f64()) as u64;
        
        println!(
            "{:20} {:6} iterations: {:8.2} µs/op, {:7} ops/sec",
            name, iterations, avg_micros, tokens_per_sec
        );
    }
    
    println!("\nDecoding Performance (Advanced):");
    println!("{}", "-".repeat(80));
    
    for (name, text, iterations) in &test_cases {
        let ids = model.encode(text, false).unwrap();
        
        let start = Instant::now();
        for _ in 0..*iterations {
            let _ = model.decode_advanced(&ids, false).unwrap();
        }
        let elapsed = start.elapsed();
        
        let avg_micros = elapsed.as_micros() as f64 / *iterations as f64;
        let tokens_per_sec = (*iterations as f64 / elapsed.as_secs_f64()) as u64;
        
        println!(
            "{:20} {:6} iterations: {:8.2} µs/op, {:7} ops/sec",
            name, iterations, avg_micros, tokens_per_sec
        );
    }
    
    println!("\nRoundtrip Performance (Encode + Decode):");
    println!("{}", "-".repeat(80));
    
    for (name, text, iterations) in &test_cases {
        let start = Instant::now();
        for _ in 0..*iterations {
            let ids = model.encode(text, false).unwrap();
            let _ = model.decode(&ids, false).unwrap();
        }
        let elapsed = start.elapsed();
        
        let avg_micros = elapsed.as_micros() as f64 / *iterations as f64;
        let roundtrips_per_sec = (*iterations as f64 / elapsed.as_secs_f64()) as u64;
        
        println!(
            "{:20} {:6} iterations: {:8.2} µs/op, {:7} ops/sec",
            name, iterations, avg_micros, roundtrips_per_sec
        );
    }
    
    println!("\n{}", "=".repeat(80));
    println!("✓ Performance test complete!");
}
