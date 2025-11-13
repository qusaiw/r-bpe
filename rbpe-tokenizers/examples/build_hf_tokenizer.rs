//! Example: Build a HuggingFace-compatible tokenizer.json
//!
//! This example shows how to create a tokenizer.json file that can be loaded
//! by HuggingFace's AutoTokenizer.from_pretrained() in Python.

use rbpe_tokenizers::HFTokenizerBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building HuggingFace-compatible tokenizer.json");
    println!("{}", "=".repeat(80));
    
    // Build the tokenizer configuration
    let builder = HFTokenizerBuilder::new()
        .old_tokenizer("../rbpe_tokenizer/old_tokenizer/tokenizer.json")
        .new_tokenizer("../rbpe_tokenizer/new_tokenizer/tokenizer.json")
        .mappings(
            "../rbpe_tokenizer/metadata/new_to_old_map.json",
            "../rbpe_tokenizer/metadata/old_to_new_map.json"
        )
        .replacement_char_map("../rbpe_tokenizer/metadata/replacement_character_map.json")
        .arabic_target()
        .normalization_map("arabic_normalization_map.json");
    
    println!("✓ Configuration created");
    println!("\nTarget language: Arabic");
    println!("Unicode ranges:");
    println!("  - U+0600-U+06FF (Arabic)");
    println!("  - U+0750-U+077F (Arabic Supplement)");
    println!("  - U+08A0-U+08FF (Arabic Extended-A)");
    println!("  - U+FB50-U+FDFF (Arabic Presentation Forms-A)");
    println!("  - U+FE70-U+FEFF (Arabic Presentation Forms-B)");
    
    // Build and save
    let output_path = "../rbpe_tokenizer_hf/tokenizer.json";
    println!("\nSaving to: {}", output_path);
    
    // Create output directory
    std::fs::create_dir_all("../rbpe_tokenizer_hf")?;
    
    builder.save(output_path)?;
    println!("✓ Saved successfully!");
    
    // Also save the config
    let config = builder.build()?;
    println!("\nTokenizer configuration:");
    println!("  Version: {}", config.get("version").unwrap_or(&serde_json::json!("unknown")));
    println!("  Model type: {}", config["model"]["type"]);
    println!("  Has pre_tokenizer: {}", config.get("pre_tokenizer").is_some());
    println!("  Has normalizer: {}", config.get("normalizer").is_some());
    println!("  Has decoder: {}", config.get("decoder").is_some());
    println!("  Has R-BPE config: {}", config.get("rbpe_config").is_some());
    
    // Copy metadata files
    println!("\nCopying metadata files...");
    std::fs::copy(
        "../rbpe_tokenizer/metadata/new_to_old_map.json",
        "../rbpe_tokenizer_hf/new_to_old_map.json"
    )?;
    std::fs::copy(
        "../rbpe_tokenizer/metadata/old_to_new_map.json",
        "../rbpe_tokenizer_hf/old_to_new_map.json"
    )?;
    std::fs::copy(
        "../rbpe_tokenizer/metadata/replacement_character_map.json",
        "../rbpe_tokenizer_hf/replacement_character_map.json"
    )?;
    
    if std::path::Path::new("arabic_normalization_map.json").exists() {
        std::fs::copy(
            "arabic_normalization_map.json",
            "../rbpe_tokenizer_hf/arabic_normalization_map.json"
        )?;
    }
    
    println!("✓ Metadata files copied");
    
    println!("\n{}", "=".repeat(80));
    println!("✓ HuggingFace-compatible tokenizer created!");
    println!("\nTo use in Python:");
    println!("  from transformers import AutoTokenizer");
    println!("  tokenizer = AutoTokenizer.from_pretrained('./rbpe_tokenizer_hf')");
    
    Ok(())
}
