//! Example: Load vocabulary mappings from R-BPE tokenizer

use rbpe_tokenizers::utils::VocabMapper;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_path = Path::new("../rbpe_tokenizer/metadata");
    
    let mapper = VocabMapper::from_json_files(
        &base_path.join("new_to_old_map.json"),
        &base_path.join("old_to_new_map.json"),
        Some(&base_path.join("replacement_character_map.json")),
    )?;
    
    let stats = mapper.stats();
    stats.print();
    
    // Test some mappings
    println!("\nSample mappings:");
    for new_id in [0, 1, 2, 100, 1000].iter() {
        if let Some(old_id) = mapper.new_to_old_id(*new_id) {
            println!("  New ID {} -> Old ID {}", new_id, old_id);
        }
    }
    
    Ok(())
}
