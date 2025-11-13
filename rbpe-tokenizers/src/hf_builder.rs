//! HuggingFace tokenizer.json builder
//!
//! This module creates a tokenizer.json file that can be loaded by
//! HuggingFace's transformers library via AutoTokenizer.from_pretrained().

use serde_json::Value;

/// Build a HuggingFace-compatible tokenizer.json
pub struct HFTokenizerBuilder {
    /// Base tokenizer (old tokenizer)
    old_tokenizer_path: String,
    
    /// Target tokenizer (new tokenizer)  
    new_tokenizer_path: String,
    
    /// Metadata paths
    new_to_old_map_path: String,
    old_to_new_map_path: String,
    replacement_char_map_path: Option<String>,
    
    /// Target language Unicode ranges
    target_language_ranges: Vec<(u32, u32)>,
    
    /// Normalization map path (optional)
    normalization_map_path: Option<String>,
}

impl HFTokenizerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            old_tokenizer_path: String::new(),
            new_tokenizer_path: String::new(),
            new_to_old_map_path: String::new(),
            old_to_new_map_path: String::new(),
            replacement_char_map_path: None,
            target_language_ranges: Vec::new(),
            normalization_map_path: None,
        }
    }
    
    /// Set the old tokenizer path
    pub fn old_tokenizer(mut self, path: &str) -> Self {
        self.old_tokenizer_path = path.to_string();
        self
    }
    
    /// Set the new tokenizer path
    pub fn new_tokenizer(mut self, path: &str) -> Self {
        self.new_tokenizer_path = path.to_string();
        self
    }
    
    /// Set mapping paths
    pub fn mappings(mut self, new_to_old: &str, old_to_new: &str) -> Self {
        self.new_to_old_map_path = new_to_old.to_string();
        self.old_to_new_map_path = old_to_new.to_string();
        self
    }
    
    /// Set replacement character map path
    pub fn replacement_char_map(mut self, path: &str) -> Self {
        self.replacement_char_map_path = Some(path.to_string());
        self
    }
    
    /// Add target language Unicode range
    pub fn add_target_range(mut self, start: u32, end: u32) -> Self {
        self.target_language_ranges.push((start, end));
        self
    }
    
    /// Set Arabic as target language
    pub fn arabic_target(mut self) -> Self {
        // Arabic Unicode ranges
        self.target_language_ranges = vec![
            (0x0600, 0x06FF),  // Arabic
            (0x0750, 0x077F),  // Arabic Supplement
            (0x08A0, 0x08FF),  // Arabic Extended-A
            (0xFB50, 0xFDFF),  // Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  // Arabic Presentation Forms-B
        ];
        self
    }
    
    /// Set normalization map path
    pub fn normalization_map(mut self, path: &str) -> Self {
        self.normalization_map_path = Some(path.to_string());
        self
    }
    
    /// Build the tokenizer.json configuration
    pub fn build(&self) -> Result<Value, Box<dyn std::error::Error>> {
        // Load the old tokenizer as base
        let old_tokenizer_json = std::fs::read_to_string(&self.old_tokenizer_path)?;
        let mut tokenizer: Value = serde_json::from_str(&old_tokenizer_json)?;
        
        // Add custom pre_tokenizer configuration
        tokenizer["pre_tokenizer"] = serde_json::json!({
            "type": "RBPE",
            "target_language_ranges": self.target_language_ranges,
        });
        
        // Add custom normalizer if provided
        if let Some(norm_path) = &self.normalization_map_path {
            tokenizer["normalizer"] = serde_json::json!({
                "type": "RBPENormalizer",
                "normalization_map_path": norm_path,
            });
        }
        
        // Add custom decoder
        tokenizer["decoder"] = serde_json::json!({
            "type": "RBPEDecoder",
            "new_to_old_map_path": self.new_to_old_map_path,
            "old_to_new_map_path": self.old_to_new_map_path,
            "replacement_char_map_path": self.replacement_char_map_path,
        });
        
        // Add metadata for R-BPE
        tokenizer["rbpe_config"] = serde_json::json!({
            "old_tokenizer_path": self.old_tokenizer_path,
            "new_tokenizer_path": self.new_tokenizer_path,
            "new_to_old_map_path": self.new_to_old_map_path,
            "old_to_new_map_path": self.old_to_new_map_path,
            "replacement_char_map_path": self.replacement_char_map_path,
            "target_language_ranges": self.target_language_ranges,
            "normalization_map_path": self.normalization_map_path,
        });
        
        Ok(tokenizer)
    }
    
    /// Build and save to file
    pub fn save(&self, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let config = self.build()?;
        let json_str = serde_json::to_string_pretty(&config)?;
        std::fs::write(output_path, json_str)?;
        Ok(())
    }
}

impl Default for HFTokenizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_builder() {
        let builder = HFTokenizerBuilder::new()
            .old_tokenizer("../rbpe_tokenizer/old_tokenizer/tokenizer.json")
            .new_tokenizer("../rbpe_tokenizer/new_tokenizer/tokenizer.json")
            .mappings(
                "../rbpe_tokenizer/metadata/new_to_old_map.json",
                "../rbpe_tokenizer/metadata/old_to_new_map.json"
            )
            .arabic_target();
        
        let config = builder.build();
        assert!(config.is_ok());
        
        let config = config.unwrap();
        assert!(config.get("rbpe_config").is_some());
        assert!(config.get("pre_tokenizer").is_some());
    }
}
