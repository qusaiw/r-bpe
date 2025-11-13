//! Advanced decoder with replacement character handling
//!
//! This module implements the sliding window algorithm from the Python version
//! to handle UTF-8 byte sequences that are split across multiple tokens.

use crate::utils::mappings::VocabMapper;
use tokenizers::Tokenizer;
use thiserror::Error;

/// Decoder errors
#[derive(Error, Debug)]
pub enum DecoderError {
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    
    #[error("Other error: {0}")]
    Other(String),
}

/// Advanced decoder that handles replacement characters with sliding window algorithm
pub struct RBPEDecoder {
    /// The new tokenizer
    new_tokenizer: Tokenizer,
    
    /// The old tokenizer
    old_tokenizer: Tokenizer,
    
    /// Vocabulary mapper
    vocab_mapper: VocabMapper,
    
    /// Last special token ID in old tokenizer
    old_last_special_token_id: u32,
}

impl RBPEDecoder {
    /// Create a new decoder
    pub fn new(
        new_tokenizer: Tokenizer,
        old_tokenizer: Tokenizer,
        vocab_mapper: VocabMapper,
        old_last_special_token_id: u32,
    ) -> Self {
        Self {
            new_tokenizer,
            old_tokenizer,
            vocab_mapper,
            old_last_special_token_id,
        }
    }

    /// Basic decode without replacement character handling
    pub fn basic_decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, DecoderError> {
        let segments = self.group_ids_by_mapping(ids);
        let mut decoded_parts = Vec::new();

        for (segment_ids, is_mapped) in segments {
            if is_mapped {
                // Map to new IDs and decode with new tokenizer
                let new_ids = self.vocab_mapper.map_old_to_new(&segment_ids);
                let text = self.new_tokenizer.decode(&new_ids, skip_special_tokens)?;
                decoded_parts.push(text);
            } else {
                // Decode directly with old tokenizer
                let text = self.old_tokenizer.decode(&segment_ids, skip_special_tokens)?;
                decoded_parts.push(text);
            }
        }

        Ok(decoded_parts.join(""))
    }

    /// Decode with replacement character handling using sliding window
    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, DecoderError> {
        // Try basic decode first
        let basic_decoded = self.basic_decode(ids, skip_special_tokens)?;
        
        // If no replacement characters, return early
        if !basic_decoded.contains('�') {
            return Ok(basic_decoded);
        }

        // Use sliding window algorithm
        let segments = self.segment_with_sliding_window(ids, skip_special_tokens)?;
        
        // Decode each segment
        let mut decoded_parts = Vec::new();
        for (segment_ids, is_mapped) in segments {
            let decoded = self.decode_segment(&segment_ids, is_mapped, skip_special_tokens)?;
            decoded_parts.push(decoded);
        }

        Ok(decoded_parts.join(""))
    }

    /// Decode a single segment with the appropriate tokenizer
    fn decode_segment(
        &self,
        segment: &[u32],
        is_mapped: bool,
        skip_special_tokens: bool,
    ) -> Result<String, DecoderError> {
        if is_mapped {
            let new_ids = self.vocab_mapper.map_old_to_new(segment);
            Ok(self.new_tokenizer.decode(&new_ids, skip_special_tokens)?)
        } else {
            Ok(self.old_tokenizer.decode(segment, skip_special_tokens)?)
        }
    }

    /// Group consecutive IDs by whether they're mapped
    fn group_ids_by_mapping(&self, ids: &[u32]) -> Vec<(Vec<u32>, bool)> {
        let mut segments = Vec::new();
        let mut current_segment = Vec::new();
        let mut current_is_mapped: Option<bool> = None;

        for &id in ids {
            let is_mapped = self.vocab_mapper.is_mapped(id);

            if let Some(was_mapped) = current_is_mapped {
                if is_mapped != was_mapped {
                    // Mapping status changed, start new segment
                    if !current_segment.is_empty() {
                        segments.push((current_segment.clone(), was_mapped));
                        current_segment.clear();
                    }
                }
            }

            current_segment.push(id);
            current_is_mapped = Some(is_mapped);
        }

        // Add final segment
        if !current_segment.is_empty() {
            if let Some(is_mapped) = current_is_mapped {
                segments.push((current_segment, is_mapped));
            }
        }

        segments
    }

    /// Segment IDs using sliding window algorithm to handle replacement characters
    fn segment_with_sliding_window(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<Vec<(Vec<u32>, bool)>, DecoderError> {
        let mut segments = Vec::new();
        let mut current_segment = Vec::new();
        let mut current_is_mapped: Option<bool> = None;
        let mut i = 0;

        while i < ids.len() {
            let token_id = ids[i];
            let is_byte = token_id > self.old_last_special_token_id 
                && token_id <= (256 + self.old_last_special_token_id);
            let is_replacement = self.vocab_mapper.has_replacement_char(token_id) || is_byte;
            let _is_common_token = self.vocab_mapper.is_common_token(token_id);
            let is_mapped = self.vocab_mapper.is_mapped(token_id);

            // Handle first token in sequence
            if current_is_mapped.is_none() {
                // If the first token is a replacement character and common, try window approach
                if is_replacement {
                    if let Some((window_size, _decoded, best_is_mapped)) =
                        self.find_optimal_window(ids, i, &current_segment, current_is_mapped, skip_special_tokens)?
                    {
                        segments.push((ids[i..i + window_size].to_vec(), best_is_mapped));
                        i += window_size;
                        continue;
                    }
                }

                // Otherwise, handle normally
                current_is_mapped = Some(is_mapped);
                current_segment.push(token_id);
                i += 1;
                continue;
            }

            // Special handling for replacement characters
            if is_replacement {
                if let Some((window_size, _decoded, best_is_mapped)) =
                    self.find_optimal_window(ids, i, &current_segment, current_is_mapped, skip_special_tokens)?
                {
                    // If we found a good window size
                    if best_is_mapped == current_is_mapped.unwrap() {
                        // Same mapping status, extend current segment
                        current_segment.extend_from_slice(&ids[i..i + window_size]);
                        i += window_size;
                    } else {
                        // Different mapping status, start new segment
                        if !current_segment.is_empty() {
                            segments.push((current_segment.clone(), current_is_mapped.unwrap()));
                        }
                        current_segment = ids[i..i + window_size].to_vec();
                        current_is_mapped = Some(best_is_mapped);
                        i += window_size;
                    }
                } else {
                    // If we couldn't resolve the replacement character, add just this token
                    if is_mapped != current_is_mapped.unwrap() {
                        if !current_segment.is_empty() {
                            segments.push((current_segment.clone(), current_is_mapped.unwrap()));
                        }
                        current_segment = vec![token_id];
                        current_is_mapped = Some(is_mapped);
                    } else {
                        current_segment.push(token_id);
                    }
                    i += 1;
                }
                continue;
            }

            // Regular segmentation logic for other tokens
            if is_mapped != current_is_mapped.unwrap() {
                if !current_segment.is_empty() {
                    segments.push((current_segment.clone(), current_is_mapped.unwrap()));
                }
                current_segment = vec![token_id];
                current_is_mapped = Some(is_mapped);
            } else {
                current_segment.push(token_id);
            }
            i += 1;
        }

        // Add final segment if it exists
        if !current_segment.is_empty() {
            if let Some(is_mapped) = current_is_mapped {
                segments.push((current_segment, is_mapped));
            }
        }

        Ok(segments)
    }

    /// Find the optimal window size that successfully decodes replacement characters
    /// 
    /// Tries different window sizes (1-4 tokens) to find one that produces text without
    /// replacement characters, testing both tokenizers as appropriate.
    /// 
    /// Returns: (window_size, decoded_text, is_mapped_flag)
    fn find_optimal_window(
        &self,
        ids: &[u32],
        start_idx: usize,
        current_segment: &[u32],
        current_is_mapped: Option<bool>,
        skip_special_tokens: bool,
    ) -> Result<Option<(usize, String, bool)>, DecoderError> {
        // Try to group with up to 3 more tokens to form complete UTF-8 character
        let max_window_size = std::cmp::min(4, ids.len() - start_idx);
        
        // Try different window sizes with both tokenizers
        for window_size in 1..=max_window_size {
            let test_window = &ids[start_idx..start_idx + window_size];
            
            // Combine current segment with test window
            let mut test_segment = current_segment.to_vec();
            test_segment.extend_from_slice(test_window);
            
            // Check if any token in the window is not mapped
            let window_has_unmapped = test_window.iter().any(|&tid| !self.vocab_mapper.is_mapped(tid));
            
            // Try decoding with old tokenizer if any token is unmapped
            if window_has_unmapped || (current_is_mapped.is_some() && !current_is_mapped.unwrap()) {
                let decoded = self.old_tokenizer.decode(&test_segment, skip_special_tokens)?;
                if !decoded.contains('�') {
                    return Ok(Some((window_size, decoded, false)));
                }
            }
            
            // Try decoding with new tokenizer if all tokens are mapped
            if test_segment.iter().all(|&tid| self.vocab_mapper.is_mapped(tid)) {
                let new_ids = self.vocab_mapper.map_old_to_new(&test_segment);
                let decoded = self.new_tokenizer.decode(&new_ids, skip_special_tokens)?;
                if !decoded.contains('�') {
                    return Ok(Some((window_size, decoded, true)));
                }
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_decoder() -> RBPEDecoder {
        // Create minimal test tokenizers
        let new_tokenizer = Tokenizer::from_file("../rbpe_tokenizer/new_tokenizer/tokenizer.json")
            .expect("Failed to load new tokenizer");
        let old_tokenizer = Tokenizer::from_file("../rbpe_tokenizer/old_tokenizer/tokenizer.json")
            .expect("Failed to load old tokenizer");
        
        // Create test mapper
        let mut new_to_old = HashMap::new();
        new_to_old.insert(0, 0);
        new_to_old.insert(1, 100);
        
        let mut old_to_new = HashMap::new();
        old_to_new.insert(0, 0);
        old_to_new.insert(100, 1);
        
        let mapper = VocabMapper::new(
            new_to_old,
            old_to_new,
            HashMap::new(),
            vec![0],
        );
        
        RBPEDecoder::new(new_tokenizer, old_tokenizer, mapper, 128000)
    }

    #[test]
    fn test_basic_decode() {
        let decoder = create_test_decoder();
        
        // Test simple decode
        let ids = vec![9906, 4435]; // "Hello World"
        let result = decoder.basic_decode(&ids, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_group_ids_by_mapping() {
        let decoder = create_test_decoder();
        
        // Test grouping logic
        let ids = vec![0, 100, 200]; // 0 and 100 are mapped, 200 is not
        let segments = decoder.group_ids_by_mapping(&ids);
        
        // Should group based on mapping status
        assert!(!segments.is_empty());
    }
}
