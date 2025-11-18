//! Python bindings for R-BPE tokenizer using PyO3
//!
//! This module exposes the R-BPE tokenizer to Python, allowing Python code
//! to use the high-performance Rust implementation with all R-BPE features.

use pyo3::prelude::*;
use crate::fast_tokenizer::RBPEFastTokenizer;
use crate::pretokenizer::RBPEPreTokenizer;
use crate::utils::UnicodeRangeChecker;
use crate::utils::unicode_ranges::ranges;
use std::path::PathBuf;

/// Python wrapper for RBPEFastTokenizer
#[pyclass(name = "RBPETokenizer")]
pub struct PyRBPETokenizer {
    inner: RBPEFastTokenizer,
}

#[pymethods]
impl PyRBPETokenizer {
    /// Create a new R-BPE tokenizer from a pretrained directory
    /// 
    /// This is the recommended way to load a tokenizer, similar to
    /// the original Python implementation.
    /// 
    /// Args:
    ///     pretrained_path: Path to directory containing tokenizer files
    ///     target_language: Target language for optimization (default: "arabic")
    /// 
    /// Expected directory structure:
    ///     pretrained_path/
    ///         new_tokenizer/tokenizer.json
    ///         old_tokenizer/tokenizer.json
    ///         metadata/new_to_old_map.json
    ///         metadata/old_to_new_map.json
    ///         metadata/replacement_character_map.json
    /// 
    /// Example:
    ///     tokenizer = RBPETokenizer.from_pretrained("rbpe_tokenizer")
    #[staticmethod]
    #[pyo3(signature = (pretrained_path, target_language="arabic"))]
    fn from_pretrained(
        pretrained_path: &str,
        target_language: &str,
    ) -> PyResult<Self> {
        use std::path::Path;
        
        let base_path = Path::new(pretrained_path);
        
        // Construct paths following the expected structure
        let new_tokenizer_path = base_path.join("new_tokenizer").join("tokenizer.json");
        let old_tokenizer_path = base_path.join("old_tokenizer").join("tokenizer.json");
        let new_to_old_map_path = base_path.join("metadata").join("new_to_old_map.json");
        let old_to_new_map_path = base_path.join("metadata").join("old_to_new_map.json");
        let replacement_char_map_path = base_path.join("metadata").join("replacement_character_map.json");
        
        // Verify that required files exist
        if !new_tokenizer_path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                format!("New tokenizer not found at: {}", new_tokenizer_path.display())
            ));
        }
        if !old_tokenizer_path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                format!("Old tokenizer not found at: {}", old_tokenizer_path.display())
            ));
        }
        if !new_to_old_map_path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                format!("new_to_old_map.json not found at: {}", new_to_old_map_path.display())
            ));
        }
        if !old_to_new_map_path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                format!("old_to_new_map.json not found at: {}", old_to_new_map_path.display())
            ));
        }
        
        // Create Unicode range checker based on target language
        let checker = match target_language {
            "arabic" => UnicodeRangeChecker::new(ranges::all_arabic()),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported target language: {}. Currently only 'arabic' is supported.", target_language)
            )),
        };
        
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
        
        // replacement_character_map is optional
        let replacement_path = if replacement_char_map_path.exists() {
            Some(replacement_char_map_path.as_path())
        } else {
            None
        };
        
        let tokenizer = RBPEFastTokenizer::from_files(
            &new_tokenizer_path,
            &old_tokenizer_path,
            &new_to_old_map_path,
            &old_to_new_map_path,
            replacement_path,
            pretokenizer,
            None,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Failed to load tokenizer from '{}': {}", pretrained_path, e)
        ))?;
        
        Ok(Self { inner: tokenizer })
    }
    
    /// Create a new R-BPE tokenizer from explicit file paths
    /// 
    /// Use this method if you have a custom directory structure or want
    /// fine-grained control over which files to load.
    /// 
    /// Args:
    ///     new_tokenizer_path: Path to new tokenizer (Arabic-optimized)
    ///     old_tokenizer_path: Path to old tokenizer (base model)
    ///     new_to_old_map_path: Path to new→old ID mapping
    ///     old_to_new_map_path: Path to old→new ID mapping
    ///     replacement_char_map_path: Optional path to replacement character map
    ///     target_language: Target language for optimization (default: "arabic")
    #[staticmethod]
    #[pyo3(signature = (new_tokenizer_path, old_tokenizer_path, new_to_old_map_path, old_to_new_map_path, replacement_char_map_path=None, target_language="arabic"))]
    fn from_files(
        new_tokenizer_path: &str,
        old_tokenizer_path: &str,
        new_to_old_map_path: &str,
        old_to_new_map_path: &str,
        replacement_char_map_path: Option<&str>,
        target_language: &str,
    ) -> PyResult<Self> {
        // Create Unicode range checker based on target language
        let checker = match target_language {
            "arabic" => UnicodeRangeChecker::new(ranges::all_arabic()),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported target language: {}", target_language)
            )),
        };
        
        let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
        
        let replacement_path = replacement_char_map_path.map(PathBuf::from);
        
        let tokenizer = RBPEFastTokenizer::from_files(
            &PathBuf::from(new_tokenizer_path),
            &PathBuf::from(old_tokenizer_path),
            &PathBuf::from(new_to_old_map_path),
            &PathBuf::from(old_to_new_map_path),
            replacement_path.as_deref(),
            pretokenizer,
            None,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load tokenizer: {}", e)))?;
        
        Ok(Self { inner: tokenizer })
    }
    
    /// Encode text to token IDs
    /// 
    /// Args:
    ///     text: Input text to encode
    ///     add_special_tokens: Whether to add special tokens (default: False)
    /// 
    /// Returns:
    ///     List of token IDs
    #[pyo3(signature = (text, add_special_tokens=false))]
    fn encode(&self, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        self.inner
            .encode_text(text, add_special_tokens)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Encoding failed: {}", e)))
    }
    
    /// Decode token IDs to text (basic decoder)
    /// 
    /// Args:
    ///     ids: List of token IDs
    ///     skip_special_tokens: Whether to skip special tokens (default: True)
    /// 
    /// Returns:
    ///     Decoded text
    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.inner
            .decode_ids(&ids, skip_special_tokens)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Decoding failed: {}", e)))
    }
    
    /// Decode token IDs to text (advanced decoder with replacement char handling)
    /// 
    /// Args:
    ///     ids: List of token IDs
    ///     skip_special_tokens: Whether to skip special tokens (default: True)
    /// 
    /// Returns:
    ///     Decoded text
    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode_advanced(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.inner
            .decode_ids_advanced(&ids, skip_special_tokens)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Advanced decoding failed: {}", e)))
    }
    
    /// Encode multiple texts in batch
    /// 
    /// Args:
    ///     texts: List of texts to encode
    ///     add_special_tokens: Whether to add special tokens (default: False)
    /// 
    /// Returns:
    ///     List of token ID lists
    #[pyo3(signature = (texts, add_special_tokens=false))]
    fn encode_batch(&self, texts: Vec<String>, add_special_tokens: bool) -> PyResult<Vec<Vec<u32>>> {
        texts.iter()
            .map(|text| {
                self.inner
                    .encode_text(text, add_special_tokens)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Batch encoding failed: {}", e)))
            })
            .collect()
    }
    
    /// Decode multiple token ID sequences in batch
    /// 
    /// Args:
    ///     ids_batch: List of token ID lists
    ///     skip_special_tokens: Whether to skip special tokens (default: True)
    /// 
    /// Returns:
    ///     List of decoded texts
    #[pyo3(signature = (ids_batch, skip_special_tokens=true))]
    fn decode_batch(&self, ids_batch: Vec<Vec<u32>>, skip_special_tokens: bool) -> PyResult<Vec<String>> {
        ids_batch.iter()
            .map(|ids| {
                self.inner
                    .decode_ids(ids, skip_special_tokens)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Batch decoding failed: {}", e)))
            })
            .collect()
    }
    
    /// Get token ID for a given token string
    /// 
    /// Args:
    ///     token: Token string to look up
    /// 
    /// Returns:
    ///     Token ID or None if not in vocabulary
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    
    /// Get token string for a given token ID
    /// 
    /// Args:
    ///     id: Token ID to look up
    /// 
    /// Returns:
    ///     Token string or None if ID not in vocabulary
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
    
    /// Get the vocabulary size
    /// 
    /// Returns:
    ///     Size of the vocabulary
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    
    /// Set whether to add BOS token
    /// 
    /// Args:
    ///     add: Whether to add BOS token
    fn set_add_bos_token(&mut self, add: bool) {
        self.inner.set_add_bos_token(add);
    }
    
    /// Set whether to add EOS token
    /// 
    /// Args:
    ///     add: Whether to add EOS token
    fn set_add_eos_token(&mut self, add: bool) {
        self.inner.set_add_eos_token(add);
    }
    
    /// Get whether BOS token is added
    /// 
    /// Returns:
    ///     True if BOS token is added
    fn get_add_bos_token(&self) -> bool {
        self.inner.add_bos_token()
    }
    
    /// Get whether EOS token is added
    /// 
    /// Returns:
    ///     True if EOS token is added
    fn get_add_eos_token(&self) -> bool {
        self.inner.add_eos_token()
    }
    
    /// Get a string representation
    fn __repr__(&self) -> String {
        "RBPETokenizer(dual_tokenizer=true, language_aware=true)".to_string()
    }
    
    /// Get a string representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python module initialization
#[pymodule]
fn rbpe_tokenizers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRBPETokenizer>()?;
    
    // Add module docstring
    m.add("__doc__", "R-BPE (Reusable BPE) Tokenizer - High-performance Rust implementation with Python bindings")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
