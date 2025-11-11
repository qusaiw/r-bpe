//! R-BPE (Reusable BPE) Tokenizer - Rust Implementation
//!
//! A high-performance Rust implementation of the R-BPE tokenizer that provides
//! language-aware tokenization with vocabulary reuse.
//!
//! ## FastTokenizer Compatibility
//!
//! The `RBPEFastTokenizer` provides compatibility with HuggingFace's tokenizers crate
//! while preserving all R-BPE logic (dual tokenizers, mappings, language routing).
//!
//! ## Python Bindings
//!
//! Python bindings are available via PyO3, allowing Python code to use the
//! high-performance Rust implementation.

pub mod utils;
pub mod normalizer;
pub mod pretokenizer;
pub mod model;
pub mod decoder;
pub mod tokenizer;
pub mod hf_builder;
pub mod fast_tokenizer;
pub mod python_bindings;

pub use utils::{UnicodeRange, UnicodeRangeChecker};
pub use tokenizer::RBPETokenizer;
pub use hf_builder::HFTokenizerBuilder;
pub use fast_tokenizer::RBPEFastTokenizer;
