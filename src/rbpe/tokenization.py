"""
R-BPE Tokenizer - HuggingFace Compatible Wrapper with Rust Backend

This file enables AutoTokenizer.from_pretrained() to load R-BPE tokenizers
while using the high-performance Rust implementation under the hood.
"""

from typing import List, Optional, Union, Dict, Any
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
    PaddingStrategy,
)
import os

# Try to import the Rust tokenizer
try:
    import rbpe_tokenizers
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    import warnings
    warnings.warn(
        "Rust R-BPE tokenizer not available. Install with: cd rbpe-tokenizers && maturin develop --release"
    )


class RBPETokenizer(PreTrainedTokenizer):
    """
    R-BPE (Reusable BPE) Tokenizer with Rust backend.
    
    This tokenizer uses a dual-tokenizer architecture with language-aware routing:
    - New tokenizer: Optimized for target language (e.g., Arabic)
    - Old tokenizer: Base multilingual model
    - Automatic routing based on language detection
    - Vocabulary mapping to maintain compatibility
    
    This class provides HuggingFace compatibility while delegating to the
    high-performance Rust implementation.
    """
    
    vocab_files_names = {
        "new_tokenizer": "new_tokenizer/tokenizer.json",
        "old_tokenizer": "old_tokenizer/tokenizer.json",
        "new_to_old_map": "metadata/new_to_old_map.json",
        "old_to_new_map": "metadata/old_to_new_map.json",
        "replacement_char_map": "metadata/replacement_character_map.json",
    }
    
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        new_tokenizer_file=None,
        old_tokenizer_file=None,
        new_to_old_map_file=None,
        old_to_new_map_file=None,
        replacement_char_map_file=None,
        target_language="arabic",
        bos_token=None,
        eos_token=None,
        unk_token=None,
        pad_token=None,
        **kwargs
    ):
        """
        Initialize R-BPE tokenizer.
        
        Args:
            new_tokenizer_file: Path to new (target language) tokenizer
            old_tokenizer_file: Path to old (base) tokenizer
            new_to_old_map_file: Path to new→old ID mapping
            old_to_new_map_file: Path to old→new ID mapping
            replacement_char_map_file: Path to replacement character map
            target_language: Target language for optimization (default: "arabic")
            **kwargs: Additional arguments passed to PreTrainedTokenizer
        """
        if not RUST_AVAILABLE:
            raise ImportError(
                "Rust R-BPE tokenizer is required. Install with: "
                "cd rbpe-tokenizers && maturin develop --release"
            )
        
        # Initialize parent class first
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs
        )
        
        # Determine paths
        self.name_or_path = kwargs.get('name_or_path', '.')
        
        # Fix special token IDs: added_tokens_decoder has duplicates (both R-BPE and original IDs)
        # We need to ensure the encoder map (added_tokens_encoder) points to R-BPE IDs (< 100)
        # Remove high ID duplicates from added_tokens_encoder to force use of low IDs
        tokens_to_fix = {
            '<BOS_TOKEN>': 5,
            '<EOS_TOKEN>': 6,
            '<PAD>': 0,
            '<UNK>': 1,
            '<MASK_TOKEN>': 4,
            '<SEP>': 3,
            '<CLS>': 2,
        }
        
        # Update the added_tokens_encoder to use R-BPE IDs
        for token_str, correct_id in tokens_to_fix.items():
            if token_str in self.added_tokens_encoder:
                # Force it to use the R-BPE ID, not the high ID
                self.added_tokens_encoder[token_str] = correct_id
        
        if new_tokenizer_file is None:
            # Load from directory using from_pretrained
            self._rust_tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained(
                self.name_or_path,
                target_language=target_language
            )
        else:
            # Load from explicit files
            self._rust_tokenizer = rbpe_tokenizers.RBPETokenizer.from_files(
                new_tokenizer_path=new_tokenizer_file,
                old_tokenizer_path=old_tokenizer_file,
                new_to_old_map_path=new_to_old_map_file,
                old_to_new_map_path=old_to_new_map_file,
                replacement_char_map_path=replacement_char_map_file,
                target_language=target_language
            )
        
        self.target_language = target_language
        
        # Store paths for backward compatibility with MappingTokenizer tests
        if new_tokenizer_file:
            self.new_tokenizer_path = new_tokenizer_file
            self.old_tokenizer_path = old_tokenizer_file
        else:
            # Infer from name_or_path
            base_path = self.name_or_path
            self.new_tokenizer_path = os.path.join(base_path, "new_tokenizer")
            self.old_tokenizer_path = os.path.join(base_path, "old_tokenizer")
        
        # Create mock tokenizer objects for backward compatibility
        # These provide minimal interface needed by tests
        self._create_mock_tokenizers()
    
    def _create_mock_tokenizers(self):
        """Create mock tokenizer objects that delegate to the Rust tokenizer."""
        class MockTokenizer:
            """Mock tokenizer object that provides basic interface."""
            def __init__(self, parent, tokenizer_type):
                self._parent = parent
                self._type = tokenizer_type
            
            def decode(self, ids, skip_special_tokens=True):
                """Decode using the parent tokenizer."""
                return self._parent.decode(ids, skip_special_tokens=skip_special_tokens)
            
            def encode(self, text, add_special_tokens=True):
                """Encode using the parent tokenizer."""
                return self._parent.encode(text, add_special_tokens=add_special_tokens)
            
            def get_vocab(self):
                """Get vocabulary."""
                return self._parent.get_vocab()
            
            @property
            def special_tokens_map(self):
                """Get special tokens map."""
                return self._parent.special_tokens_map
        
        # Create mock objects
        self.new_tokenizer = MockTokenizer(self, "new")
        self.old_tokenizer = MockTokenizer(self, "old")
    
    @property
    def is_fast(self) -> bool:
        """
        Returns True because this tokenizer uses a fast Rust backend.
        
        Note: HuggingFace's default is_fast property checks for PreTrainedTokenizerFast
        inheritance, but R-BPE uses its own Rust implementation (rbpe_tokenizers).
        We override this to accurately reflect that the tokenizer is fast.
        """
        return RUST_AVAILABLE
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """
        Returns the BOS token ID from R-BPE vocabulary.
        Queries the Rust tokenizer dynamically.
        """
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self._bos_token)
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """
        Returns the EOS token ID from R-BPE vocabulary.
        Queries the Rust tokenizer dynamically.
        """
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self._eos_token)
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """
        Returns the PAD token ID from R-BPE vocabulary.
        Queries the Rust tokenizer dynamically.
        """
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self._pad_token)
    
    @property  
    def unk_token_id(self) -> Optional[int]:
        """
        Returns the UNK token ID from R-BPE vocabulary.
        Queries the Rust tokenizer dynamically.
        """
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self._unk_token)
    
    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size.
        Note: This is the combined vocabulary size after mapping.
        """
        # Query the actual vocabulary size from the Rust tokenizer
        # This makes it work with any R-BPE tokenizer regardless of base model
        if hasattr(self, '_rust_tokenizer') and self._rust_tokenizer is not None:
            return self._rust_tokenizer.vocab_size()
        # During initialization, before _rust_tokenizer is created, return a large default
        # This will be replaced with the actual value once initialization completes
        return 256000  # Temporary default during initialization
    
    @property
    def new_to_old_map(self) -> Dict[int, int]:
        """
        Token ID mapping from new tokenizer to old tokenizer.
        
        Note: This is encapsulated in the Rust implementation and not directly accessible.
        Returns empty dict for backward compatibility.
        """
        import warnings
        warnings.warn(
            "new_to_old_map is not directly accessible in Rust implementation. "
            "The mapping is handled internally.",
            UserWarning
        )
        return {}
    
    @property
    def old_to_new_map(self) -> Dict[int, int]:
        """
        Token ID mapping from old tokenizer to new tokenizer.
        
        Note: This is encapsulated in the Rust implementation and not directly accessible.
        Returns empty dict for backward compatibility.
        """
        import warnings
        warnings.warn(
            "old_to_new_map is not directly accessible in Rust implementation. "
            "The mapping is handled internally.",
            UserWarning
        )
        return {}
    
    @property
    def replacement_character_map(self) -> Dict[int, str]:
        """
        Map of token IDs that contain replacement characters.
        
        Note: This is encapsulated in the Rust implementation and not directly accessible.
        Returns empty dict for backward compatibility.
        """
        import warnings
        warnings.warn(
            "replacement_character_map is not directly accessible in Rust implementation. "
            "Replacement character handling is done internally.",
            UserWarning
        )
        return {}
    
    @property
    def common_token_ids_map(self) -> Dict[int, bool]:
        """
        Map of common token IDs between tokenizers.
        
        Note: This is encapsulated in the Rust implementation and not directly accessible.
        Returns empty dict for backward compatibility.
        """
        import warnings
        warnings.warn(
            "common_token_ids_map is not directly accessible in Rust implementation. "
            "Common token tracking is handled internally.",
            UserWarning
        )
        return {}
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.
        Note: For R-BPE, this returns a conceptual vocab. The actual
        tokenization uses dual tokenizers internally.
        """
        # This is a simplified implementation
        # In practice, R-BPE's vocabulary is distributed across two tokenizers
        return {f"<token_{i}>": i for i in range(self.vocab_size)}
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize a string into tokens.
        Note: This returns token IDs as strings since R-BPE works with IDs directly.
        """
        # R-BPE works directly with IDs, not string tokens
        # This method is required by PreTrainedTokenizer but not the primary interface
        ids = self._rust_tokenizer.encode(text, add_special_tokens=False)
        return [str(id) for id in ids]
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token (str) to an ID (int)."""
        try:
            return int(token)
        except ValueError:
            return self.unk_token_id or 0
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID (int) to a token (str)."""
        return str(index)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens to a single string."""
        ids = [int(token) for token in tokens]
        return self._rust_tokenizer.decode(ids, skip_special_tokens=True)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Override to ensure R-BPE vocab IDs are used instead of original vocab IDs.
        
        This queries the Rust tokenizer's vocabulary, making it work with any R-BPE tokenizer
        regardless of which base model it was trained from.
        """
        if isinstance(tokens, str):
            # Query Rust tokenizer for the token ID
            token_id = self._rust_tokenizer.token_to_id(tokens)
            if token_id is not None:
                return token_id
            # Fall back to parent class if token not found
            return super().convert_tokens_to_ids(tokens)
        else:
            # List of tokens
            return [self.convert_tokens_to_ids(token) for token in tokens]
    
    def convert_tok_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to token strings.
        
        This method is for backward compatibility with MappingTokenizer tests.
        Each ID is decoded individually to get its string representation.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of decoded token strings
        """
        tokens = []
        for token_id in ids:
            # Decode each token individually
            decoded = self._rust_tokenizer.decode([token_id], skip_special_tokens=False)
            tokens.append(decoded)
        return tokens
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = False,
        **kwargs
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Override the base class method to directly use the Rust tokenizer,
        bypassing the complex PreTrainedTokenizer path that goes through
        _tokenize() -> _convert_token_to_id() etc.
        
        Note: Defaults to add_special_tokens=False to match MappingTokenizer behavior.
        This differs from standard HuggingFace tokenizers which default to True.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens (default: False)
            
        Returns:
            List of token IDs
        """
        if isinstance(text, list):
            # Batch encoding
            return self._rust_tokenizer.encode_batch(text, add_special_tokens=add_special_tokens)
        else:
            # Single text encoding
            return self._rust_tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Handles both Rust vocabulary and Python added_tokens_decoder.
        
        Args:
            token_ids: Token IDs to decode (single list or batch)
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text (single string or list of strings)
        """
        # Handle empty input
        if not token_ids:
            return ""
        
        # Check if it's a batch (list of lists)
        if isinstance(token_ids[0], list):
            # Batch decoding
            return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]
        
        # Single sequence decoding with added_tokens support
        # Separate added tokens from regular tokens
        decoded_parts = []
        rust_ids = []
        
        for token_id in token_ids:
            # Check if this ID is in added_tokens_decoder
            if token_id in self.added_tokens_decoder:
                # Decode any accumulated rust IDs first
                if rust_ids:
                    rust_decoded = self._rust_tokenizer.decode(rust_ids, skip_special_tokens=skip_special_tokens)
                    decoded_parts.append(rust_decoded)
                    rust_ids = []
                
                # Always add added tokens (they were explicitly provided, not auto-added)
                decoded_parts.append(str(self.added_tokens_decoder[token_id]))
            else:
                # Accumulate regular vocab IDs for Rust decoder
                rust_ids.append(token_id)
        
        # Decode any remaining rust IDs
        if rust_ids:
            rust_decoded = self._rust_tokenizer.decode(rust_ids, skip_special_tokens=skip_special_tokens)
            decoded_parts.append(rust_decoded)
        
        return "".join(decoded_parts)
    
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model one or several sequence(s).
        """
        # Handle truncation properly with special tokens
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length is not None and add_special_tokens:
            # Reserve space for special tokens (BOS + EOS = 2 tokens)
            content_max_length = max_length - 2
            if content_max_length < 1:
                content_max_length = 1
            
            # Encode without special tokens
            if isinstance(text, str):
                input_ids = self._rust_tokenizer.encode(text, add_special_tokens=False)
            else:
                input_ids = text
            
            # Truncate content
            if len(input_ids) > content_max_length:
                input_ids = input_ids[:content_max_length]
            
            # Add special tokens manually
            input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
            
        else:
            # Normal encoding (no truncation, or truncation without special tokens)
            if isinstance(text, str):
                input_ids = self._rust_tokenizer.encode(text, add_special_tokens=add_special_tokens)
            else:
                input_ids = text
            
            # Truncate if needed (simple case)
            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length is not None:
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
        
        # Handle text pairs (for sentence pair tasks)
        if text_pair is not None:
            if isinstance(text_pair, str):
                pair_ids = self._rust_tokenizer.encode(text_pair, add_special_tokens=False)
            else:
                pair_ids = text_pair
            
            # Combine: [BOS] text [SEP] text_pair [EOS]
            if add_special_tokens and truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
                sep_token_id = self.eos_token_id  # Use EOS as separator
                input_ids = input_ids + [sep_token_id] + pair_ids + [self.eos_token_id]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Prepare output
        encoded_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Pad if needed
        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            return self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_tensors=return_tensors,
                verbose=verbose,
            )
        
        # Convert to BatchEncoding
        return BatchEncoding(encoded_inputs, tensor_type=return_tensors)
    
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[TextInput], List[TextInput]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
        """
        # Check if we have pairs
        is_pair = isinstance(batch_text_or_text_pairs[0], (list, tuple))
        
        if is_pair:
            # Handle text pairs
            batch_outputs = {
                "input_ids": [],
                "attention_mask": [],
            }
            for text, text_pair in batch_text_or_text_pairs:
                encoded = self._encode_plus(
                    text,
                    text_pair=text_pair,
                    add_special_tokens=add_special_tokens,
                    padding_strategy=PaddingStrategy.DO_NOT_PAD,
                    truncation_strategy=truncation_strategy,
                    max_length=max_length,
                    **kwargs
                )
                batch_outputs["input_ids"].append(encoded["input_ids"])
                batch_outputs["attention_mask"].append(encoded["attention_mask"])
        else:
            # Handle truncation properly with special tokens
            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length is not None and add_special_tokens:
                # Reserve space for special tokens (BOS + EOS = 2 tokens)
                content_max_length = max_length - 2
                if content_max_length < 1:
                    content_max_length = 1
                
                # Encode without special tokens
                batch_ids = self._rust_tokenizer.encode_batch(
                    batch_text_or_text_pairs,
                    add_special_tokens=False
                )
                
                # Truncate and add special tokens to each sequence
                processed_batch_ids = []
                for ids in batch_ids:
                    if len(ids) > content_max_length:
                        ids = ids[:content_max_length]
                    # Add special tokens manually
                    ids = [self.bos_token_id] + ids + [self.eos_token_id]
                    processed_batch_ids.append(ids)
                
                batch_outputs = {
                    "input_ids": processed_batch_ids,
                    "attention_mask": [[1] * len(ids) for ids in processed_batch_ids],
                }
            else:
                # Normal encoding (no truncation, or truncation without special tokens)
                batch_ids = self._rust_tokenizer.encode_batch(
                    batch_text_or_text_pairs,
                    add_special_tokens=add_special_tokens
                )
                
                # Truncate if needed (simple case)
                if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length is not None:
                    batch_ids = [ids[:max_length] if len(ids) > max_length else ids for ids in batch_ids]
                
                batch_outputs = {
                    "input_ids": batch_ids,
                    "attention_mask": [[1] * len(ids) for ids in batch_ids],
                }
        
        # Apply padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            batch_outputs = self.pad(
                batch_outputs,
                padding=padding_strategy,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_tensors=return_tensors,
                verbose=verbose,
            )
        
        return BatchEncoding(batch_outputs, tensor_type=return_tensors)
    
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs
    ) -> str:
        """
        Decode a sequence of token IDs to a string.
        
        Note: Replacement character handling is now automatic in the Rust implementation.
        The decode() method internally uses decode_advanced() when needed, providing
        optimal performance with automatic fallback to advanced decoding when replacement
        characters are detected.
        """
        # Simply call decode() - it now handles replacement characters automatically
        # by checking for � and falling back to decode_advanced() when needed
        return self._rust_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """
        Save the tokenizer vocabulary to a directory.
        
        For R-BPE, this copies the entire tokenizer directory structure.
        
        Returns:
            tuple: Paths to the saved vocabulary files.
        """
        import shutil
        from pathlib import Path
        
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the source directory (where the tokenizer was loaded from)
        source_dir = Path(self.name_or_path) if hasattr(self, 'name_or_path') and self.name_or_path else None
        
        if source_dir and source_dir.exists():
            # Copy the entire directory structure
            for subdir in ["new_tokenizer", "old_tokenizer", "metadata"]:
                src = source_dir / subdir
                dst = save_dir / subdir
                if src.exists():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            
            # Copy tokenization.py if it exists
            tokenization_file = source_dir / "tokenization.py"
            if tokenization_file.exists():
                shutil.copy2(tokenization_file, save_dir / "tokenization.py")
            
            return tuple()
        else:
            # Fallback: return empty tuple (tokenizer config will still be saved)
            import warnings
            warnings.warn(
                "Could not determine source directory for R-BPE tokenizer. "
                "Only tokenizer_config.json will be saved. "
                "To save the complete tokenizer, please ensure name_or_path is set correctly."
            )
            return tuple()


# For backward compatibility
class RBPETokenizerFast(RBPETokenizer):
    """Alias for RBPETokenizer (all R-BPE tokenizers use fast Rust backend)."""
    
    @property
    def is_fast(self) -> bool:
        """
        Returns True because this tokenizer uses a fast Rust backend.
        
        Note: HuggingFace's default is_fast property checks for PreTrainedTokenizerFast
        inheritance, but R-BPE uses its own Rust implementation (rbpe_tokenizers).
        We override this to accurately reflect that the tokenizer is fast.
        """
        return True
