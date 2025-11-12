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
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        unk_token=None,
        pad_token="<|finetune_right_pad_id|>",
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
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size.
        Note: This is the combined vocabulary size after mapping.
        """
        # R-BPE maintains compatibility with the old tokenizer's vocab space
        # The actual size is determined by the old tokenizer
        return 128256  # Llama 3 vocab size (adjust if using different base model)
    
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
    
    # def convert_tokens_to_string(self, tokens: List[str]) -> str:
    #     """Convert a sequence of tokens to a single string."""
    #     ids = [int(token) for token in tokens]
    #     return self._rust_tokenizer.decode(ids, skip_special_tokens=True)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens to a single string."""
        ids = []
        for token in tokens:
            # Skip special tokens that are not numeric
            if token.startswith('<') and token.endswith('>'):
                # Handle special tokens - try to map them to their IDs
                if token == '<BOS_TOKEN>' or token == self.bos_token:
                    if self.bos_token_id is not None:
                        ids.append(self.bos_token_id)
                elif token == '<EOS_TOKEN>' or token == self.eos_token:
                    if self.eos_token_id is not None:
                        ids.append(self.eos_token_id)
                elif token == '<PAD_TOKEN>' or token == self.pad_token:
                    if self.pad_token_id is not None:
                        ids.append(self.pad_token_id)
                elif token == '<UNK_TOKEN>' or token == self.unk_token:
                    if self.unk_token_id is not None:
                        ids.append(self.unk_token_id)
                # Otherwise skip unknown special tokens
                continue
            else:
                # Convert numeric token to int
                try:
                    ids.append(int(token))
                except ValueError:
                    # Skip tokens that can't be converted
                    continue
        
        if not ids:
            return ""
        
        return self._rust_tokenizer.decode(ids, skip_special_tokens=True)

        
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
        # Encode with Rust tokenizer
        if isinstance(text, str):
            input_ids = self._rust_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        else:
            # Already tokenized
            input_ids = text
        
        # Handle text pairs (for sentence pair tasks)
        if text_pair is not None:
            if isinstance(text_pair, str):
                pair_ids = self._rust_tokenizer.encode(text_pair, add_special_tokens=False)
            else:
                pair_ids = text_pair
            
            # Combine: [BOS] text [SEP] text_pair [EOS]
            if add_special_tokens:
                sep_token_id = self.eos_token_id  # Use EOS as separator
                input_ids = input_ids + [sep_token_id] + pair_ids + [self.eos_token_id]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Truncate if needed
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length is not None:
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
        
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
        
        # For single sequences with tensor output, wrap in a list to get 2D tensor
        if return_tensors is not None:
            encoded_inputs = {
                "input_ids": [input_ids],
                "attention_mask": [attention_mask],
            }
        
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
            # Use Rust batch encoding
            batch_ids = self._rust_tokenizer.encode_batch(
                batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens
            )
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
        """
        # Convert tensor to list if needed
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            # Handle single integer or other iterables
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            else:
                token_ids = list(token_ids)
        
        # Use advanced decoder for better handling of replacement characters
        use_advanced = kwargs.pop('use_advanced_decoder', True)
        
        if use_advanced:
            return self._rust_tokenizer.decode_advanced(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
        else:
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
